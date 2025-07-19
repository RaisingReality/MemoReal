import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import numpy as np
import trimesh
from google.cloud import storage
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

# --- Configuration ---
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'MemoReal-Bucket')

# --- App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Model and GCS Client Loading ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    conf = get_config("zoedepth", "infer")
    model = build_model(conf).to(device).eval()
    #model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True).to(device)
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
except Exception as e:
    model = None
    storage_client = None
    bucket = None
    print(f"Error during initialization: {e}")

# --- Helper Functions ---

def get_intrinsics(H,W):
    """
    Intrinsics for a pinhole camera model.
    Assume fov of 55 degrees and central principal point.
    """
    f = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]])

def depth_to_points(depth, R=None, t=None):

    K = get_intrinsics(depth.shape[1], depth.shape[2])
    Kinv = np.linalg.inv(K)
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    # M converts from your coordinate to PyTorch3D's coordinate system
    M = np.eye(3)
    M[0, 0] = -1.0
    M[1, 1] = -1.0

    height, width = depth.shape[1:3]

    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
    coord = coord.astype(np.float32)
    # coord = torch.as_tensor(coord, dtype=torch.float32, device=device)
    coord = coord[None]  # bs, h, w, 3

    D = depth[:, :, :, None, None]
    # print(D.shape, Kinv[None, None, None, ...].shape, coord[:, :, :, :, None].shape )
    pts3D_1 = D * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
    # pts3D_1 live in your coordinate system. Convert them to Py3D's
    pts3D_1 = M[None, None, None, ...] @ pts3D_1
    # from reference to targe tviewpoint
    pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]
    # pts3D_2 = pts3D_1
    # depth_2 = pts3D_2[:, :, :, 2, :]  # b,1,h,w
    return pts3D_2[:, :, :, :3, 0][0]

def pano_depth_to_world_points(depth):
    """
    360 depth to world points
    given 2D depth is an equirectangular projection of a spherical image
    Treat depth as radius
    longitude : -pi to pi
    latitude : -pi/2 to pi/2
    """

    # Convert depth to radius
    radius = depth.flatten()

    lon = np.linspace(-np.pi, np.pi, depth.shape[1])
    lat = np.linspace(-np.pi/2, np.pi/2, depth.shape[0])

    lon, lat = np.meshgrid(lon, lat)
    lon = lon.flatten()
    lat = lat.flatten()

    # Convert to cartesian coordinates
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)

    pts3d = np.stack([x, y, z], axis=1)

    return pts3d


def create_triangles(h, w, mask=None):
    """Creates mesh triangle indices from a given pixel grid size.
        This function is not and need not be differentiable as triangle indices are
        fixed.
    Args:
    h: (int) denoting the height of the image.
    w: (int) denoting the width of the image.
    Returns:
    triangles: 2D numpy array of indices (int) with shape (2(W-1)(H-1) x 3)
    """
    x, y = np.meshgrid(range(w - 1), range(h - 1))
    tl = y * w + x
    tr = y * w + x + 1
    bl = (y + 1) * w + x
    br = (y + 1) * w + x + 1
    triangles = np.array([tl, bl, tr, br, tr, bl])
    triangles = np.transpose(triangles, (1, 2, 0)).reshape(
        ((w - 1) * (h - 1) * 2, 3))
    if mask is not None:
        mask = mask.reshape(-1)
        triangles = triangles[mask[triangles].all(1)]
    return triangles

# --- API Endpoints ---
@app.route('/generate-scene', methods=['POST'])
def generate_scene():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and model and bucket:
        try:
            image_pil = Image.open(file.stream).convert("RGB")
            
            # 1. Run ZoeDepth inference
            with torch.no_grad():
                depth_tensor = model.infer_pil(image_pil)
            #depth_pil = Image.fromarray(depth_tensor).convert("L")

            pts3d = depth_to_points(depth_tensor[None])
            pts3d = pts3d.reshape(-1, 3)

            # Create a trimesh mesh from the points
            # Each pixel is connected to its 4 neighbors
            # colors are the RGB values of the image

            verts = pts3d.reshape(-1, 3)
            image = np.array(image)
            #if keep_edges:
            triangles = create_triangles(image.shape[0], image.shape[1])
            #else:
                #triangles = create_triangles(image.shape[0], image.shape[1], mask=~depth_edges_mask(depth))
            colors = image.reshape(-1, 3)
            mesh = trimesh.Trimesh(vertices=verts, faces=triangles, vertex_colors=colors)

            '''
            # 2. Prepare data for mesh creation
            image_pil = image_pil.resize(depth_pil.size)
            image_np = np.array(image_pil) / 255.0
            depth_np = np.array(depth_pil) / 255.0

            # 3. Create .glb mesh
            mesh = create_glb(image_np, depth_np)'''

            glb_data = mesh.export(file_type='glb')

            # 4. Upload to Google Cloud Storage
            unique_id = str(uuid.uuid4())
            blob_name = f"scene_{unique_id}.glb"
            blob = bucket.blob(blob_name)
            blob.upload_from_string(glb_data, content_type='model/gltf-binary')

            # 5. Return the public URL
            return jsonify({"downloadUrl": blob.public_url})

        except Exception as e:
            return jsonify({"error": f"An error occurred: {e}"}), 500
    
    return jsonify({"error": "Server not ready or misconfigured"}), 503

@app.route('/generate-360-scene', methods=['POST'])
def generate_360_scene():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and model and bucket:
        try:
            image_pil = Image.open(file.stream).convert("RGB")
            
            # 1. Run ZoeDepth inference
            with torch.no_grad():
                depth_tensor = model.infer_pil(image_pil)
            #depth_pil = Image.fromarray(depth_tensor).convert("L")

            pts3d = pano_depth_to_world_points(depth_tensor)

            # Create a trimesh mesh from the points
            # Each pixel is connected to its 4 neighbors
            # colors are the RGB values of the image

            verts = pts3d.reshape(-1, 3)
            image = np.array(image)
            #if keep_edges:
            triangles = create_triangles(image.shape[0], image.shape[1])
            #else:
                #triangles = create_triangles(image.shape[0], image.shape[1], mask=~depth_edges_mask(depth))
            colors = image.reshape(-1, 3)
            mesh = trimesh.Trimesh(vertices=verts, faces=triangles, vertex_colors=colors)

            '''
            # 2. Prepare data for mesh creation
            image_pil = image_pil.resize(depth_pil.size)
            image_np = np.array(image_pil) / 255.0
            depth_np = np.array(depth_pil) / 255.0

            # 3. Create .glb mesh
            mesh = create_glb(image_np, depth_np)'''

            glb_data = mesh.export(file_type='glb')

            # 4. Upload to Google Cloud Storage
            unique_id = str(uuid.uuid4())
            blob_name = f"scene_{unique_id}.glb"
            blob = bucket.blob(blob_name)
            blob.upload_from_string(glb_data, content_type='model/gltf-binary')

            # 5. Return the public URL
            return jsonify({"downloadUrl": blob.public_url})

        except Exception as e:
            return jsonify({"error": f"An error occurred: {e}"}), 500
    
    return jsonify({"error": "Server not ready or misconfigured"}), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)