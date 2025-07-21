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
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'memoreal-bucket')

# --- App Initialization ---
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16 MB

# --- Model and GCS Client Loading ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    conf = get_config("zoedepth", "infer")
    model = build_model(conf).to(device).eval()
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
except Exception as e:
    model = None
    storage_client = None
    bucket = None
    print(f"Error during initialization: {e}")

# --- Utility Functions ---
def get_intrinsics(H, W):
    f = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

def depth_to_points(depth, R=None, t=None):
    K = get_intrinsics(depth.shape[1], depth.shape[2])
    Kinv = np.linalg.inv(K)
    R = np.eye(3) if R is None else R
    t = np.zeros(3) if t is None else t
    M = np.diag([-1.0, -1.0, 1.0])

    height, width = depth.shape[1:3]
    x, y = np.arange(width), np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[..., :1]), -1).astype(np.float32)
    coord = coord[None]

    D = depth[:, :, :, None, None]
    pts3D_1 = D * Kinv[None, None, None, ...] @ coord[..., None]
    pts3D_1 = M[None, None, None, ...] @ pts3D_1
    pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]
    return pts3D_2[..., 0][0]

def pano_depth_to_world_points(depth):
    radius = depth.flatten()
    lon = np.linspace(-np.pi, np.pi, depth.shape[1])
    lat = np.linspace(-np.pi/2, np.pi/2, depth.shape[0])
    lon, lat = np.meshgrid(lon, lat)
    x = radius * np.cos(lat.flatten()) * np.cos(lon.flatten())
    y = radius * np.cos(lat.flatten()) * np.sin(lon.flatten())
    z = radius * np.sin(lat.flatten())
    return np.stack([x, y, z], axis=1)

def create_triangles(h, w, mask=None):
    x, y = np.meshgrid(range(w - 1), range(h - 1))
    tl = y * w + x
    tr = y * w + x + 1
    bl = (y + 1) * w + x
    br = (y + 1) * w + x + 1
    triangles = np.array([tl, bl, tr, br, tr, bl])
    triangles = np.transpose(triangles, (1, 2, 0)).reshape(-1, 3)
    if mask is not None:
        mask = mask.reshape(-1)
        triangles = triangles[mask[triangles].all(1)]
    return triangles

# --- Health Check Route ---
@app.route('/')
def home():
    return "ZoeDepth 3D Scene Generator is running"

# --- Standard Scene Generation ---
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
            with torch.no_grad():
                depth_tensor = model.infer_pil(image_pil)

            pts3d = depth_to_points(depth_tensor[None]).reshape(-1, 3)
            image = np.array(image_pil)
            triangles = create_triangles(image.shape[0], image.shape[1])
            colors = image.reshape(-1, 3)

            mesh = trimesh.Trimesh(vertices=pts3d, faces=triangles, vertex_colors=colors)
            glb_data = mesh.export(file_type='glb')

            unique_id = str(uuid.uuid4())
            blob_name = f"scene_{unique_id}.glb"
            blob = bucket.blob(blob_name)
            blob.upload_from_string(glb_data, content_type='model/gltf-binary')

            return jsonify({"glb_url": blob.public_url})
        except Exception as e:
            return jsonify({"error": f"An error occurred: {e}"}), 500

    return jsonify({"error": "Server not ready or misconfigured"}), 503

# --- 360° Scene Generation ---
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
            with torch.no_grad():
                depth_tensor = model.infer_pil(image_pil)

            pts3d = pano_depth_to_world_points(depth_tensor)
            image = np.array(image_pil)
            triangles = create_triangles(image.shape[0], image.shape[1])
            colors = image.reshape(-1, 3)

            mesh = trimesh.Trimesh(vertices=pts3d, faces=triangles, vertex_colors=colors)
            glb_data = mesh.export(file_type='glb')

            unique_id = str(uuid.uuid4())
            blob_name = f"scene_{unique_id}.glb"
            blob = bucket.blob(blob_name)
            blob.upload_from_string(glb_data, content_type='model/gltf-binary')

            return jsonify({"glb_url": blob.public_url})
        except Exception as e:
            return jsonify({"error": f"An error occurred: {e}"}), 500

    return jsonify({"error": "Server not ready or misconfigured"}), 503

# --- Run App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
