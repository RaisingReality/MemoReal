import os
import gc
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import numpy as np
import trimesh
import boto3
from botocore.exceptions import NoCredentialsError
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from unik3d.models import UniK3D

# --- Configuration ---
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
AWS_REGION = os.environ.get('AWS_REGION')

# --- App Initialization ---
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# --- Model and S3 Client Loading ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- IMPROVEMENT: Lazy Loading with Unloading ---
# We initialize models to None. They will be loaded on-demand,
# and the other model will be unloaded to save memory.
mesh_model = None
point_cloud_model = None
s3_client = None

# Initialize S3 client at startup
try:
    if S3_BUCKET_NAME and AWS_REGION:
        print(f"Initializing S3 client for bucket '{S3_BUCKET_NAME}' in region '{AWS_REGION}'")
        s3_client = boto3.client('s3', region_name=AWS_REGION)
    else:
        print("Warning: S3_BUCKET_NAME and/or AWS_REGION environment variables not set. S3 functionality will be disabled.")
except Exception as e:
    print(f"Error during S3 client initialization: {e}")


def get_point_cloud_model():
    """Loads the UniK3D model, unloading the other model first if necessary."""
    global point_cloud_model, mesh_model
    # --- IMPROVEMENT: Unload the other model ---
    if mesh_model is not None and mesh_model != "ERROR":
        print("Unloading ZoeDepth (mesh) model to free memory...")
        del mesh_model
        mesh_model = None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if point_cloud_model is None:
        print("Lazy loading UniK3D model...")
        try:
            name = f"unik3d-vitl"
            point_cloud_model = UniK3D.from_pretrained(f"lpiccinelli/{name}")
            point_cloud_model.resolution_level = 9
            point_cloud_model.interpolation_mode = "bilinear"
            point_cloud_model = point_cloud_model.to(device).eval()
            print("UniK3D model loaded successfully.")
        except Exception as e:
            print(f"Error loading UniK3D model: {e}")
            point_cloud_model = "ERROR"
    return point_cloud_model

def get_mesh_model():
    """Loads the ZoeDepth model, unloading the other model first if necessary."""
    global mesh_model, point_cloud_model
    # --- IMPROVEMENT: Unload the other model ---
    if point_cloud_model is not None and point_cloud_model != "ERROR":
        print("Unloading UniK3D (point cloud) model to free memory...")
        del point_cloud_model
        point_cloud_model = None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if mesh_model is None:
        print("Lazy loading ZoeDepth model...")
        try:
            conf = get_config("zoedepth", "infer")
            mesh_model = build_model(conf).to(device).eval()
            print("ZoeDepth model loaded successfully.")
        except Exception as e:
            print(f"Error loading ZoeDepth model: {e}")
            mesh_model = "ERROR"
    return mesh_model

# --- Utility Functions ---
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
    tl, tr, bl, br = y * w + x, y * w + x + 1, (y + 1) * w + x, (y + 1) * w + x + 1
    triangles = np.array([tl, bl, tr, br, tr, bl]).transpose((1, 2, 0)).reshape(-1, 3)
    if mask is not None:
        mask = mask.reshape(-1)
        triangles = triangles[mask[triangles].all(1)]
    return triangles

def predictions_to_glb(predictions, mask_black_bg=False, mask_far_points=False):
    print("Building GLB scene")
    images = predictions["image"].squeeze().permute(1, 2, 0).cpu().numpy()
    world_points = predictions["points"].squeeze().permute(1, 2, 0).cpu().numpy()
    vertices_3d = world_points.reshape(-1, 3)
    vertices_3d[:, 1] *= -1
    vertices_3d[:, 0] *= -1
    colors_rgb = (images.reshape(-1, 3)).astype(np.uint8)
    if mask_black_bg:
        black_bg_mask = colors_rgb.sum(axis=1) >= 16
        vertices_3d, colors_rgb = vertices_3d[black_bg_mask], colors_rgb[black_bg_mask]
    if mask_far_points:
        far_points_mask = np.linalg.norm(vertices_3d, axis=-1) < 100.0
        vertices_3d, colors_rgb = vertices_3d[far_points_mask], colors_rgb[far_points_mask]
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)
    return trimesh.Scene(point_cloud_data)

# --- Health Check Route ---
@app.route('/')
def home():
    return "UniK3D and ZoeDepth 3D Scene Generator (AWS Version) is running"

# --- Standard Scene Generation ---
@app.route('/generate-scene', methods=['POST'])
def generate_scene():
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    if not s3_client: return jsonify({"error": "S3 client not configured."}), 503

    model = get_point_cloud_model()
    if model == "ERROR" or model is None:
        return jsonify({"error": "Point cloud model could not be loaded."}), 503

    try:
        input_image = np.array(Image.open(file.stream).convert("RGB"))
        image_tensor = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).float().to(device)

        print("Running UniK3D inference...")
        with torch.no_grad():
            outputs = model.infer(image_tensor, camera=None, normalize=True)
        
        outputs["image"] = image_tensor
        glbscene = predictions_to_glb(outputs, mask_black_bg=False, mask_far_points=False)
        glb_data = glbscene.export(file_type='glb')

        del input_image, image_tensor, outputs, glbscene
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        unique_id = str(uuid.uuid4())
        object_name = f"scene_{unique_id}.glb"
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=object_name, Body=glb_data, ContentType='model/gltf-binary')
        public_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{object_name}"
        
        print(f"Successfully processed and uploaded to {public_url}")
        return jsonify({"glb_url": public_url})

    except Exception as e:
        print(f"An error occurred in /generate-scene: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500

# --- 360° Scene Generation ---
@app.route('/generate-360-scene', methods=['POST'])
def generate_360_scene():
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    if not s3_client: return jsonify({"error": "S3 client not configured."}), 503

    model = get_mesh_model()
    if model == "ERROR" or model is None:
        return jsonify({"error": "Mesh model could not be loaded."}), 503

    try:
        image_pil = Image.open(file.stream).convert("RGB")
        
        print("Running ZoeDepth inference...")
        with torch.no_grad():
            depth_tensor = model.infer_pil(image_pil)

        pts3d = pano_depth_to_world_points(depth_tensor.cpu().numpy())
        image = np.array(image_pil)
        triangles = create_triangles(image.shape[0], image.shape[1])
        colors = image.reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=pts3d, faces=triangles, vertex_colors=colors)
        glb_data = mesh.export(file_type='glb')

        del image_pil, depth_tensor, pts3d, image, triangles, colors, mesh
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        unique_id = str(uuid.uuid4())
        object_name = f"scene_{unique_id}.glb"
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=object_name, Body=glb_data, ContentType='model/gltf-binary')
        public_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{object_name}"

        print(f"Successfully processed and uploaded to {public_url}")
        return jsonify({"glb_url": public_url})

    except Exception as e:
        print(f"An error occurred in /generate-360-scene: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500

# --- Run App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
