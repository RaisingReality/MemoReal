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
# AWS S3 bucket configuration. These should be set as environment variables.
# Note: AWS credentials (access key and secret key) are not explicitly set here.
# Boto3 will automatically look for them in standard locations, such as
# environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) or an IAM role
# if running on an EC2 instance.
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
AWS_REGION = os.environ.get('AWS_REGION')

# --- App Initialization ---
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16 MB

# --- Model and S3 Client Loading ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

mesh_model = None
s3_client = None
point_cloud_model = None

try:
    # Load the ZoeDepth model
    conf = get_config("zoedepth", "infer")
    mesh_model = build_model(conf).to(device).eval()

    # Load the UniK3D model
    name = f"unik3d-vitl"
    point_cloud_model = UniK3D.from_pretrained(f"lpiccinelli/{name}")
    # Set resolution level and interpolation mode as specified.
    point_cloud_model.resolution_level = 9
    point_cloud_model.interpolation_mode = "bilinear"
    point_cloud_model = point_cloud_model.to(device).eval()

    # Initialize the S3 client
    if S3_BUCKET_NAME and AWS_REGION:
        print(f"Initializing S3 client for bucket '{S3_BUCKET_NAME}' in region '{AWS_REGION}'")
        s3_client = boto3.client('s3', region_name=AWS_REGION)
    else:
        print("Warning: S3_BUCKET_NAME and/or AWS_REGION environment variables not set. S3 functionality will be disabled.")

except Exception as e:
    print(f"Error during initialization: {e}")

# --- Utility Functions ---

def pano_depth_to_world_points(depth):
    """Converts a panoramic depth map to a 3D point cloud."""
    radius = depth.flatten()
    lon = np.linspace(-np.pi, np.pi, depth.shape[1])
    lat = np.linspace(-np.pi/2, np.pi/2, depth.shape[0])
    lon, lat = np.meshgrid(lon, lat)
    x = radius * np.cos(lat.flatten()) * np.cos(lon.flatten())
    y = radius * np.cos(lat.flatten()) * np.sin(lon.flatten())
    z = radius * np.sin(lat.flatten())
    return np.stack([x, y, z], axis=1)

def create_triangles(h, w, mask=None):
    """Creates a mesh of triangles for a given height and width."""
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

def predictions_to_glb(
    predictions,
    mask_black_bg=False,
    mask_far_points=False,
) -> trimesh.Scene:
    print("Building GLB scene")
    images = predictions["image"].squeeze().permute(1, 2, 0).cpu().numpy()
    world_points = predictions["points"].squeeze().permute(1, 2, 0).cpu().numpy()

    vertices_3d = world_points.reshape(-1, 3)
    # flip x and y
    vertices_3d[:, 1] *= -1
    vertices_3d[:, 0] *= -1
    colors_rgb = (images.reshape(-1, 3)).astype(np.uint8)

    if mask_black_bg:
        black_bg_mask = colors_rgb.sum(axis=1) >= 16
        vertices_3d = vertices_3d[black_bg_mask]
        colors_rgb = colors_rgb[black_bg_mask]

    if mask_far_points:
        far_points_mask = np.linalg.norm(vertices_3d, axis=-1) < 100.0
        vertices_3d = vertices_3d[far_points_mask]
        colors_rgb = colors_rgb[far_points_mask]

    scene_3d = trimesh.Scene()
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)
    scene_3d.add_geometry(point_cloud_data)

    return scene_3d


# --- Health Check Route ---
@app.route('/')
def home():
    return "UniK3D and ZoeDepth 3D Scene Generator (AWS Version) is running"

# --- Standard Scene Generation ---
@app.route('/generate-scene', methods=['POST'])
def generate_scene():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not point_cloud_model or not s3_client:
        return jsonify({"error": "Server not ready or misconfigured. Check model and S3 configuration."}), 503

    try:
      
        gc.collect()

        input_image = np.array(Image.open(file.stream))
        image_tensor = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).float()
        device = next(point_cloud_model.parameters()).device
        image_tensor = image_tensor.to(device)


        # Perform inference with the model.
        print("Running UniK3D inference...")
        outputs = point_cloud_model.infer(image_tensor, camera=None, normalize=True)
        outputs["image"] = image_tensor

        # Convert predictions to GLB
        glbscene = predictions_to_glb(
            outputs,
            mask_black_bg=False,
            mask_far_points=False,
        )
        glb_data = glbscene.export(file_type='glb')

        # Cleanup
        del outputs
        gc.collect()

        unique_id = str(uuid.uuid4())
        # This is the object key in the S3 bucket
        object_name = f"scene_{unique_id}.glb"

        # Upload the file to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=object_name,
            Body=glb_data,
            ContentType='model/gltf-binary',
            ACL='public-read'  # Makes the object publicly accessible via URL
        )

        print(f"Uploaded to S3: {S3_BUCKET_NAME}/{object_name}")
        
        # Construct the public URL for the object
        public_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{object_name}"

        print("Public URL : ", public_url)

        return jsonify({"glb_url": public_url})

    except NoCredentialsError:
        return jsonify({"error": "AWS credentials not found. Configure credentials or IAM role."}), 500
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500


# --- 360° Scene Generation ---
@app.route('/generate-360-scene', methods=['POST'])
def generate_360_scene():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not mesh_model or not s3_client:
        return jsonify({"error": "Server not ready or misconfigured. Check model and S3 configuration."}), 503

    try:
        gc.collect()

        image_pil = Image.open(file.stream).convert("RGB")

        print("Running ZoeDepth inference...")
        with torch.no_grad():
            depth_tensor = mesh_model.infer_pil(image_pil)

        pts3d = pano_depth_to_world_points(depth_tensor)
        image = np.array(image_pil)
        triangles = create_triangles(image.shape[0], image.shape[1])
        colors = image.reshape(-1, 3)

        mesh = trimesh.Trimesh(vertices=pts3d, faces=triangles, vertex_colors=colors)
        glb_data = mesh.export(file_type='glb')

        # Cleanup
        del depth_tensor, pts3d, image, triangles, colors, mesh
        gc.collect()

        unique_id = str(uuid.uuid4())
        # This is the object key in the S3 bucket
        object_name = f"scene_{unique_id}.glb"

        # Upload the file to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=object_name,
            Body=glb_data,
            ContentType='model/gltf-binary',
            ACL='public-read'  # Makes the object publicly accessible via URL
        )

        print(f"Uploaded to S3: {S3_BUCKET_NAME}/{object_name}")

        # Construct the public URL for the object
        public_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{object_name}"

        print("Public URL : ", public_url)

        return jsonify({"glb_url": public_url})

    except NoCredentialsError:
        return jsonify({"error": "AWS credentials not found. Configure credentials or IAM role."}), 500
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

# --- Run App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
