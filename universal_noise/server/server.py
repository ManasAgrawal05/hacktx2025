import numpy as np
import os
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1
from huggingface_hub import hf_hub_download
from io import BytesIO
import torch.nn.functional as F
from flask import jsonify, Flask, request
app = Flask(__name__)




def try_load_state_dict(model, ckpt_path):
    try:
        obj = torch.load(ckpt_path, map_location="cpu")
        if isinstance(obj, dict):
            for key in ["state_dict", "model_state_dict", "model", "weights"]:
                if key in obj and isinstance(obj[key], dict):
                    model.load_state_dict(obj[key], strict=False)
                    return True
            try:
                model.load_state_dict(obj, strict=False)
                return True
            except Exception:
                pass
        model.load_state_dict(obj, strict=False)
        return True
    except Exception as e:
        print(f"[warn] Failed to load state_dict from '{ckpt_path}': {e}")
        return False

def load_facenet_from_hf(device='cpu', repo_id=None, filename=None, fallback=True):
    # baseline model (will be overwritten if HF checkpoint loaded)
    model = InceptionResnetV1(pretrained="vggface2", classify=False)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    # try HF candidates if provided
    candidates = []
    if repo_id and filename:
        candidates.append((repo_id, filename))
    candidates.extend([
        ("py-feat/facenet", "facenet_20180402_114759_vggface2.pth"),
        ("lllyasviel/Annotators", "facenet.pth"),
    ])

    for rid, fname in candidates:
        try:
            print(f"Trying HF repo '{rid}' file '{fname}'...")
            ckpt_path = hf_hub_download(repo_id=rid, filename=fname)
            if try_load_state_dict(model, ckpt_path):
                print(f"Loaded FaceNet weights from: {rid}/{fname}")
                return model
        except Exception as e:
            print(f"Could not fetch {rid}/{fname}: {e}")

    if fallback:
        print("Falling back to facenet-pytorch pretrained='vggface2' model.")
        model = InceptionResnetV1(pretrained="vggface2", classify=False).to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    raise RuntimeError("Failed to load FaceNet model")

def preprocess_for_facenet_tensor(x: torch.Tensor):
    # x: (B, C, H, W) in [0,1] floats
    # Resize to 160x160 and map to [-1, 1]
    x_resized = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)
    return x_resized * 2.0 - 1.0

def pil_image_to_tensor(img_pil: Image.Image) -> torch.Tensor:
    """Converts a PIL.Image (RGB) to a torch tensor in shape (C, H, W), float32 in [0,1]."""
    img = img_pil.convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0   # H x W x C
    # convert to C x H x W
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return t


@app.route('/inference', methods=['POST'])
def infer():
    # Accept a few possible keys for flexibility:
    if "original" not in request.files or "modified" not in request.files:
        return jsonify({
            "error": "Please upload two images with keys 'original' and 'modified'."
        }), 400

    # Read files
    try:
        pil_orig = Image.open(BytesIO(request.files["original"].read()))
        pil_mod = Image.open(BytesIO(request.files["modified"].read()))
    except Exception as e:
        return jsonify({"error": f"Failed to read uploaded images: {str(e)}"}), 400
    
    try:
        t1 = pil_image_to_tensor(pil_orig).unsqueeze(0).to(device=device, dtype=torch.float32)  # 1,C,H,W
        t2 = pil_image_to_tensor(pil_mod).unsqueeze(0).to(device=device, dtype=torch.float32)
    except Exception as e:
        return jsonify({"error": f"Failed to convert images to tensors: {str(e)}"}), 500

    # Preprocess same as training (resizing + [-1,1])
    inp1 = preprocess_for_facenet_tensor(t1)
    inp2 = preprocess_for_facenet_tensor(t2)

    # Run model (FaceNet) to get embeddings
    model.eval()
    with torch.no_grad():
        e1 = model(inp1)   # shape (1, embedding_dim)
        e2 = model(inp2)

    # Compute metrics: L2 distance and cosine similarity
    # L2:
    l2_dist = torch.norm(e1 - e2, p=2, dim=1).cpu().item()
    # Cosine similarity (1 means identical direction)
    cos_sim = F.cosine_similarity(e1, e2, dim=1).cpu().item()

    # Optionally return embeddings (as lists) for downstream use
    emb1 = e1.squeeze(0).cpu().numpy().tolist()
    emb2 = e2.squeeze(0).cpu().numpy().tolist()

    response = {
        "l2_distance": float(l2_dist),
        "cosine_similarity": float(cos_sim),
        "embedding_dim": len(emb1),
        "embedding_1": emb1,
        "embedding_2": emb2,
        # If you want a more direct "fooled" scalar, you can expose l2_distance.
        # Higher L2 => model more fooled.
        "fooled_score": float(l2_dist)
    }

    return jsonify(response), 200


# start the development server
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_facenet_from_hf(device=device)
print("Saved recognizer to facenet_recognizer.pt")
