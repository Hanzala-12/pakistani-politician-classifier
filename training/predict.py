"""
predict.py – Single-image inference CLI for the Pakistani Politician Classifier.

Usage:
    python training/predict.py --image path/to/face.jpg
    python training/predict.py --image face.jpg --model resnet50 --checkpoint /kaggle/working/models/resnet50_best.pth
    python training/predict.py --image face.jpg --top-k 5

Supports both:
  - Generic CE models (resnet50, vgg16, convnext_base, …)
  - ArcFace embedding models (inception_resnet_v1, efficientnet_b3)
    using the arcface_eval dict stored in the checkpoint.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from training.config import config
from training.datasets import get_transforms
from training.models import EfficientNetEmbeddingModel, FaceEmbeddingModel, get_model

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model(model_name: str):
    """Instantiate the correct model class for *model_name*."""
    if "inception_resnet_v1" in model_name:
        pretrained = "casia-webface" if "casia" in model_name else "vggface2"
        return FaceEmbeddingModel(num_classes=config.NUM_CLASSES, pretrained=pretrained)
    if model_name == "efficientnet_b3":
        return EfficientNetEmbeddingModel(num_classes=config.NUM_CLASSES)
    return get_model(model_name, num_classes=config.NUM_CLASSES)


def load_model(model_name: str, checkpoint_path: str):
    """Load a trained model from *checkpoint_path*.

    Returns:
        model : model in eval mode on the correct device
    """
    model = _build_model(model_name)
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    # Attach ArcFace eval weights if available
    if isinstance(checkpoint, dict) and "arcface_eval" in checkpoint:
        model.arcface_eval = checkpoint["arcface_eval"]

    model = model.to(_device)
    model.eval()
    return model


def _embeddings_to_logits(model, embeddings: torch.Tensor) -> torch.Tensor:
    """Convert ArcFace embeddings → class logits using stored weights."""
    if hasattr(model, "arcface_eval") and isinstance(model.arcface_eval, dict):
        weight = model.arcface_eval.get("weight")
        scale = float(model.arcface_eval.get("scale", config.ARCFACE_SCALE))
        if weight is not None:
            weight = weight.to(embeddings.device)
            emb_norm = F.normalize(embeddings, dim=1)
            w_norm = F.normalize(weight, dim=1)
            return scale * (emb_norm @ w_norm.t())
    if hasattr(model, "head"):
        logits = model.head(embeddings)
        if logits.dim() == 2 and logits.size(1) == config.NUM_CLASSES:
            return logits
    raise RuntimeError(
        "Cannot reconstruct logits: no 'arcface_eval' dict or 'head' attribute found."
    )


def predict_image(model, image_path: str, top_k: int = 3):
    """Run inference on a single image.

    Returns:
        list of (class_name, confidence_float) tuples, length *top_k*
    """
    transform = get_transforms("val")
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        outputs = model(x)

    # Handle embedding-only models
    if isinstance(outputs, tuple):
        # (logits_or_none, embeddings) – e.g. FaceEmbeddingModel in training mode
        logits_candidate = outputs[0] if isinstance(outputs[0], torch.Tensor) else None
        embeddings = outputs[1] if len(outputs) > 1 else None
        if logits_candidate is None and embeddings is not None:
            logits = _embeddings_to_logits(model, embeddings)
        else:
            logits = logits_candidate
    else:
        if outputs.dim() == 2 and outputs.size(1) != config.NUM_CLASSES:
            logits = _embeddings_to_logits(model, outputs)
        else:
            logits = outputs

    probs = torch.softmax(logits, dim=1)[0]
    top_probs, top_indices = torch.topk(probs, min(top_k, config.NUM_CLASSES))

    return [(config.CLASS_NAMES[idx], prob.item()) for prob, idx in zip(top_probs, top_indices)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Predict Pakistani politician from a face image."
    )
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument(
        "--model", default="inception_resnet_v1",
        help="Model name (default: inception_resnet_v1)"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to .pth checkpoint. Defaults to OUTPUT_DIR/models/<model>_best.pth"
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions")
    args = parser.parse_args()

    # Resolve checkpoint
    checkpoint = args.checkpoint or str(
        Path(config.OUTPUT_DIR) / "models" / f"{args.model}_best.pth"
    )

    # Validate inputs
    if not Path(args.image).exists():
        print(f"Error: image not found: {args.image}")
        return
    if not Path(checkpoint).exists():
        print(f"Error: checkpoint not found: {checkpoint}")
        return

    print(f"Device   : {_device}")
    print(f"Model    : {args.model}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Image    : {args.image}\n")

    model = load_model(args.model, checkpoint)
    results = predict_image(model, args.image, top_k=args.top_k)

    print(f"Top-{args.top_k} Predictions:")
    print("-" * 42)
    for rank, (class_name, confidence) in enumerate(results, 1):
        bar = "█" * int(confidence * 20)
        print(f"  {rank}. {class_name:<26} {confidence*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
