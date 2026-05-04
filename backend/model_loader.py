import glob
import math
import os
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import models, transforms

ARC_IMAGE_SIZE = 336
ARC_MEAN = [0.5, 0.5, 0.5]
ARC_STD = [0.5, 0.5, 0.5]
CLS_IMAGE_SIZE = 224
CLS_MEAN = [0.485, 0.456, 0.406]
CLS_STD = [0.229, 0.224, 0.225]
MARGIN_RATIO = 0.2

MODEL_CHOICES = {
    "inception_resnet_v1": {
        "filename": "inception_resnet_v1_best.pth",
        "pretrained": "vggface2",
        "type": "arcface",
    },
    "inception_resnet_v1_casia": {
        "filename": "inception_resnet_v1_casia_best.pth",
        "pretrained": "casia-webface",
        "type": "arcface",
    },
    "resnet50": {
        "filename": "resnet50_best.pth",
        "type": "classifier",
        "arch": "resnet50",
    },
}


def rotate_point(x: float, y: float, cx: float, cy: float, angle_rad: float):
    x -= cx
    y -= cy
    xr = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    yr = x * math.sin(angle_rad) + y * math.cos(angle_rad)
    return xr + cx, yr + cy


def align_face_with_landmarks(
    img: Image.Image,
    box,
    landmarks,
    image_size: int = ARC_IMAGE_SIZE,
    margin_ratio: float = MARGIN_RATIO,
) -> Optional[Image.Image]:
    left_eye = landmarks[0]
    right_eye = landmarks[1]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))
    eye_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)

    rotated = img.rotate(-angle, resample=Image.BICUBIC, center=eye_center, expand=False)

    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    angle_rad = -math.radians(angle)
    rotated_corners = [
        rotate_point(x, y, eye_center[0], eye_center[1], angle_rad) for x, y in corners
    ]

    xs = [p[0] for p in rotated_corners]
    ys = [p[1] for p in rotated_corners]
    x1r, x2r = min(xs), max(xs)
    y1r, y2r = min(ys), max(ys)

    w = x2r - x1r
    h = y2r - y1r
    if w <= 0 or h <= 0:
        return None

    side = max(w, h)
    pad = side * margin_ratio
    side = side + 2 * pad

    cx = (x1r + x2r) / 2.0
    cy = (y1r + y2r) / 2.0

    x1c = cx - side / 2.0
    y1c = cy - side / 2.0
    x2c = cx + side / 2.0
    y2c = cy + side / 2.0

    x1c = max(0, x1c)
    y1c = max(0, y1c)
    x2c = min(rotated.width, x2c)
    y2c = min(rotated.height, y2c)

    if x2c <= x1c or y2c <= y1c:
        return None

    face = rotated.crop((x1c, y1c, x2c, y2c)).resize((image_size, image_size), Image.BICUBIC)
    return face


def candidate_model_dirs() -> List[str]:
    candidates = []
    env_dir = os.getenv("MODEL_DIR")
    if env_dir:
        candidates.append(env_dir)
    candidates.extend([
        "project_outputs/models",
        "/kaggle/working/models",
        "models",
        os.path.join(os.getcwd(), "models"),
    ])
    return [c for c in candidates if c]


def infer_pretrained(model_key: Optional[str], model_path: str) -> str:
    if model_key in MODEL_CHOICES:
        return MODEL_CHOICES[model_key].get("pretrained", "vggface2")
    if "casia" in os.path.basename(model_path).lower():
        return "casia-webface"
    return "vggface2"


def resolve_model_path(model_key: Optional[str] = None) -> str:
    env_path = os.getenv("MODEL_PATH")

    if model_key:
        if os.path.exists(model_key):
            return model_key
        model_info = MODEL_CHOICES.get(model_key)
        if model_info is None:
            available = ", ".join(sorted(MODEL_CHOICES.keys()))
            raise ValueError(f"Unknown model '{model_key}'. Available: {available}")
        filename = model_info["filename"]
        for model_dir in candidate_model_dirs():
            if not os.path.exists(model_dir):
                continue
            path = os.path.join(model_dir, filename)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"Model file '{filename}' not found. Set MODEL_DIR or MODEL_PATH."
        )

    if env_path and os.path.exists(env_path):
        return env_path

    for model_dir in candidate_model_dirs():
        if not model_dir or not os.path.exists(model_dir):
            continue
        for model_info in MODEL_CHOICES.values():
            path = os.path.join(model_dir, model_info["filename"])
            if os.path.exists(path):
                return path
        matches = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
        if matches:
            return matches[0]

    raise FileNotFoundError("No .pth checkpoint found. Set MODEL_PATH or MODEL_DIR.")


class FaceEmbeddingModel(nn.Module):
    def __init__(self, num_classes: int, pretrained: str = "vggface2", dropout: float = 0.5, use_arcface: bool = True):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained=pretrained, classify=False, device=None)
        self.embedding_size = 512
        self.use_arcface = use_arcface
        if not use_arcface:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.embedding_size, num_classes),
            )
        else:
            self.head = nn.Identity()

    def forward(self, x, return_embeddings: bool = False):
        embeddings = self.backbone(x)
        if self.use_arcface:
            return embeddings
        if return_embeddings:
            return embeddings
        logits = self.head(embeddings)
        return logits, embeddings


def build_classifier_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet50":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported classifier architecture '{arch}'.")


class ModelPredictor:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        model_key: Optional[str] = None,
    ):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path or resolve_model_path(model_key)
        self.model_key = model_key

        checkpoint = torch.load(self.model_path, map_location=self.device)
        class_names = checkpoint.get("class_names")
        if not class_names:
            raise RuntimeError("Checkpoint missing class_names.")

        self.class_names = class_names
        model_info = MODEL_CHOICES.get(model_key) if model_key else None
        model_type = model_info.get("type") if model_info else None
        if model_type is None:
            model_type = "arcface" if isinstance(checkpoint.get("arcface_eval"), dict) else "classifier"

        self.model_type = model_type
        self.arcface_eval = None

        if self.model_type == "arcface":
            self.arcface_eval = checkpoint.get("arcface_eval")
            if not isinstance(self.arcface_eval, dict):
                raise RuntimeError("Checkpoint missing arcface_eval.")

            pretrained = infer_pretrained(model_key, self.model_path)
            self.model = FaceEmbeddingModel(
                num_classes=len(class_names),
                pretrained=pretrained,
                use_arcface=True,
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            self.mtcnn = MTCNN(keep_all=True, device=self.device)
            self.transform = transforms.Compose([
                transforms.Resize(ARC_IMAGE_SIZE),
                transforms.CenterCrop(ARC_IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=ARC_MEAN, std=ARC_STD),
            ])
        else:
            if not model_info or "arch" not in model_info:
                raise ValueError(
                    "Classifier checkpoints require a model key with a known architecture."
                )
            self.model = build_classifier_model(model_info["arch"], num_classes=len(class_names))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            self.mtcnn = None
            self.transform = transforms.Compose([
                transforms.Resize(CLS_IMAGE_SIZE),
                transforms.CenterCrop(CLS_IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=CLS_MEAN, std=CLS_STD),
            ])

    def align_face(self, image: Image.Image) -> Optional[Image.Image]:
        if self.mtcnn is None:
            return None
        boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
        if boxes is None or landmarks is None:
            return None

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = int(np.argmax(areas))
        return align_face_with_landmarks(image, boxes[idx], landmarks[idx], image_size=ARC_IMAGE_SIZE)

    def embeddings_to_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.arcface_eval is None:
            raise RuntimeError("arcface_eval missing for arcface model.")
        weight = self.arcface_eval.get("weight")
        scale = float(self.arcface_eval.get("scale", 64.0))
        if weight is None:
            raise RuntimeError("arcface_eval weight missing.")
        # Ensure weight is a torch tensor on the correct device
        if not isinstance(weight, torch.Tensor):
            try:
                weight = torch.as_tensor(weight, device=embeddings.device)
            except Exception:
                # Fallback: move device after tensor conversion
                weight = torch.tensor(weight).to(embeddings.device)
        else:
            weight = weight.to(embeddings.device)
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(weight, dim=1)
        return scale * (embeddings @ weight.t())

    def predict_pil(self, image: Image.Image, top_k: int = 3) -> Dict:
        if self.model_type == "arcface":
            aligned = self.align_face(image)
            if aligned is None:
                return {"error": "No face detected."}
            tensor = self.transform(aligned).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embeddings = self.model(tensor)
                if isinstance(embeddings, tuple):
                    embeddings = embeddings[-1]
                logits = self.embeddings_to_logits(embeddings)
                probs = torch.softmax(logits, dim=1).squeeze(0)
        else:
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0)

        top_k = min(top_k, probs.numel())
        confs, indices = torch.topk(probs, k=top_k)

        top_preds = []
        for conf, idx in zip(confs.tolist(), indices.tolist()):
            top_preds.append({"label": self.class_names[idx], "confidence": float(conf)})

        return {
            "predicted_label": top_preds[0]["label"],
            "confidence": top_preds[0]["confidence"],
            "top_k": top_preds,
        }


_PREDICTORS: Dict[str, ModelPredictor] = {}


def get_predictor(model_key: Optional[str] = None) -> ModelPredictor:
    resolved_path = resolve_model_path(model_key)
    cache_key = model_key or resolved_path
    if cache_key not in _PREDICTORS:
        _PREDICTORS[cache_key] = ModelPredictor(
            model_path=resolved_path,
            model_key=model_key,
        )
    return _PREDICTORS[cache_key]


def predict(image_path: str, model_key: Optional[str] = None) -> Dict:
    image = Image.open(image_path).convert("RGB")
    return get_predictor(model_key=model_key).predict_pil(image, top_k=3)
