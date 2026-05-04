import io
import time
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

import model_loader as model_loader
from model_loader import get_predictor

app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    # Accept either 'image' or legacy 'file' form field
    uploaded = None
    if "image" in request.files:
        uploaded = request.files["image"]
    elif "file" in request.files:
        uploaded = request.files["file"]
    else:
        return jsonify({"error": "No image file uploaded. Use form field 'image' or 'file'."}), 400

    if not uploaded or uploaded.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    model_key = request.form.get("model") or request.args.get("model")
    if model_key is not None:
        model_key = model_key.strip()
        if model_key == "":
            model_key = None

    try:
        image_bytes = uploaded.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file."}), 400

    try:
        predictor = get_predictor(model_key=model_key)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400

    result = predictor.predict_pil(image, top_k=3)

    def _normalize_result(r: dict) -> dict:
        if not isinstance(r, dict):
            return {"error": "Invalid predictor result."}
        if "error" in r:
            return {"error": r.get("error")}

        predicted = r.get("predicted_class") or r.get("predicted_label") or r.get("predicted")
        confidence = float(r.get("confidence") or 0.0)

        topk = r.get("top_k") or r.get("top3") or []
        predictions = []
        for item in topk:
            if not isinstance(item, dict):
                continue
            label = item.get("label") or item.get("class") or item.get("name")
            if label is None:
                continue
            conf_i = float(item.get("confidence", 0.0))
            predictions.append({"label": label, "confidence": conf_i})

        if not predictions and predicted:
            predictions = [{"label": predicted, "confidence": confidence}]

        return {
            "predicted_class": predicted,
            "confidence": confidence,
            "predictions": predictions,
            "top3": [{"class": p["label"], "confidence": p["confidence"]} for p in predictions],
        }

    normalized = _normalize_result(result)
    # Always return 200 with normalized JSON; frontend will inspect 'error' field if present
    return jsonify(normalized)


@app.route("/health", methods=["GET"])
def health():
    # Report device and loaded models
    try:
        loaded = []
        for key, predictor in getattr(model_loader, '_PREDICTORS', {}).items():
            try:
                loaded.append({
                    'cache_key': key,
                    'model_key': predictor.model_key,
                    'model_path': getattr(predictor, 'model_path', None),
                    'model_type': getattr(predictor, 'model_type', None),
                    'device': str(getattr(predictor, 'device', None)),
                })
            except Exception:
                loaded.append({'cache_key': key})
        available = list(getattr(model_loader, 'MODEL_CHOICES', {}).keys())
        device = None
        if loaded:
            device = loaded[0].get('device')
        else:
            # fallback to torch availability
            import torch
            device = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return jsonify({
            'status': 'ok',
            'device': device,
            'loaded_models': loaded,
            'available_models': available,
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': "No files uploaded. Use 'files' field for multiple images."}), 400

    model_key = request.form.get('model') or request.args.get('model')
    if model_key is not None:
        model_key = model_key.strip() or None

    try:
        predictor = get_predictor(model_key=model_key)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        return jsonify({'error': str(exc)}), 400

    results = []
    for uploaded in files:
        filename = getattr(uploaded, 'filename', None)
        try:
            image_bytes = uploaded.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception:
            results.append({'filename': filename, 'error': 'Invalid image file.'})
            continue

        start = time.time()
        try:
            r = predictor.predict_pil(image, top_k=3)
            # Normalize single predictor result into frontend-friendly shape
            if not isinstance(r, dict):
                results.append({'filename': filename, 'error': 'Invalid predictor response.'})
                continue
            if 'error' in r:
                results.append({'filename': filename, 'error': r.get('error')})
                continue

            predicted = r.get('predicted_label') or r.get('predicted_class') or r.get('predicted')
            confidence = float(r.get('confidence', 0.0))

            raw_topk = r.get('top_k') or r.get('top3') or []
            predictions = []
            for item in raw_topk[:3]:
                label = item.get('label') or item.get('name') or item.get('class')
                if label is None:
                    continue
                conf = float(item.get('confidence', 0.0))
                predictions.append({'label': label, 'confidence': conf})

            # Fallback: if no predictions but predicted present, create one
            if not predictions and predicted:
                predictions = [{'label': predicted, 'confidence': confidence}]

            inference_time_ms = int((time.time() - start) * 1000)
            results.append({
                'filename': filename,
                'predicted_class': predicted,
                'confidence': confidence,
                'predictions': predictions,
                'inference_time_ms': inference_time_ms,
            })
        except Exception as exc:
            results.append({'filename': filename, 'error': str(exc)})

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
