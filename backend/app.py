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
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded."}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    model_key = request.form.get("model") or request.args.get("model")
    if model_key is not None:
        model_key = model_key.strip()
        if model_key == "":
            model_key = None

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file."}), 400

    try:
        predictor = get_predictor(model_key=model_key)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400
    result = predictor.predict_pil(image, top_k=3)
    if "error" in result:
        return jsonify(result), 400

    return jsonify(result)


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
            if 'error' in r:
                results.append({'filename': filename, 'error': r.get('error')})
                continue

            top3 = []
            for item in r.get('top_k', [])[:3]:
                label = item.get('label') or item.get('name') or item.get('class')
                conf = float(item.get('confidence', 0.0))
                top3.append({'class': label, 'confidence': conf})

            inference_time_ms = int((time.time() - start) * 1000)
            results.append({
                'filename': filename,
                'predicted_class': r.get('predicted_label') or r.get('predicted_class'),
                'confidence': float(r.get('confidence', 0.0)),
                'top3': top3,
                'inference_time_ms': inference_time_ms,
            })
        except Exception as exc:
            results.append({'filename': filename, 'error': str(exc)})

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
