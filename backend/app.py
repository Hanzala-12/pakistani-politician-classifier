import io
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

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
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
