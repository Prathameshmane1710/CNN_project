from flask import Blueprint, request, jsonify, render_template
from app.predictor import predict_image

bp = Blueprint("routes", __name__)

@bp.route("/")
def home():
    return render_template("upload.html")

@bp.route("/health")
def upload_page():
    return "Defect detector is running."

@bp.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    label, prob = predict_image(file.stream)
    
    if prob is None:
        return jsonify({"prediction": label, "defective_probability": None})
    
    return jsonify({"prediction": label, "defective_probability": prob})