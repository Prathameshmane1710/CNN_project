from flask import Flask,request,jsonify
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

model = models.resnet18()
model.fc = nn.Sequential(nn.Linear(512,2))


model.load_state_dict(torch.load("defect_model.pth",map_location="cpu"))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route("/")
def home():
    return "Defect detector is running"

@app.route("/predict",methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        prob_defective = torch.softmax(outputs, dim=1)[0, 1].item()

    label = "Defective" if prob_defective >= 0.5 else "OK"
    return jsonify({
        "prediction": label,
        "defective_probability": round(prob_defective, 3)
    })
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)