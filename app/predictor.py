import torch
from PIL import Image
from app.model_loader import load_model, preprocess
import numpy as np

model = load_model()

def looks_like_casting(image):
    arr = np.array(image).astype(float)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    color_spread = (abs(r - g).mean() + abs(g - b).mean() + abs(r - b).mean()) / 3

    return color_spread < 5

def predict_image(file_stream, threshold=0.5):
    image = Image.open(file_stream).convert("RGB")

    if not looks_like_casting(image):
        return "Not a casting", None
    
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        prob_defective = torch.softmax(outputs, dim=1)[0, 1].item()

    label = "Defective" if prob_defective >= threshold else "OK"
    return label, round(prob_defective, 3)



