import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

model = models.resnet18()
model.fc = nn.Sequential(nn.Linear(512,2))

model.load_state_dict(torch.load("defect_model.pth", map_location="cpu"))

model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image_path,threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        prob_defective = torch.softmax(outputs,dim=1)[0,1].item()
    
    label = "Defective" if prob_defective>=threshold else "Ok"
    return label,prob_defective

if __name__ == "__main__":
    label,prob = predict("./test2.jpeg")
    print(f"Prediction: {label}  (defective probability: {prob:.3f})")