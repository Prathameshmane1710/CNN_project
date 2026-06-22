import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import os
import boto3
import tempfile

S3_BUCKET = os.environ.get("MODEL_BUCKET", "pm-defect-models-2026")
S3_KEY = os.environ.get("MODEL_KEY", "defect_model.pth")
LOCAL_PATH = os.path.join(tempfile.gettempdir(), "defect_model.pth")

def download_model_from_s3():
    if not os.path.exists(LOCAL_PATH):
        s3 = boto3.client("s3")
        s3.download_file(S3_BUCKET, S3_KEY, LOCAL_PATH)
    return LOCAL_PATH

def load_model():
    weights_path = download_model_from_s3()
    model = models.resnet18()
    model.fc = nn.Sequential(
        nn.Linear(512, 2)
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])