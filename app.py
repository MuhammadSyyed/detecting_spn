# app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from datetime import datetime
from PIL import Image

# =====================
# CONFIG
# =====================
CLASS_NAMES = ["benign", "malignant", "normal"]
MODEL_PATH = "./models/spn_classifier.pth"  # your saved state_dict

# =====================
# RECREATE MODEL & LOAD WEIGHTS
# =====================
@st.cache_resource
def load_model():
    # Same architecture as training
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Replace final layer for your 3 classes
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    
    # Load weights
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    model.eval()
    return model

model = load_model()

# =====================
# IMAGE PREPROCESSING
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# =====================
# STREAMLIT UI
# =====================
st.title("🩻 SPN Classification App")
st.write("Upload a lung CT scan image and the model will predict if it's **Benign**, **Malignant**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = CLASS_NAMES[predicted.item()]



    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}_{predicted_class}.jpg"
    image.save(f"./uploaded_images/{filename}")

    st.markdown(f"### 🏷 Prediction: **{predicted_class.upper()}**")
