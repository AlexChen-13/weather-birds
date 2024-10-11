import streamlit as st
import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import numpy as np

# Set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the class names
# Assuming you have a JSON file with class names, adjust the path accordingly

class_names = ['Black_footed_Albatross',
 'Laysan_Albatross',
 'Sooty_Albatross',
 'Groove_billed_Ani',
 'Crested_Auklet',
 'Least_Auklet',
 'Parakeet_Auklet',
 'Rhinoceros_Auklet',
 'Brewer_Blackbird',
 'Red_winged_Blackbird']

# Define the image transformations
trnsfrms = T.Compose([
    T.Resize((224, 224)),  # Resize to 224x224
    T.ToTensor(),  # Convert to tensor
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load the trained model
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_classes = len(class_names)  # Assuming class_names contains all class labels
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load('resnet18_birds_state_dict2.pth', map_location=DEVICE))
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()

# Function to make predictions
def predict(image):
    image = trnsfrms(image).unsqueeze(0)  # Add batch dimension
    image = image.to(DEVICE)  # Move image to the device
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

# Streamlit app layout
st.title("Bird Species Classification")
st.write("Upload an image of a bird to classify.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make prediction
    prediction = predict(image)
    st.write(f"Prediction: {prediction}")

