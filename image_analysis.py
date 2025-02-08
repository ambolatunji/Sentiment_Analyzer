import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import json
import requests

# Load ImageNet class labels
imagenet_url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
imagenet_classes = requests.get(imagenet_url).json()

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def analyze_facial_expression(image):
    try:
        analysis = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion'], analysis[0]['emotion']
    except Exception as e:
        return "Error", {}

def classify_objects(image):
    model = models.resnet50(pretrained=True)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
    
    _, predicted = torch.max(outputs, 1)
    class_id = str(predicted.item())
    return imagenet_classes[class_id]

def run():
    st.title("🖼️ Image-Based Sentiment & Object Analysis")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_cv2 = np.array(image.convert('RGB'))
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        st.subheader("Processing Image...")
        
        faces = detect_faces(image_cv2)
        dominant_emotion, emotions = analyze_facial_expression(image_cv2)
        object_category = classify_objects(image)
        
        st.subheader("Analysis Results:")
        
        if dominant_emotion != "Error":
            confidence = emotions.get(dominant_emotion, 0) * 100
            st.write(f"**Detected Emotion:** {dominant_emotion.capitalize()} 😃 (Confidence: {confidence:.2f}%)")
        else:
            st.write("**Detected Emotion:** No face detected.")
        
        st.write(f"**Detected Object:** {object_category[1]} 🏷️ (Class ID: {object_category[0]})")
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(image_cv2, (x, y), (x+w, y+h), (255, 0, 0), 2)
            st.image(image_cv2, caption="Detected Faces", use_column_width=True)
        else:
            st.write("No faces detected.")

if __name__ == "__main__":
    run()
