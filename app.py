import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import io
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import time

# Import model definition
from model import create_model

# Set page configuration
st.set_page_config(
    page_title="Skin Lesion Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SkinLesionPredictor:
    def __init__(self, model_path):
        # Load model information
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            st.sidebar.success(f"Model loaded successfully from {model_path}")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {str(e)}")
            st.stop()
        
        # Create model architecture
        self.model = create_model(num_classes=7, backbone="efficientnet")
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Get class mapping
        self.class_mapping = checkpoint.get('class_mapping', {
            0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'
        })
        
        self.class_names = checkpoint.get('class_names', [
            'Actinic Keratoses', 'Basal Cell Carcinoma', 'Benign Keratosis',
            'Dermatofibroma', 'Melanoma', 'Melanocytic Nevi', 'Vascular Lesion'
        ])
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):
        # Make sure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Preprocess the image
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(image_tensor)
            inference_time = time.time() - start_time
            
            probabilities = F.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
        
        # Get result
        diagnosis_code = self.class_mapping[predicted_class]
        diagnosis_name = self.class_names[predicted_class]
        confidence = probabilities[predicted_class].item() * 100
        
        # Create probability dictionary for all classes
        prob_dict = {self.class_names[i]: prob.item() * 100 
                     for i, prob in enumerate(probabilities)}
        
        # Return results
        result = {
            'diagnosis_code': diagnosis_code,
            'diagnosis_name': diagnosis_name,
            'confidence': confidence,
            'probabilities': prob_dict,
            'inference_time': inference_time
        }
        
        return result

# Function to plot probability bars
def plot_probability_bars(probabilities):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort probabilities for better visualization
    sorted_probs = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))
    
    classes = list(sorted_probs.keys())
    probs = list(sorted_probs.values())
    
    # Shortened class names for better display
    short_classes = [c.split(" ")[0] if len(c.split(" ")) > 1 else c for c in classes]
    
    # Create horizontal bar chart
    bars = ax.barh(short_classes, probs, color='skyblue')
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(min(width + 1, 105), bar.get_y() + bar.get_height()/2,
                f'{probs[i]:.1f}%', va='center')
    
    # Set chart properties
    ax.set_xlim(0, 105)
    ax.set_xlabel('Probability (%)')
    ax.set_title('Prediction Probabilities by Class')
    
    # Add grid lines
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    return fig

# App title and description
st.title("🔬 Skin Lesion Classification")
st.markdown("""
This application helps classify skin lesion images into 7 different diagnostic categories using deep learning.
Upload a dermatoscopic image to get a prediction.
""")

# Sidebar information
st.sidebar.header("About")
st.sidebar.markdown("""
This application uses a pre-trained CNN model trained on the HAM10000 dataset 
to classify skin lesions into the following categories:

1. Actinic Keratoses (akiec)
2. Basal Cell Carcinoma (bcc)
3. Benign Keratosis (bkl)
4. Dermatofibroma (df)
5. Melanoma (mel)
6. Melanocytic Nevi (nv)
7. Vascular Lesion (vasc)
""")

# st.sidebar.header("Model Information")
# model_type = st.sidebar.selectbox(
#     "Select Model Architecture",
#     ["Custom CNN with Residuals", "Standard Custom CNN"]
# )

# Path to model file
model_path = "results/efficientnet-exp1/final_model.pth"

# Initialize the model
@st.cache_resource
def load_predictor(model_path):
    return SkinLesionPredictor(model_path)

try:
    predictor = load_predictor(model_path)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.info("Please make sure the model file exists and is correctly formatted.")
    st.stop()

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    # File uploader
    uploaded_file = st.file_uploader("Choose a skin lesion image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Add a button to make prediction
        if st.button("Classify Lesion"):
            with st.spinner("Analyzing image..."):
                # Make prediction
                result = predictor.predict(image)
                
                # Store results in session state to share with the other column
                st.session_state.result = result
                st.session_state.has_result = True

with col2:
    # Check if we have a result to display
    if 'has_result' in st.session_state and st.session_state.has_result:
        result = st.session_state.result
        
        # Display result
        st.markdown("## Prediction Result")
        
        # Create a colored box for the prediction
        confidence = result['confidence']
        if confidence > 80:
            box_color = "rgba(76, 175, 80, 0.2)"  # Green for high confidence
        elif confidence > 50:
            box_color = "rgba(255, 193, 7, 0.2)"  # Yellow for medium confidence
        else:
            box_color = "rgba(244, 67, 54, 0.2)"  # Red for low confidence
        
        # Display prediction in a styled box
        st.markdown(
            f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {box_color};">
                <h3 style="margin: 0;">Diagnosis: {result['diagnosis_name']}</h3>
                <p style="margin: 5px 0;">Code: {result['diagnosis_code']}</p>
                <p style="margin: 5px 0;">Confidence: {result['confidence']:.2f}%</p>
                <p style="margin: 5px 0; font-size: 0.8em;">Inference time: {result['inference_time']*1000:.1f} ms</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Plot probabilities
        st.markdown("### Probability Distribution")
        fig = plot_probability_bars(result['probabilities'])
        st.pyplot(fig)
        
        # Warning message for low confidence predictions
        if confidence < 70:
            st.warning("""
                ⚠️ **Low confidence prediction**
                
                This prediction has relatively low confidence. Consider:
                - Using a better quality image
                - Consulting with a dermatologist
                """)
        
        # Disclaimer
        st.info("""
            **Medical Disclaimer**: This tool is for educational purposes only 
            and should not replace professional medical advice. Always consult 
            with a qualified dermatologist for proper diagnosis.
            """)

# Add model information at the bottom
st.sidebar.markdown("---")
st.sidebar.markdown("### Technical Details")
st.sidebar.markdown(f"Running on: **{predictor.device}**")
st.sidebar.markdown("""
This application demonstrates how deep learning can assist in
dermatological diagnosis. However, it should be used only as a
supplementary tool to professional medical evaluation.
""")

# Add download button for a sample image if user wants to test
st.sidebar.markdown("---")
st.sidebar.markdown("### Sample Images")
st.sidebar.markdown("Don't have an image to test? Download a sample:")

# Define sample images for each class - replace these with actual sample image file paths if available
sample_image_path = "data/ISIC2018_Task3_Training_Input/ISIC_0024307.jpg"
if os.path.exists(sample_image_path):
    with open(sample_image_path, "rb") as file:
        btn = st.sidebar.download_button(
            label="Download Sample Image",
            data=file,
            file_name="sample_skin_lesion.jpg",
            mime="image/jpeg"
        )