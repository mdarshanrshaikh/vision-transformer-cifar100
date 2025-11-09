import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image
import io

# --- 1. Configurations ---
NUM_CLASSES = 100
MODEL_NAME = 'vit_tiny_patch16_224'
IMG_SIZE = 224
DEVICE = "cpu" 
SAVE_PATH = 'vit_cifar100_finetuned.pth' 

# --- 2. Utility Functions ---

def get_cifar100_class_names(num_classes):
    """
    Creates mock class names.
    """
    return [f"CIFAR-100 Category {i+1}" for i in range(num_classes)]

@st.cache_resource
def get_prediction_transforms(img_size):
    """Define transforms for prediction: resize, center crop, normalize."""
    # Note: Using the same normalization stats as used during training
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return transform

@st.cache_resource
def load_prediction_model(num_classes, model_name, device):
    """Loads the ViT architecture and then loads the fine-tuned state dictionary."""
    with st.spinner(f"Loading ViT model and fine-tuned weights..."):
        # 1. Initialize the empty model architecture with the correct head size
        model = timm.create_model(
            model_name,
            pretrained=False, # We load custom weights, not ImageNet defaults
            num_classes=num_classes 
        ).to(device)

        # 2. Load the saved weights (state dictionary) from the .pth file
        try:
            # Use map_location to ensure compatibility regardless of where it was saved (CPU/GPU)
            state_dict = torch.load(SAVE_PATH, map_location=device)
            model.load_state_dict(state_dict)
            st.sidebar.success(f"Weights loaded successfully from **{SAVE_PATH}**")
        except FileNotFoundError:
            st.sidebar.error(f"Error: Model file '{SAVE_PATH}' not found!")
            st.sidebar.warning("The app is currently using randomly initialized weights, not your trained weights. Please place the '.pth' file in the same directory as this app.")
        except Exception as e:
            st.sidebar.error(f"Error loading state dictionary: {e}")
            
    # Set model to evaluation mode for inference
    model.eval()
    return model

# --- 3. Prediction Logic ---

def predict_image(model, image_file, transform, class_names):
    """Processes an uploaded image and performs inference."""
    try:
        img = Image.open(image_file).convert("RGB")
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top 5 predictions
    top_prob, top_catid = torch.topk(probabilities, 5)

    with col2:
        st.subheader("Top 5 Predicted Classes")
        
        # Create prediction table for display
        results = []
        for i in range(top_prob.size(0)):
            class_index = top_catid[i].item()
            results.append({
                "Rank": i + 1,
                "Class Name": class_names[class_index],
                "Confidence": f"{top_prob[i].item() * 100:.2f}%"
            })

        st.table(results)


# --- 4. Streamlit UI Setup ---

def main():
    st.set_page_config(layout="wide", page_title="ViT Prediction Demo")

    st.title("👁️ Vision Transformer (ViT) Inference")
    st.markdown("Use this app to classify an image using your **fine-tuned PyTorch model** (weights loaded from a file).")
    st.divider()

    # Get necessary components
    transform = get_prediction_transforms(IMG_SIZE)
    model = load_prediction_model(NUM_CLASSES, MODEL_NAME, DEVICE)
    class_names = get_cifar100_class_names(NUM_CLASSES)

    st.sidebar.header("Model Info")
    st.sidebar.markdown(f"**Architecture:** `{MODEL_NAME}`")
    st.sidebar.markdown(f"**Input Size:** `{IMG_SIZE}x{IMG_SIZE}`")
    st.sidebar.markdown(f"**Prediction Device:** `{DEVICE.upper()}`")

    st.subheader("Upload an Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, PNG) for classification", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.info("Processing image and running inference...")
        predict_image(model, uploaded_file, transform, class_names)
        
    else:
        st.info("Upload an image to see the model's prediction.")

# Ensure the app starts
if __name__ == "__main__":
    main()