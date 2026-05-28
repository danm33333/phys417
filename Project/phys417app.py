import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path

# ---------------------------------------------------
# Base directory (deployment-safe paths)
# ---------------------------------------------------

BASE_DIR = Path(__file__).parent

# ---------------------------------------------------
# Streamlit page setup
# ---------------------------------------------------

st.set_page_config(
    page_title="EcoBirdNet",
    layout="centered"
)

# ---------------------------------------------------
# Custom background styling
# ---------------------------------------------------

st.markdown(
    """
    <style>

    .stApp {
        background-color: #87CEEB;
    }

    h1, h2, h3, h4, h5, h6, p, div {
        color: black;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("EcoBirdNet, Bird Species Classifier")

st.write(
    """
    Upload an image of a Pacific Northwest bird and EcoBirdNet will identify the species
    using a convolutional neural network (CNN).
    """
)

# ---------------------------------------------------
# Device configuration
# ---------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ---------------------------------------------------
# Constants
# ---------------------------------------------------

IMAGE_SIZE = 96

# ---------------------------------------------------
# Load class names
# ---------------------------------------------------

class_names = np.load(
    BASE_DIR / "class_names.npy",
    allow_pickle=True
)

NUM_CLASSES = len(class_names)

# ---------------------------------------------------
# Updated CNN Architecture
# ---------------------------------------------------

class BirdCNNModel(nn.Module):

    def __init__(self, num_classes=16):

        super(BirdCNNModel, self).__init__()

        self.features = nn.Sequential(

            # ---------------------------------------------------
            # Conv Block 1
            # ---------------------------------------------------

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # ---------------------------------------------------
            # Conv Block 2
            # ---------------------------------------------------

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # ---------------------------------------------------
            # Conv Block 3
            # ---------------------------------------------------

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # ---------------------------------------------------
            # Conv Block 4
            # ---------------------------------------------------

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # ---------------------------------------------------
            # Adaptive average pooling
            # ---------------------------------------------------

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # ---------------------------------------------------
        # Classifier
        # ---------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x

# ---------------------------------------------------
# Load trained model
# ---------------------------------------------------

@st.cache_resource
def load_model():

    model = BirdCNNModel(num_classes=NUM_CLASSES)

    model.load_state_dict(
        torch.load(
            BASE_DIR / "EcoBirdNet_best_model_state_dict.pt",
            map_location=device
        )
    )

    model.to(device)

    model.eval()

    return model

model = load_model()

# ---------------------------------------------------
# Image preprocessing
# ---------------------------------------------------

def preprocess_image(image):

    image = image.convert("RGB")

    # ---------------------------------------------------
    # Center crop to square
    # ---------------------------------------------------

    width, height = image.size

    crop_size = min(width, height)

    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    image = image.crop((left, top, right, bottom))

    # ---------------------------------------------------
    # Resize
    # ---------------------------------------------------

    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    # ---------------------------------------------------
    # Convert to NumPy
    # ---------------------------------------------------

    image_np = np.array(image).astype(np.float32)

    # Normalize to 0-1
    image_np = image_np / 255.0

    # Clip values
    image_np = np.clip(image_np, 0.0, 1.0)

    # ---------------------------------------------------
    # Convert to tensor
    # ---------------------------------------------------

    image_tensor = (
        torch.tensor(image_np)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )

    return image_tensor.to(device), image

# ---------------------------------------------------
# Species name cleanup helper
# ---------------------------------------------------

def clean_species_name(raw_species):

    # Remove numeric dataset prefix
    # Example:
    # "047.American_Goldfinch"
    # -> "American_Goldfinch"

    if "." in raw_species:

        species = raw_species.split(".", 1)[1]

    else:

        species = raw_species

    return species

# ---------------------------------------------------
# File uploader
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a bird image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# Prediction section
# ---------------------------------------------------

if uploaded_file is not None:

    original_image = Image.open(uploaded_file)

    input_tensor, processed_image = preprocess_image(
        original_image
    )

    # ---------------------------------------------------
    # Display uploaded image
    # ---------------------------------------------------

    st.subheader("Uploaded Image")

    st.image(
        original_image,
        use_container_width=True
    )

    # ---------------------------------------------------
    # Prediction
    # ---------------------------------------------------

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            1
        )

    # ---------------------------------------------------
    # Main prediction
    # ---------------------------------------------------

    raw_prediction = class_names[
        predicted_class.item()
    ]

    cleaned_prediction = clean_species_name(
        raw_prediction
    )

    display_prediction = cleaned_prediction.replace(
        "_",
        " "
    )

    confidence_percent = confidence.item() * 100

    # ---------------------------------------------------
    # Display top prediction
    # ---------------------------------------------------

    st.success(
        f"Predicted Species: {display_prediction}"
    )

    st.info(
        f"Confidence: {confidence_percent:.2f}%"
    )

    # ---------------------------------------------------
    # Top 5 Predictions
    # ---------------------------------------------------

    st.subheader("Top 5 Predictions")

    st.markdown(
        """
        <p style="font-size:14px;">
        <i>
        Click names to verify with the National Audubon Society's
        guide to North American birds
        </i>
        </p>
        """,
        unsafe_allow_html=True
    )

    top_probs, top_classes = torch.topk(
        probabilities,
        5
    )

    for i in range(5):

        # ---------------------------------------------------
        # Raw dataset label
        # ---------------------------------------------------

        raw_species = class_names[
            top_classes[0][i].item()
        ]

        prob = top_probs[0][i].item() * 100

        # ---------------------------------------------------
        # Remove dataset numbering
        # ---------------------------------------------------

        cleaned_species = clean_species_name(
            raw_species
        )

        # ---------------------------------------------------
        # Create display name
        # ---------------------------------------------------

        display_name = cleaned_species.replace(
            "_",
            " "
        )

        # ---------------------------------------------------
        # Create Audubon-compatible slug
        # ---------------------------------------------------

        # Special correction for dataset typo:
        # "Anna_Hummingbird" should map to:
        # "annas-hummingbird"

        if cleaned_species == "Anna_Hummingbird":

            slug = "annas-hummingbird"

        else:

            slug = (
                cleaned_species.lower()
                .replace("_", "-")
                .replace(" ", "-")
            )

        audubon_url = (
            f"https://www.audubon.org/field-guide/bird/{slug}"
        )

        # ---------------------------------------------------
        # Display clickable species link
        # ---------------------------------------------------

        st.markdown(
            f"""
            **{i+1}.**
            <a href="{audubon_url}" target="_blank">
            {display_name}
            </a>
            — {prob:.2f}%
            """,
            unsafe_allow_html=True
        )

# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.markdown("---")

st.markdown(
    """
    EcoBirdNet CNN Bird Species Classifier  
    Built with Streamlit + PyTorch  
    By Rasool Ray, Arthur Wang, and Daniel Michel for PHYS 417
    """
)
