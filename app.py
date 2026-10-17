import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="Waste Classification | Transfer Learning MobileNetV2",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2e7d32;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stProgress > div > div > div > div {
        background-color: #2e7d32;
    }
    .badge-primary {
        background-color: #e8f5e9;
        color: #1b5e20;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Constants & Model Cache
# ==============================================================================
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_ICONS = {
    "cardboard": "📦",
    "glass": "🍾",
    "metal": "🥫",
    "paper": "📄",
    "plastic": "🧴",
    "trash": "🗑️"
}

RECYCLING_GUIDES = {
    "cardboard": {
        "bin": "Blue / Paper & Cardboard Recycling Bin",
        "action": "Flatten all corrugated boxes to save bin volume. Remove plastic tape and styrofoam inserts. Keep dry.",
        "recyclable": "✅ 100% Recyclable",
        "color": "#8d6e63"
    },
    "glass": {
        "bin": "Glass Recycling Bin (often Green/Brown/Clear separation)",
        "action": "Empty and lightly rinse remaining liquids. Remove plastic/metal screw caps. Avoid broken window panes or ceramics.",
        "recyclable": "✅ 100% Recyclable (Infinitely reusable)",
        "color": "#0288d1"
    },
    "metal": {
        "bin": "Metals & Cans Bin (Yellow / Mixed Dry Recyclables)",
        "action": "Rinse beverage and soup cans. Flatten aluminum cans if feasible. Aerosol cans must be completely empty.",
        "recyclable": "✅ 100% Recyclable",
        "color": "#78909c"
    },
    "paper": {
        "bin": "Blue / Paper Recycling Bin",
        "action": "Keep dry and clean. Flatten newspapers, magazines, and envelopes. Do not recycle greasy pizza boxes or wax-coated cups.",
        "recyclable": "✅ Recyclable (if clean & dry)",
        "color": "#5c6bc0"
    },
    "plastic": {
        "bin": "Yellow / Plastics & Bottles Bin",
        "action": "Check resin code (PET #1 and HDPE #2 are most accepted). Empty liquids and replace cap. Squash bottles to conserve space.",
        "recyclable": "✅ Recyclable depending on resin code",
        "color": "#ff9800"
    },
    "trash": {
        "bin": "Black / Grey Residual Landfill Bin",
        "action": "Residual non-recyclable solid waste (soiled wraps, composite polymers, chip bags). Dispose responsibly.",
        "recyclable": "❌ Non-Recyclable / Residual Waste",
        "color": "#d32f2f"
    }
}

MODEL_PATH = "waste_model.keras"

@st.cache_resource(show_spinner="Loading trained MobileNetV2 model...")
def load_classification_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_classification_model()

# ==============================================================================
# Helper Functions
# ==============================================================================
def predict_image(image: Image.Image, trained_model):
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize to model input resolution
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    
    predictions = trained_model.predict(img_batch, verbose=0)[0]
    return predictions

def get_sample_images_by_class(class_name):
    folder = os.path.join("trashnet_data", "dataset-resized", class_name)
    if os.path.exists(folder):
        files = glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png"))
        return sorted(files)
    return []

# ==============================================================================
# Sidebar Navigation & Information
# ==============================================================================
with st.sidebar:
    st.markdown("## ♻️ Waste Classification")
    st.markdown("**Transfer Learning Project (TrashNet Benchmark)**")
    st.markdown("---")
    
    nav_option = st.radio(
        "Navigation Menu:",
        [
            "🔍 Image Classifier",
            "📊 Evaluation & Metrics",
            "💡 Recycling Guidelines",
            "🔬 Architecture & Methodology"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Specifications")
    st.markdown("""
    - **Base Network:** MobileNetV2
    - **Pretrained Weights:** ImageNet
    - **Input Size:** 224 × 224 × 3
    - **Total Parameters:** 2.26 Million
    - **Number of Classes:** 6
    - **Test Accuracy:** **85.03%**
    - **Framework:** TensorFlow / Keras 3
    """)
    
    if model is not None:
        st.success("✅ Model: `waste_model.keras` Active")
    else:
        st.error("⚠️ Model file not found!")
    
    st.caption("College Deep Learning Project | TrashNet Dataset")

# ==============================================================================
# View 1: Real-Time Classifier
# ==============================================================================
if nav_option == "🔍 Image Classifier":
    st.markdown('<div class="main-header">♻️ Intelligent Waste Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Classify waste items into 6 categories (Cardboard, Glass, Metal, Paper, Plastic, Trash) using Transfer Learning with MobileNetV2.</div>', unsafe_allow_html=True)
    
    if model is None:
        st.error("The model file `waste_model.keras` is missing. Please ensure the model is trained and saved.")
        st.stop()

    input_mode = st.radio(
        "Select Image Input Source:",
        ["📁 Upload Your Own Image", "🖼️ Test with TrashNet Dataset Samples"],
        horizontal=True
    )
    
    selected_image = None
    image_source_label = ""
    
    if input_mode == "📁 Upload Your Own Image":
        uploaded_file = st.file_uploader(
            "Upload a waste image (JPG, JPEG, PNG):",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of cardboard, glass, metal, paper, plastic, or trash item."
        )
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)
            image_source_label = f"Uploaded File: {uploaded_file.name}"
            
    else:
        col_cat, col_samp = st.columns([1, 2])
        with col_cat:
            sample_class = st.selectbox("1. Choose Waste Class:", CLASS_NAMES, index=1)
        
        sample_files = get_sample_images_by_class(sample_class)
        with col_samp:
            if sample_files:
                sample_file = st.selectbox(
                    "2. Choose Sample Image:",
                    sample_files[:15],
                    format_func=lambda x: os.path.basename(x)
                )
                if sample_file:
                    selected_image = Image.open(sample_file)
                    image_source_label = f"Ground Truth Class: **{sample_class.upper()}** ({os.path.basename(sample_file)})"
            else:
                st.warning("No sample images found in `trashnet_data/dataset-resized`.")

    if selected_image is not None:
        st.markdown("---")
        col_img, col_pred = st.columns([1.1, 1.4], gap="large")
        
        with col_img:
            st.markdown(f"#### 📷 Input Image")
            st.image(selected_image, caption=image_source_label, use_container_width=True)
            st.caption(f"Original Resolution: {selected_image.size[0]} × {selected_image.size[1]} px | Format: {selected_image.format or 'RGB'}")
            
        with col_pred:
            st.markdown("#### 🎯 Prediction Results")
            
            with st.spinner("Classifying image..."):
                preds = predict_image(selected_image, model)
                top_idx = int(np.argmax(preds))
                top_class = CLASS_NAMES[top_idx]
                top_conf = float(preds[top_idx])
                top_icon = CLASS_ICONS[top_class]
            
            # Prediction Card
            st.markdown(f"""
            <div style="background-color: #e8f5e9; border: 2px solid #81c784; border-radius: 12px; padding: 18px; margin-bottom: 20px;">
                <div style="font-size: 0.9rem; color: #2e7d32; font-weight: 600; text-transform: uppercase;">Primary Prediction</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #1b5e20;">
                    {top_icon} {top_class.upper()}
                </div>
                <div style="font-size: 1.1rem; color: #388e3c; font-weight: 600; margin-top: 4px;">
                    Confidence Score: {top_conf * 100:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar for top class
            st.progress(top_conf)
            
            # Probability Distribution Chart
            st.markdown("##### Class Probability Distribution")
            df_preds = pd.DataFrame({
                "Class": [f"{CLASS_ICONS[c]} {c.capitalize()}" for c in CLASS_NAMES],
                "Confidence (%)": [round(float(p) * 100, 2) for p in preds]
            }).sort_values(by="Confidence (%)", ascending=True)
            
            st.bar_chart(df_preds.set_index("Class"), horizontal=True, color="#2e7d32")
            
            # Recycling Instructions
            guide = RECYCLING_GUIDES[top_class]
            st.markdown("---")
            st.markdown(f"#### ♻️ Disposal & Recycling Recommendations")
            st.info(f"**Disposal Bin:** {guide['bin']}\n\n**Recyclability Status:** {guide['recyclable']}\n\n**Action Advice:** {guide['action']}")

# ==============================================================================
# View 2: Model Evaluation & Metrics
# ==============================================================================
elif nav_option == "📊 Evaluation & Metrics":
    st.markdown('<div class="main-header">📊 Model Performance & Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation of MobileNetV2 on the held-out TrashNet test set (374 unseen images).</div>', unsafe_allow_html=True)
    
    # Summary Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown("""
        <div class="card">
            <div class="metric-label">Stage 1 Test Acc</div>
            <div class="metric-value">84.76%</div>
            <div style="color: #64748b; font-size: 0.8rem;">Frozen Base</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="card">
            <div class="metric-label">Stage 2 Test Acc</div>
            <div class="metric-value">85.03%</div>
            <div style="color: #2e7d32; font-size: 0.8rem;">+0.27% after FT</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="card">
            <div class="metric-label">Test Samples</div>
            <div class="metric-value">374</div>
            <div style="color: #64748b; font-size: 0.8rem;">15% deterministic split</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="card">
            <div class="metric-label">Correctly Classified</div>
            <div class="metric-value">318 / 374</div>
            <div style="color: #64748b; font-size: 0.8rem;">85.03% correct</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown("""
        <div class="card">
            <div class="metric-label">Training Time</div>
            <div class="metric-value">7.6 min</div>
            <div style="color: #64748b; font-size: 0.8rem;">457.2 seconds total</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Tabs for evaluation visuals
    t1, t2, t3, t4, t5 = st.tabs([
        "📈 Learning Curves",
        "🎯 Confusion Matrix",
        "📦 Class Distribution",
        "⚠️ Error Analysis (Wrong Predictions)",
        "📋 Full Classification Report"
    ])
    
    with t1:
        st.markdown("#### Training and Validation Curves")
        st.markdown("Accuracy and loss progression across **Stage 1 (Frozen Base, epochs 1–9)** and **Stage 2 (Fine-tuning top 40 layers, epochs 10–13)**. The dashed vertical line marks the start of fine-tuning.")
        if os.path.exists("training_curves.png"):
            st.image("training_curves.png", use_container_width=True)
        else:
            st.warning("training_curves.png not found.")

    with t2:
        st.markdown("#### Confusion Matrix Heatmap")
        st.markdown("Distribution of actual true labels vs. MobileNetV2 predicted classes on the 374 test set images.")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_container_width=True)
        else:
            st.warning("confusion_matrix.png not found.")
            
        st.markdown("""
        **Key Observations & Class Confusion:**
        - **Plastic vs. Glass:** 7 glass items were classified as plastic, and 3 plastic items as glass. This occurs because clear glass bottles and translucent PET plastic bottles share similar specular highlights and edge geometries.
        - **Metal vs. Glass:** 8 metal cans were misclassified as glass due to reflections on metallic surfaces.
        - **Trash Class:** Trash had the lowest recall (58.33%) because it is the smallest class in the dataset (137 images vs 594 for paper) and contains heterogeneous mixed garbage.
        """)

    with t3:
        st.markdown("#### TrashNet Class Distribution")
        st.markdown("Number of images per class across the TrashNet dataset (Total: 2,527 images).")
        if os.path.exists("class_distribution.png"):
            st.image("class_distribution.png", use_container_width=True)
        else:
            st.warning("class_distribution.png not found.")

    with t4:
        st.markdown("#### Sample Misclassified Test Images")
        st.markdown("8 test samples where the model's prediction differed from the ground truth label:")
        if os.path.exists("wrong_predictions.png"):
            st.image("wrong_predictions.png", use_container_width=True)
        else:
            st.warning("wrong_predictions.png not found.")

    with t5:
        st.markdown("#### Detailed Classification Report")
        report_data = {
            "Waste Class": ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash", "Macro Average", "Weighted Average"],
            "Precision": [0.8846, 0.8706, 0.8852, 0.8837, 0.7714, 0.7000, 0.8326, 0.8499],
            "Recall":    [0.8846, 0.9024, 0.8182, 0.8941, 0.8308, 0.5833, 0.8189, 0.8503],
            "F1-Score":  [0.8846, 0.8862, 0.8504, 0.8889, 0.8000, 0.6364, 0.8244, 0.8493],
            "Test Support": [52, 82, 66, 85, 65, 24, 374, 374]
        }
        df_report = pd.DataFrame(report_data)
        st.dataframe(
            df_report.style.format({
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1-Score": "{:.2%}"
            }),
            use_container_width=True
        )

# ==============================================================================
# View 3: Recycling Guidelines
# ==============================================================================
elif nav_option == "💡 Recycling Guidelines":
    st.markdown('<div class="main-header">💡 Waste Segregation & Recycling Guidelines</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Proper disposal and recycling practices for each of the 6 TrashNet waste categories.</div>', unsafe_allow_html=True)
    
    for cls in CLASS_NAMES:
        g = RECYCLING_GUIDES[cls]
        with st.expander(f"{CLASS_ICONS[cls]} {cls.upper()} - {g['recyclable']}", expanded=True):
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.markdown(f"**Target Bin:**\n`{g['bin']}`")
                st.markdown(f"**Status:** {g['recyclable']}")
            with col_b:
                st.markdown(f"**Preparation Guidelines:**\n{g['action']}")

# ==============================================================================
# View 4: Architecture & Methodology
# ==============================================================================
elif nav_option == "🔬 Architecture & Methodology":
    st.markdown('<div class="main-header">🔬 Model Architecture & Transfer Learning Strategy</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Technical design choices, transfer learning methodology, and comparison with alternative architectures.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 1. Two-Stage Transfer Learning Pipeline
    
    Transfer learning leverages rich feature extractors pretrained on large-scale datasets (ImageNet with 1.4 million images) and adapts them to domain-specific waste images:
    
    - **Stage 1 (Frozen Base Feature Extractor):**
      - The pretrained MobileNetV2 backbone is frozen (`trainable = False`).
      - Only the newly added classification head (`GlobalAveragePooling2D` + `Dropout(0.3)` + `Dense(6, softmax)`) is trained.
      - Optimizer: Adam with $\\text{lr} = 10^{-3}$.
      - Reaches **84.76% test accuracy** in 9 epochs.
      
    - **Stage 2 (Fine-Tuning Top 40 Layers):**
      - The top 40 layers of MobileNetV2 are unfrozen to allow high-level feature adaptation.
      - Lower layers remain frozen to preserve foundational edge/texture filters and prevent catastrophic forgetting.
      - Optimizer: Adam with a small learning rate $\\text{lr} = 10^{-5}$.
      - Lifts final test accuracy to **85.03%**.
    """)
    
    st.markdown("---")
    st.markdown("### 2. Architectural Comparison")
    
    comp_data = {
        "Architecture": ["MobileNetV2 (Selected)", "ResNet50", "EfficientNetB0"],
        "Parameters": ["2.26 Million", "25.6 Million", "5.3 Million"],
        "Base Model Size (FP32)": ["~8.6 MB (22.1 MB .keras)", "~98.0 MB", "~20.5 MB"],
        "FLOPs (MACS)": ["300 M", "3,800 M (~12.6x)", "390 M (~1.3x)"],
        "Edge Device Inference": ["Sub-50ms (Optimal)", "Slow / High Memory", "Moderate"],
        "Overfitting Risk on TrashNet": ["Low (Compact)", "High (Excess Capacity)", "Low-Moderate"]
    }
    df_comp = pd.DataFrame(comp_data)
    st.table(df_comp)
    
    st.markdown("""
    **Why MobileNetV2 is the Best Choice for Automated Waste Sorting:**
    1. **Edge Hardware Compatibility:** Smart waste bins and automated conveyor systems operate on resource-constrained embedded devices (e.g., Raspberry Pi 4, NVIDIA Jetson Nano). MobileNetV2 requires only 300M FLOPs, enabling real-time classification.
    2. **Regularization & Dataset Fit:** TrashNet contains 2,527 images. Over-parameterized architectures like ResNet50 (25.6M parameters) tend to overfit rapidly without massive regularizations, whereas MobileNetV2 achieves the ideal trade-off between representation power and generalization.
    """)
