"""Main Streamlit application for ChartGemma model."""
import os

# Set file watcher type to poll to avoid PyTorch introspection issues
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll"

import streamlit as st
from PIL import Image
import torch

from src.utils.env import load_environment, get_env
from src.models.chartgemma import load_model, setup_device
from src.inference.predictor import predict_chartgemma
from src.providers.openai_provider import analyze_chart_with_openai
from src.providers.gemini_provider import analyze_chart_with_gemini
from src.utils.examples import get_local_examples
from src.config import AppConfig

# Load environment variables from .env if available
load_environment()
config = AppConfig()

# Page configuration
st.set_page_config(
    page_title="ChartGemma - Chart Analysis AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Hide sidebar */
    .stSidebar { display: none !important; }
    .main .block-container { padding-top: 3rem; padding-bottom: 3rem; max-width: 1400px; }
    h1 { text-align: center; color: #1f77b4; margin-bottom: 0.5rem; }
    .subtitle { text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 3rem; }
    .stButton > button { width: 100%; border-radius: 8px; padding: 0.75rem 1.5rem; font-size: 1.1rem; font-weight: 600; transition: all 0.3s; }
    .output-card { background-color: #f0f7ff; border-left: 4px solid #1f77b4; padding: 1.5rem; border-radius: 8px; margin-top: 1rem; }
    .status-badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.85rem; font-weight: 500; margin-left: 0.5rem; }
    .status-ready { background-color: #d4edda; color: #155724; }
    .footer { text-align: center; color: #666; margin-top: 3rem; padding-top: 2rem; border-top: 1px solid #e0e0e0; }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<h1>📊 ChartGemma - Chart Analysis AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a chart or pick a local example to get AI-powered insights</p>', unsafe_allow_html=True)

# Initialize session state for model loading
if 'model' not in st.session_state or 'processor' not in st.session_state:
    with st.spinner("⏳ Loading model... This may take a moment."):
        model, processor = load_model()
        device = setup_device()
        model.to(device)
        st.session_state.model = model
        st.session_state.processor = processor
        st.session_state.device = device
        st.session_state.model_loaded = True

# Show model status
if st.session_state.get('model_loaded', False):
    device_icon = "🚀" if torch.cuda.is_available() else "🖥️"
    device_text = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.markdown(
        f'<div style="text-align: center; margin-bottom: 2rem;">'
        f'<span class="status-badge status-ready">✓ Model Ready</span>'
        f'<span style="color: #666; font-size: 0.9rem;">{device_icon} Running on {device_text}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

# Load a few local examples from the dataset (val split by default)
local_examples = get_local_examples(dataset_root="ChartQA Dataset", split="val", max_examples=3)

# Main content area with better layout
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 📤 Input")
    
    # Image upload section
    uploaded_file = st.file_uploader(
        "**Upload Your Chart Image**",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a chart image in PNG, JPG, or JPEG format",
        label_visibility="visible"
    )
    
    # Example images section - only show preview when selected
    st.markdown("---")
    st.markdown("#### 💡 Or Pick a Local Example (from ChartQA Dataset)")
    
    # Build radio options from local examples
    options = ["None"]
    example_map = {}
    for idx, ex in enumerate(local_examples):
        title = f"{ex['imgname']} — {ex['query'][:60]}{'…' if len(ex['query'])>60 else ''}"
        options.append(title)
        example_map[title] = ex
    
    with st.expander("Select Example Chart", expanded=False):
        example_choice = st.radio(
            "Choose an example:",
            options,
            key="example_radio",
            label_visibility="collapsed"
        )
    
    # Determine which image to use and show preview only when selected or uploaded
    image = None
    image_source = None
    selected_example = example_map.get(example_choice)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_source = "uploaded"
    elif selected_example is not None:
        image = Image.open(selected_example['image_path'])
        image_source = "example"
    
    # Show preview only if image is available
    if image is not None:
        st.markdown("---")
        if image_source == "uploaded":
            st.markdown("**📷 Uploaded Chart Preview:**")
            st.image(image, caption="Your Chart", use_container_width=True)
        else:
            st.markdown("**📷 Example Chart Preview:**")
            cap = selected_example['imgname'] if selected_example else "Example"
            st.image(image, caption=cap, use_container_width=True)
    else:
        # Show placeholder when no image
        st.markdown("---")
        st.markdown(
            '<div style="text-align: center; padding: 3rem; background-color: #f8f9fa; border-radius: 8px; border: 2px dashed #ccc;">'
            '<p style="color: #999; font-size: 1.1rem;">📊 Chart preview will appear here</p>'
            '<p style="color: #999; font-size: 0.9rem; margin-top: 0.5rem;">Upload an image or select an example above</p>'
            '</div>',
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Input prompt section
    st.markdown("#### ✍️ Your Question")
    suggested_prompt = selected_example['query'] if selected_example else ""
    input_prompt = st.text_area(
        "Enter your question or prompt about the chart:",
        value=suggested_prompt,
        height=120,
        help="Ask questions like 'What is the highest value?', 'Describe the trend', etc.",
        placeholder="e.g., What is the highest value in this chart? Describe the trend shown in this chart..."
    )

with col2:
    st.markdown("### 📊 Analysis")
    
    # Provider selection (defaults from config)
    provider = st.selectbox(
        "Model Provider",
        options=["ChartGemma", "OpenAI GPT-4o", "Google Gemini"],
        index=["ChartGemma", "OpenAI GPT-4o", "Google Gemini"].index(config.default_provider),
        help="Choose which model provider to use for analysis"
    )
    
    # Optional model override per provider
    if provider == "ChartGemma":
        model_name_hint = config.chartgemma_model
    elif provider == "OpenAI GPT-4o":
        model_name_hint = config.openai_model
    else:
        model_name_hint = config.gemini_model
    
    with st.expander("Advanced: Model Name", expanded=False):
        custom_model_name = st.text_input("Model identifier", value=model_name_hint)
    
    # Analyze button - prominent placement
    analyze_disabled = image is None or not input_prompt.strip()
    
    if st.button(
        "🔍 Analyze Chart",
        type="primary",
        use_container_width=True,
        disabled=analyze_disabled,
        help="Analyze the chart with AI" if not analyze_disabled else "Please upload an image and enter a question first"
    ):
        if image is None:
            st.error("❌ Please upload an image or select an example chart.")
        elif not input_prompt.strip():
            st.error("❌ Please enter a question or prompt about the chart.")
        else:
            with st.spinner("🤖 AI is analyzing your chart... This may take a few moments."):
                try:
                    if provider == "ChartGemma":
                        output_text = predict_chartgemma(
                            image,
                            input_prompt,
                            st.session_state.model,
                            st.session_state.processor
                        )
                    elif provider == "OpenAI GPT-4o":
                        output_text = analyze_chart_with_openai(image, input_prompt, model=custom_model_name)
                    else:
                        output_text = analyze_chart_with_gemini(image, input_prompt, model=custom_model_name)
                    
                    # Store output in session state for persistence
                    st.session_state.last_output = output_text
                    st.session_state.last_prompt = input_prompt
                    
                    st.success("✅ Analysis Complete!")
                    
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    st.exception(e)
                    st.session_state.last_output = None
    
    # Display output
    st.markdown("---")
    
    if 'last_output' in st.session_state and st.session_state.last_output:
        st.markdown("#### 💬 Analysis Result")
        st.markdown('<div class="output-card">', unsafe_allow_html=True)
        st.markdown(st.session_state.last_output)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show what question was asked
        if 'last_prompt' in st.session_state:
            st.caption(f"📝 Question: {st.session_state.last_prompt}")
    else:
        st.markdown("#### 💬 Analysis Result")
        st.markdown(
            '<div style="text-align: center; padding: 3rem; background-color: #f8f9fa; border-radius: 8px; border: 2px dashed #ccc;">'
            '<p style="color: #999; font-size: 1rem;">Analysis results will appear here</p>'
            '<p style="color: #999; font-size: 0.85rem; margin-top: 0.5rem;">Click "Analyze Chart" to get started</p>'
            '</div>',
            unsafe_allow_html=True
        )

# Footer
st.markdown("---")
st.markdown(
    '<div class="footer">'
    '<p>Built with ❤️ using <strong>Streamlit</strong> and <strong>ChartGemma</strong></p>'
    '<p style="font-size: 0.85rem; color: #999;">Powered by PaliGemma | Multimodal Chart Understanding AI</p>'
    '</div>',
    unsafe_allow_html=True
)
