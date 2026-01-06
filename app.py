"""
Image-to-Text Generator - Interactive Demo Application

A professional Streamlit-based web application for automatic image captioning
using BLIP architecture with multi-image support and batch processing.

Author: Tharun Ponnam
GitHub: @tharun-ship-it
"""

import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageDraw, ImageFont
import io
import zipfile

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Image-to-Text Generator | Tharun Ponnam",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS - Pink/Rose Theme with Tinted Background
# ============================================================
st.markdown("""
<style>
    /* Main background - soft pink tint */
    .stApp {
        background: linear-gradient(135deg, #fdf2f8 0%, #fce7f3 50%, #fbcfe8 100%);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    /* Sidebar header styling - BLACK text */
    .sidebar-header {
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #f9a8d4;
    }
    
    .sidebar-header-emoji {
        font-size: 2rem;
    }
    
    .sidebar-header-text {
        color: #1e293b;
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }
    
    .author-info {
        background: linear-gradient(135deg, #ec4899 0%, #be185d 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    
    .author-info h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
    }
    
    .author-info p {
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
    
    .author-info a {
        color: #fde047;
        text-decoration: none;
    }
    
    .caption-result {
        background: linear-gradient(135deg, #ffffff 0%, #fdf2f8 100%);
        border-left: 5px solid #ec4899;
        padding: 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(236, 72, 153, 0.15);
    }
    
    .caption-text {
        font-size: 1.3rem;
        font-weight: 600;
        color: #9d174d;
        margin-bottom: 0;
    }
    
    .stat-box {
        background: linear-gradient(135deg, #ffffff 0%, #fdf2f8 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #f9a8d4;
        box-shadow: 0 2px 4px rgba(236, 72, 153, 0.1);
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #be185d;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #64748b;
    }
    
    /* Compact stat box for sidebar - single line */
    .stat-box-compact {
        background: linear-gradient(135deg, #ffffff 0%, #fdf2f8 100%);
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #f9a8d4;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 70px;
    }
    
    .stat-value-compact {
        font-size: 1.2rem;
        font-weight: 700;
        color: #be185d;
        white-space: nowrap;
        line-height: 1.2;
    }
    
    .stat-label-compact {
        font-size: 0.7rem;
        color: #64748b;
        white-space: nowrap;
        line-height: 1.2;
    }
    
    .image-info {
        background: #ffffff;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #ec4899;
        margin-bottom: 1rem;
        font-size: 0.9rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .section-card {
        background: linear-gradient(135deg, #ffffff 0%, #fdf2f8 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #f9a8d4;
        margin-bottom: 1rem;
    }
    
    .section-header {
        color: #be185d;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #ffffff 0%, #fdf2f8 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 2px dashed #f9a8d4;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(236, 72, 153, 0.1);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stButton > button {
        background: linear-gradient(135deg, #ec4899 0%, #db2777 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 10px;
        min-height: 50px;
        transition: all 0.3s ease;
        white-space: nowrap !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #db2777 0%, #be185d 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);
    }
    
    .image-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #f9a8d4;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .image-number {
        background: linear-gradient(135deg, #ec4899 0%, #db2777 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #f9a8d4, transparent);
        margin: 1.5rem 0;
    }
    
    .tech-badge {
        background: linear-gradient(135deg, #fdf2f8 0%, #fce7f3 100%);
        color: #be185d;
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
        display: inline-block;
        margin: 0.2rem;
        border: 1px solid #f9a8d4;
    }
    
    .link-item {
        color: #be185d;
        text-decoration: none;
        font-size: 0.85rem;
        display: block;
        padding: 0.3rem 0;
    }
    
    .link-item:hover {
        color: #9d174d;
        text-decoration: underline;
    }
    
    .feature-item {
        font-size: 0.85rem;
        color: #475569;
        padding: 0.2rem 0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fdf2f8 0%, #fce7f3 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Initialize Session State
# ============================================================
if 'captions' not in st.session_state:
    st.session_state.captions = {}
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'results' not in st.session_state:
    st.session_state.results = []
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# ============================================================
# Clear Function
# ============================================================
def clear_all():
    """Clear all results and reset state."""
    st.session_state.captions = {}
    st.session_state.processed = False
    st.session_state.results = []
    st.session_state.uploader_key += 1

# ============================================================
# Load Model (Cached)
# ============================================================
@st.cache_resource
def load_model():
    """Load and cache the BLIP model."""
    model_name = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, processor, device

# ============================================================
# Helper Functions
# ============================================================
def generate_caption(image, model, processor, device, max_length=50, num_beams=5, min_length=5, conditional_text=None):
    """Generate caption for a single image."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    if conditional_text:
        inputs = processor(image, conditional_text, return_tensors="pt").to(device)
    else:
        inputs = processor(image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            min_length=min_length,
            repetition_penalty=1.5,
            length_penalty=1.0,
            early_stopping=True
        )
    
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    # Capitalize first letter
    caption = caption.strip()
    if caption:
        caption = caption[0].upper() + caption[1:]
    return caption

def get_image_size(image_file):
    """Get image file size in human readable format."""
    size_bytes = image_file.size
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def get_image_dimensions(image):
    """Get image dimensions."""
    return f"{image.width} × {image.height}"

def add_caption_to_image(image, caption):
    """Add caption text to the bottom of an image with white background and black text."""
    img = image.copy()
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    caption_height = max(60, img.height // 8)
    new_height = img.height + caption_height
    
    new_img = Image.new("RGB", (img.width, new_height), color=(255, 255, 255))
    new_img.paste(img, (0, 0))
    
    draw = ImageDraw.Draw(new_img)
    
    try:
        font_size = max(14, min(24, img.width // 30))
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    avg_char_width = font_size * 0.6
    max_chars = int(img.width * 0.9 / avg_char_width)
    
    display_caption = caption
    if len(caption) > max_chars and max_chars > 10:
        display_caption = caption[:max_chars-3] + "..."
    
    text_bbox = draw.textbbox((0, 0), display_caption, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (img.width - text_width) // 2
    text_y = img.height + (caption_height - text_height) // 2
    
    if text_x < 5:
        text_x = 5
    
    draw.text((text_x, text_y), display_caption, fill="black", font=font)
    
    return new_img

def create_zip_with_captions(results):
    """Create a ZIP file with images and their captions."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        captions_text = "Image Captions Generated by BLIP Model\n"
        captions_text += "=" * 50 + "\n\n"
        
        for idx, result in enumerate(results):
            img_with_caption = add_caption_to_image(result['image'], result['caption'])
            img_buffer = io.BytesIO()
            img_with_caption.save(img_buffer, format='PNG', quality=95)
            img_buffer.seek(0)
            
            base_name = result['filename'].rsplit('.', 1)[0]
            filename = f"{idx+1:02d}_{base_name}_captioned.png"
            zip_file.writestr(filename, img_buffer.getvalue())
            
            captions_text += f"Image {idx+1}: {result['filename']}\n"
            captions_text += f"Caption: {result['caption']}\n"
            captions_text += "-" * 50 + "\n\n"
        
        zip_file.writestr("captions_summary.txt", captions_text)
    
    zip_buffer.seek(0)
    return zip_buffer

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    # Sidebar Header (BLACK text - matching main heading)
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-header-emoji">🖼️</div>
        <div class="sidebar-header-text">Image-to-Text Generator</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Author Info
    st.markdown("""
    <div class="author-info">
        <h3>👤 Author</h3>
        <p><strong>Tharun Ponnam</strong></p>
        <p>🔗 <a href="https://github.com/tharun-ship-it" target="_blank">@tharun-ship-it</a></p>
        <p>📧 tharunponnam007@gmail.com</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Performance
    st.markdown('<div class="section-header">📊 Model Performance</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stat-box-compact">
            <div class="stat-value-compact">39.7%</div>
            <div class="stat-label-compact">BLEU-4</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-box-compact">
            <div class="stat-value-compact">136.7</div>
            <div class="stat-label-compact">CIDEr</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-box-compact">
            <div class="stat-value-compact">24.1%</div>
            <div class="stat-label-compact">SPICE</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Dataset Section
    st.markdown("""
    <div class="section-card">
        <div class="section-header">📚 Dataset</div>
        <p style="margin: 0; font-size: 0.9rem; font-weight: 600; color: #be185d;">TextCaps</p>
        <p style="margin: 0.2rem 0; font-size: 0.8rem; color: #64748b;">28,408 images • 142,040 captions</p>
        <p style="margin: 0.2rem 0; font-size: 0.75rem; color: #64748b;">OCR + Visual Reasoning</p>
        <a href="https://textvqa.org/textcaps/" target="_blank" class="link-item">📎 textvqa.org/textcaps</a>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features Section
    st.markdown("""
    <div class="section-card">
        <div class="section-header">✨ Key Features</div>
        <div class="feature-item">🖼️ Batch processing (25 images)</div>
        <div class="feature-item">🎯 Conditional captioning</div>
        <div class="feature-item">📥 ZIP download with captions</div>
        <div class="feature-item">⚡ Real-time inference</div>
        <div class="feature-item">🧠 BLIP-Large (129M params)</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Technologies Section
    st.markdown('<div class="section-header">🛠️ Technologies</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <span class="tech-badge">Python</span>
        <span class="tech-badge">PyTorch</span>
        <span class="tech-badge">Transformers</span>
        <span class="tech-badge">BLIP</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">PIL</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Links Section
    st.markdown("""
    <div class="section-card">
        <div class="section-header">🔗 Links</div>
        <a href="https://github.com/tharun-ship-it/image-to-text-generator" target="_blank" class="link-item">📂 GitHub Repository</a>
        <a href="https://arxiv.org/abs/2201.12086" target="_blank" class="link-item">📄 BLIP Paper</a>
        <a href="https://arxiv.org/abs/2003.12462" target="_blank" class="link-item">📄 TextCaps Paper (ECCV 2020)</a>
        <a href="https://huggingface.co/Salesforce/blip-image-captioning-large" target="_blank" class="link-item">🤗 Model on HuggingFace</a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Settings Section
    st.markdown('<div class="section-header">⚙️ Generation Settings</div>', unsafe_allow_html=True)
    
    max_length = st.slider(
        "Max Caption Length",
        min_value=20,
        max_value=75,
        value=50,
        step=5,
        help="Maximum tokens in caption"
    )
    
    num_beams = st.slider(
        "Beam Width",
        min_value=1,
        max_value=10,
        value=5,
        help="Higher = more accurate but slower"
    )
    
    min_length = st.slider(
        "Min Caption Length",
        min_value=5,
        max_value=20,
        value=5,
        help="Minimum tokens in caption"
    )
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Conditional Captioning
    st.markdown('<div class="section-header">🎨 Conditional Captioning</div>', unsafe_allow_html=True)
    use_conditional = st.checkbox("Enable prompt-guided captioning", value=False)
    conditional_text = None
    if use_conditional:
        conditional_text = st.text_input(
            "Start caption with:",
            placeholder="a photograph of",
            help="Guide the caption style"
        )
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Device Info
    device_name = "🚀 GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
    st.info(f"Running on: **{device_name}**")

# ============================================================
# Main Content
# ============================================================

# Title (BLACK heading)
st.markdown('<h1 class="main-title">🖼️ Image-to-Text Generator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Generate Natural Language Descriptions for Images using BLIP Architecture</p>', unsafe_allow_html=True)

# Quick Stats Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">BLIP</div>
        <div class="stat-label">Architecture</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">129M</div>
        <div class="stat-label">Parameters</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">25</div>
        <div class="stat-label">Max Images</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">2 GB</div>
        <div class="stat-label">Max Size</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Load Model
with st.spinner("🔄 Loading BLIP model... (first load takes ~60 seconds)"):
    model, processor, device = load_model()

# ============================================================
# Image Upload Section
# ============================================================
st.markdown("### 📤 Upload Images")

uploaded_files = st.file_uploader(
    "Batch processing: Upload up to 25 images per session",
    type=["jpg", "jpeg", "png", "webp", "bmp", "gif"],
    accept_multiple_files=True,
    help="Supported: JPG, JPEG, PNG, WEBP, BMP, GIF (Max 2GB total)",
    key=f"uploader_{st.session_state.uploader_key}"
)

if uploaded_files:
    if len(uploaded_files) > 25:
        st.warning("⚠️ Maximum 25 images allowed. Only first 25 will be processed.")
        uploaded_files = uploaded_files[:25]
    
    total_size = sum(f.size for f in uploaded_files)
    total_size_mb = total_size / (1024 * 1024)
    
    st.markdown(f"""
    <div class="image-info">
        <strong>📊 Upload Summary:</strong> {len(uploaded_files)} image(s) • Total size: {total_size_mb:.2f} MB
    </div>
    """, unsafe_allow_html=True)
    
    # Buttons Row
    col_btn1, col_space1, col_btn2, col_space2 = st.columns([1.5, 0.2, 0.8, 3.5])
    
    with col_btn1:
        generate_btn = st.button("🪄 Generate Captions", type="primary")
    
    with col_btn2:
        st.button("🗑️ Refresh", on_click=clear_all)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    if generate_btn:
        st.markdown("### 📝 Generated Captions")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"🔍 Processing image {idx + 1} of {len(uploaded_files)}...")
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
            image = Image.open(uploaded_file)
            
            caption = generate_caption(
                image, model, processor, device,
                max_length=max_length,
                num_beams=num_beams,
                min_length=min_length,
                conditional_text=conditional_text if use_conditional and conditional_text else None
            )
            
            results.append({
                'filename': uploaded_file.name,
                'size': get_image_size(uploaded_file),
                'dimensions': get_image_dimensions(image),
                'caption': caption,
                'image': image
            })
        
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.results = results
        st.session_state.captions = {r['filename']: r['caption'] for r in results}
        st.session_state.processed = True
        
        st.success(f"✅ Successfully generated captions for {len(results)} image(s)!")
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        for idx, result in enumerate(results):
            st.markdown(f'<span class="image-number">Image {idx + 1}</span>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(result['image'], use_container_width=True)
                st.markdown(f"""
                <div style="font-size: 0.85rem; color: #64748b; text-align: center;">
                    📁 {result['filename']}<br>
                    📐 {result['dimensions']} • 💾 {result['size']}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="caption-result">
                    <div class="caption-text">"{result['caption']}"</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    elif st.session_state.processed and st.session_state.results:
        st.markdown("### 📝 Generated Captions")
        
        for idx, result in enumerate(st.session_state.results):
            st.markdown(f'<span class="image-number">Image {idx + 1}</span>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(result['image'], use_container_width=True)
                st.markdown(f"""
                <div style="font-size: 0.85rem; color: #64748b; text-align: center;">
                    📁 {result['filename']}<br>
                    📐 {result['dimensions']} • 💾 {result['size']}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="caption-result">
                    <div class="caption-text">"{result['caption']}"</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    else:
        st.markdown("### 👁️ Preview Uploaded Images")
        
        cols = st.columns(min(5, len(uploaded_files)))
        for idx, uploaded_file in enumerate(uploaded_files[:10]):
            with cols[idx % 5]:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"{uploaded_file.name[:15]}...", use_container_width=True)
                st.markdown(f"""
                <div style="font-size: 0.75rem; color: #64748b; text-align: center;">
                    {get_image_size(uploaded_file)} • {get_image_dimensions(image)}
                </div>
                """, unsafe_allow_html=True)
        
        if len(uploaded_files) > 10:
            st.info(f"📷 +{len(uploaded_files) - 10} more image(s) uploaded")

else:
    st.markdown("""
    <div class="upload-section">
        <p style="font-size: 1.2rem; color: #be185d; margin-bottom: 0.5rem;">🖼️ No images uploaded</p>
        <p style="color: #64748b;">Upload up to 25 images to generate AI-powered captions</p>
        <p style="font-size: 0.85rem; color: #9ca3af;">Supported formats: JPG, JPEG, PNG, WEBP, BMP, GIF</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# Download Section
# ============================================================
if st.session_state.processed and st.session_state.results:
    st.markdown("### 📥 Download Results")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        zip_buffer = create_zip_with_captions(st.session_state.results)
        
        st.download_button(
            label="📥 Download Images with Captions (ZIP)",
            data=zip_buffer,
            file_name="captioned_images.zip",
            mime="application/zip",
            use_container_width=True
        )
        
        st.markdown("""
        <p style="text-align: center; font-size: 0.8rem; color: #64748b; margin-top: 0.5rem;">
            ZIP contains all images with captions (black text on white) + summary text file
        </p>
        """, unsafe_allow_html=True)
