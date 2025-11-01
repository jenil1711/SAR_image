
import streamlit as st
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import os
import io
import matplotlib.pyplot as plt
import sys



# Add current directory to path to ensure model imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import your model - using explicit import to avoid torch._classes error
from src.model import SARColorizer

# Set page configuration
st.set_page_config(
    page_title="SAR Image Colorization",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4285F4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5F6368;
        margin-bottom: 1rem;
    }
    .description {
        font-size: 1rem;
        text-align: justify;
    }
    .image-container {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #5F6368;
    }
    .stButton>button {
        background-color: #4285F4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #3367D6;
    }
</style>
""", unsafe_allow_html=True)

def load_model(model_path):
    """Load the trained colorization model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SARColorizer().to(device)
    
    try:
        # Use torch.load with strict map_location setting
        checkpoint = torch.load(model_path, map_location=device)
        # Check if it's a full checkpoint dictionary or just the state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, device

def process_image(image, model, device):
    """Process the image through the colorization model"""
    # Transform the image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Get original image dimensions for later resizing
    original_size = image.size
    
    # Convert to grayscale if it's not already
    if image.mode != 'L':
        image = image.convert('L')
    
    # Transform and prepare for model
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate prediction
    with torch.no_grad():
        output = model(input_tensor)
    
    # Convert output tensor to image
    output_image = output.squeeze().cpu()
    output_image = output_image.permute(1, 2, 0)  # Change from CxHxW to HxWxC
    output_image = ((output_image + 1) / 2).numpy()  # Denormalize
    output_image = np.clip(output_image, 0, 1)
    
    # Convert to PIL Image and resize back to original dimensions
    output_image = Image.fromarray((output_image * 255).astype(np.uint8))
    output_image = output_image.resize(original_size, Image.Resampling.BICUBIC)
    
    return output_image

def create_comparison(input_image, output_image):
    """Create a side-by-side comparison of input and output images"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot input image
    ax1.imshow(input_image, cmap='gray')
    ax1.set_title("Input SAR Image")
    ax1.axis('off')
    
    # Plot output image
    ax2.imshow(output_image)
    ax2.set_title("Colorized Output")
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    comparison_img = Image.open(buf)
    plt.close(fig)
    
    return comparison_img

def main():
    st.markdown("<h1 class='main-header'>SAR Image Colorization</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Convert grayscale SAR images to colorized versions</h2>", unsafe_allow_html=True)
    
    st.markdown("<p class='description'>This application uses a deep learning model trained on Sentinel-1 (SAR) and Sentinel-2 (optical) image pairs to colorize grayscale Synthetic Aperture Radar (SAR) images. The model was trained on various terrain types including urban, grassland, barren land, and agricultural areas.</p>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.image("https://cdn.pixabay.com/photo/2017/08/30/01/05/milky-way-2695569_960_720.jpg")   #changes 
    st.sidebar.title("Model Options")
    
    # Use your specific model path - updated to find the checkpoint relative to current directory
    checkpoint_filename = "checkpoint_epoch_10_batch_2000.pth"
    default_checkpoint_path = os.path.join(current_dir, "checkpoints", checkpoint_filename)
    
    # Check if the model exists at the specified path
    if os.path.exists(default_checkpoint_path):
        model_path = default_checkpoint_path
    else:
        # Try the hardcoded path
        model_path = r"C:\Desktop\New folder (2)\checkpoints\checkpoint_epoch_10_batch_2000.pth"
        if not os.path.exists(model_path):
            st.sidebar.error("Model checkpoint not found. Please specify a valid path.")
            model_path = st.sidebar.text_input("Enter model path:", model_path)
    
    st.sidebar.info(f"Using model: {os.path.basename(model_path)}")
    
    # Load model
    model, device = load_model(model_path)
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Image", "Sample Images"])
    
    with tab1:
        st.header("Upload a grayscale SAR image")
        uploaded_file = st.file_uploader("Choose a grayscale image", type=["jpg", "jpeg", "png", "tif"])
        
        if uploaded_file is not None:
            # Process the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with st.spinner("Generating colorized image..."):
                if model:
                    # Process the image through the model
                    colorized_image = process_image(image, model, device)
                    
                    # Display result
                    comparison = create_comparison(image, colorized_image)
                    st.image(comparison, caption="Input vs. Colorized", use_container_width=True)
                    
                    # Download button for the colorized image
                    buf = io.BytesIO()
                    colorized_image.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.download_button(
                            label="Download Colorized Image",
                            data=byte_im,
                            file_name="colorized_image.png",
                            mime="image/png",
                        )
    
    with tab2:
        st.header("Try with sample images")
        
        # Use the predictions directory path
        sample_dir = r"C:\Desktop\New folder (2)\predictions"
        
        # Check for sample images
        sample_files = []
        for root, _, files in os.walk(sample_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    sample_files.append(os.path.join(root, file))
        
        if sample_files:
            # Create a dropdown for sample selection
            selected_sample = st.selectbox("Select a sample image", 
                                          [os.path.basename(f) for f in sample_files])
            
            # Get selected sample path
            sample_path = os.path.join(sample_dir, selected_sample)
            
            # Check if sample exists
            if os.path.exists(sample_path):
                image = Image.open(sample_path)
                st.image(image, caption=f"Sample SAR Image: {selected_sample}", use_container_width=True)
                
                if st.button("Colorize Sample"):
                    with st.spinner("Generating colorized image..."):
                        if model:
                            # Process the sample image
                            colorized_image = process_image(image, model, device)
                            
                            # Display result
                            comparison = create_comparison(image, colorized_image)
                            st.image(comparison, caption="Input vs. Colorized", use_container_width=True)
            else:
                st.warning(f"Sample {selected_sample} not found. Please add sample images to the 'predictions' directory.")
            
            # Offer to copy a few SAR images from user data as samples
            if st.button("Create Sample Images"):
                try:
                    # Get source directory (assuming the data directory structure)
                    source_dir = r"C:\Desktop\New folder (2)\unzipped_folder\v_2"
                    if os.path.exists(source_dir):
                        # Copy a few images from each category
                        categories = ['urban', 'grassland', 'barrenland', 'agri']
                        import shutil
                        count = 0
                        for category in categories:
                            s1_path = os.path.join(source_dir, category, 's1')
                            if os.path.exists(s1_path):
                                for file in os.listdir(s1_path)[:2]:  # Copy first 2 files
                                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                                        src = os.path.join(s1_path, file)
                                        dst = os.path.join(sample_dir, f"{category}_{file}")
                                        shutil.copy2(src, dst)
                                        count += 1
                        st.success(f"Created {count} sample images in the samples directory.")
                        st.experimental_rerun()
                    else:
                        st.error("Source data directory not found.")
                except Exception as e:
                    st.error(f"Error creating sample images: {str(e)}")
    
    # Add batch processing capability
    with st.expander("Batch Processing"):
        st.write("Upload multiple SAR images to process them all at once.")
        batch_files = st.file_uploader("Upload multiple images", 
                                      type=["jpg", "jpeg", "png", "tif"], 
                                      accept_multiple_files=True)
        
        if batch_files and st.button("Process All Images"):
            with st.spinner("Processing batch of images..."):
                results = []
                for file in batch_files:
                    image = Image.open(file)
                    if model:
                        colorized_image = process_image(image, model, device)
                        results.append((image, colorized_image, file.name))
                
                # Create a ZIP file with all results
                import zipfile
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for i, (_, colorized, filename) in enumerate(results):
                        img_buffer = io.BytesIO()
                        colorized.save(img_buffer, format="PNG")
                        zip_file.writestr(f"colorized_{filename}", img_buffer.getvalue())
                
                # Display results
                for i, (original, colorized, filename) in enumerate(results):
                    st.subheader(f"Image {i+1}: {filename}")
                    comparison = create_comparison(original, colorized)
                    st.image(comparison, caption=f"Result for {filename}", use_container_width=True)
                
                # Download button for all colorized images
                st.download_button(
                    label="Download All Colorized Images",
                    data=zip_buffer.getvalue(),
                    file_name="colorized_images.zip",
                    mime="application/zip",
                )
    
    # Footer
    st.markdown("<div class='footer'>Developed for SAR Image Colorization Project - ISRO Challenge SIH1733</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    # Create samples directory if it doesn't exist
    os.makedirs("samples", exist_ok=True)
    
    main() 