import torch
from model import SARColorizer
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np

def predict_color(model_path, input_image_path, output_dir):
    """
    Predict colorized version of a SAR image
    Args:
        model_path: Path to the trained model checkpoint
        input_image_path: Path to input SAR image
        output_dir: Directory to save the colorized output
    """
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    else:
        print("CUDA is not available. Using CPU")
        device = torch.device('cpu')
    
    # Load model
    model = SARColorizer().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both simple state_dict and checkpoint dictionary
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Prepare transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load and transform input image
    input_image = Image.open(input_image_path).convert('L')
    original_size = input_image.size
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    
    # Generate prediction
    with torch.no_grad():
        output = model(input_tensor)
    
    # Convert output tensor to image
    output_image = output.squeeze().cpu()
    output_image = output_image.permute(1, 2, 0)  # Change from CxHxW to HxWxC
    output_image = ((output_image + 1) / 2).numpy()  # Denormalize
    output_image = np.clip(output_image, 0, 1)
    
    # Convert to PIL Image and resize to original dimensions
    output_image = Image.fromarray((output_image * 255).astype(np.uint8))
    output_image = output_image.resize(original_size, Image.Resampling.BICUBIC)
    
    # Save results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_colorized.png")
    output_image.save(output_path)
    
    # Display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(input_image, cmap='gray')
    ax1.set_title('Input SAR Image')
    ax1.axis('off')
    
    ax2.imshow(output_image)
    ax2.set_title('Colorized Output')
    ax2.axis('off')
    
    plt.savefig(os.path.join(output_dir, f"{base_name}_comparison.png"))
    plt.show()
    
    return output_path

def predict_batch(model_path, input_dir, output_dir):
    """
    Predict colorized versions for all SAR images in a directory
    Args:
        model_path: Path to the trained model checkpoint
        input_dir: Directory containing input SAR images
        output_dir: Directory to save the colorized outputs
    """
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tif')):
            input_path = os.path.join(input_dir, filename)
            try:
                output_path = predict_color(model_path, input_path, output_dir)
                print(f"Processed {filename} -> {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    # Example usage for single image
    model_path = "checkpoints\checkpoint_epoch_10_batch_1500.pth"  # Path to your trained model
    input_image = "7.png"  # Path to your input SAR image
    output_dir = "predictions"  # Directory to save outputs
    
    predict_color(model_path, input_image, output_dir)
    
    # Example usage for batch processing
    # input_dir = "path/to/sar_images"
    # predict_batch(model_path, input_dir, output_dir) 