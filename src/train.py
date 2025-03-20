import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import SARColorizer
from data_loader import SARDataset
from tqdm import tqdm
import os

def train(data_dir, epochs=100, batch_size=8, learning_rate=0.0001):
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    else:
        print("CUDA is not available. Using CPU")
        device = torch.device('cpu')
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create dataset and dataloader
    print("Loading dataset...")
    dataset = SARDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, 
                          batch_size=batch_size, 
                          shuffle=True,
                          num_workers=2,
                          pin_memory=True)
    print(f"Dataset loaded with {len(dataset)} image pairs")
    
    # Initialize model
    model = SARColorizer().to(device)
    
    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create directory for checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, 
                          desc=f'Epoch {epoch+1}/{epochs}',
                          leave=True)
        
        for batch_idx, (sar_images, optical_images) in enumerate(progress_bar):
            # Move data to GPU
            sar_images = sar_images.to(device)
            optical_images = optical_images.to(device)
            
            # Forward pass
            outputs = model(sar_images)
            loss = criterion(outputs, optical_images)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Save checkpoint every 500 batches
            if (batch_idx + 1) % 500 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'batch': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }
                torch.save(checkpoint, 
                          f'checkpoints/checkpoint_epoch_{epoch+1}_batch_{batch_idx+1}.pth')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')

if __name__ == "__main__":
    # Set your data directory
    data_dir = r"C:\Desktop\New folder (2)\unzipped_folder\v_2"
    
    # Set CUDA device settings
    torch.backends.cudnn.benchmark = True
    
    try:
        train(data_dir, 
              epochs=100, 
              batch_size=8,  # Adjusted for GTX 1650
              learning_rate=0.0001)
    except Exception as e:
        print(f"Error during training: {str(e)}") 