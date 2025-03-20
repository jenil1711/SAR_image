import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class SARDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.sar_images = []
        self.optical_images = []
        
        categories = ['urban', 'grassland', 'barrenland', 'agri']
        
        for category in categories:
            category_path = os.path.join(root_dir, category)
            s1_path = os.path.join(category_path, 's1')
            s2_path = os.path.join(category_path, 's2')
            
            if not os.path.exists(s1_path) or not os.path.exists(s2_path):
                print(f"Warning: Missing s1 or s2 directory in {category}")
                continue
            
            s1_files = sorted([f for f in os.listdir(s1_path) if f.endswith('.png')])
            s2_files = sorted([f for f in os.listdir(s2_path) if f.endswith('.png')])
            
            for s1_file, s2_file in zip(s1_files, s2_files):
                self.sar_images.append(os.path.join(s1_path, s1_file))
                self.optical_images.append(os.path.join(s2_path, s2_file))

    def __len__(self):
        return len(self.sar_images)

    def __getitem__(self, idx):
        try:
            sar_image = Image.open(self.sar_images[idx]).convert('L')
            optical_image = Image.open(self.optical_images[idx]).convert('RGB')
            
            if self.transform:
                sar_image = self.transform(sar_image)
                optical_image = self.transform(optical_image)
            
            return sar_image, optical_image
        except Exception as e:
            print(f"Error loading image pair {idx}: {str(e)}")
            raise e

def print_directory_structure(path):
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files[:5]:  # Print first 5 files only
            print(f"{subindent}{f}")

# Use this to check your directory structure
print_directory_structure(r"C:\Desktop\New folder (2)\unzipped_folder\v_2") 