import os
from ultralytics import YOLO
import cv2
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torch
from patch_extractor import extract_leg_segment,extract_arm_segment,get_nonzero_patch_indices,visualize_patches

model = YOLO('yolo11n-pose.pt')

def get_patch_idx(image, target_class):
    results = model(image)
    if target_class=="Upper Extremities":
        for result in results:
            keypoints = result.keypoints.data
            

                
            img = cv2.imread(image)
            if img is None:
                print("Error: Could not load image")
                continue
                
            height, width = img.shape[:2]
            print(f"Image dimensions: {width}x{height}")
            
            # Extract left arm
            left_arm, left_mask = extract_arm_segment(img, keypoints, 'left', thickness=100)
            
            # Extract right arm
            right_arm, right_mask = extract_arm_segment(img, keypoints, 'right', thickness=100)
            
    
    
            combined_mask = cv2.bitwise_or(left_mask, right_mask)
            combined_result = cv2.bitwise_and(img, img, mask=combined_mask)

            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
         
        
        nonzero_patches = get_nonzero_patch_indices(combined_result)
        visualize_patches(image,nonzero_patches)
        print(f"Non-zero patch indices: {nonzero_patches}")
        print(f"Total patches with content: {len(nonzero_patches)}")

  	            
                
    else:
        for result in results:
            keypoints = result.keypoints.data
            
            img = cv2.imread(image)
            if img is None:
                print("Error: Could not load image")
                continue
                
            height, width = img.shape[:2]
            print(f"Image dimensions: {width}x{height}")
            
            # Extract left leg
            left_leg, left_mask = extract_leg_segment(img, keypoints, 'left', thickness=100)
            
            # Extract right leg
            right_leg, right_mask = extract_leg_segment(img, keypoints, 'right', thickness=100)
            
    
        
            combined_mask = cv2.bitwise_or(left_mask, right_mask)
            combined_result = cv2.bitwise_and(img, img, mask=combined_mask)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
          

            
            
        
        nonzero_patches = get_nonzero_patch_indices(combined_result)
        visualize_patches(image,nonzero_patches)
        print(f"Non-zero patch indices: {nonzero_patches}")
        print(f"Total patches with content: {len(nonzero_patches)}")


        
        
    indices = nonzero_patches
    return indices


def custom_collate_fxn(batch):
    images, labels, indices = zip(*batch)
    return torch.stack(images, 0), list(labels), list(indices)

class VisionFineTuneDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.ToTensor()
        ])
        
        # Load and clean CSV
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        
        print(f"Original dataset size: {len(self.df)}")
        print(f"Image directory: {self.image_dir}")
        print(f"Directory exists: {os.path.exists(self.image_dir)}")
        
        # Debug: Show first few image names
        if len(self.df) > 0:
            print(f"Sample image names from CSV: {self.df['image_name'].head().tolist()}")
        
        # Filter out rows where image files don't exist
        self.df = self._filter_existing_images()
        
        self.label_mapping = {
            'Normal': '0',
            'Wound': '1', 
            'Amputation': '1',
            'Not Testable': '0',
        }

    def _filter_existing_images(self):
   
        existing_rows = []
        
        # Get all files in the image directory for debugging
        if os.path.exists(self.image_dir):
            all_files = os.listdir(self.image_dir)
      
            # Show some example files
            image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
            if image_files:
                print(f"Sample image files: {image_files[:5]}")
        else:
            print(f"ERROR: Image directory does not exist: {self.image_dir}")
            return pd.DataFrame()
        
        for idx, row in self.df.iterrows():
            # Clean the image name more carefully
            image_name = str(row['image_name']).strip()
            
            # Remove .json extension if present, but handle cases where it's not there
            if image_name.endswith('.json'):
                image_name = image_name.replace('.json', '')
            
            # Try multiple possible extensions and cases
            possible_extensions = ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG']
            image_found = False
            actual_path = None
            
            for ext in possible_extensions:
                test_path = os.path.join(self.image_dir, image_name + ext)
                if os.path.exists(test_path):
                    image_found = True
                    actual_path = test_path
                    break
            
            if image_found:
                existing_rows.append(row)
            else:
                print(f"Warning: Image not found for '{image_name}' (tried extensions: {possible_extensions})")
                # Debug: Show what files are actually there with similar names
                similar_files = [f for f in all_files if image_name.lower() in f.lower()]
                if similar_files:
                    print(f"  Similar files found: {similar_files}")
        
        filtered_df = pd.DataFrame(existing_rows)
        print(f"Filtered dataset: {len(filtered_df)} out of {len(self.df)} samples have corresponding images")
        
        return filtered_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if len(self.df) == 0:
            raise RuntimeError("Dataset is empty! No images were found.")
            
        row = self.df.iloc[idx]
        
        # Clean the image name
        image_name = str(row['image_name']).strip()
        if image_name.endswith('.json'):
            image_name = image_name.replace('.json', '')
        
        # Try to find the image file
        possible_extensions = ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG']
        image_path = None
        
        for ext in possible_extensions:
            test_path = os.path.join(self.image_dir, image_name + ext)
            if os.path.exists(test_path):
                image_path = test_path
                break
        
        if image_path is None:
            raise FileNotFoundError(f"Image not found for base name: {image_name}")
        
        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        # Prepare labels
        labels = {
            "trauma_upper_ext": self.label_mapping.get(row['Upper Extremities'], '0'),
            "trauma_lower_ext": self.label_mapping.get(row['Lower Extremities'], '0'),
        }
        
        # Get patch indices 
        patch_indices = {
            "trauma_upper_ext": get_patch_idx(image_path, "Upper Extremities"),
            "trauma_lower_ext": get_patch_idx(image_path, "Lower Extremities"),
        }
        
        return image, labels, patch_indices

def create_dataloader(image_dir, csv_path, batch_size=1, shuffle=True):
   
    try:
        dataset = VisionFineTuneDataset(image_dir, csv_path)
        
        if len(dataset) == 0:
            raise ValueError("Dataset is empty! Please check your image directory and CSV file.")
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=custom_collate_fxn
        )
        
        print(f"Successfully created dataloader with {len(dataset)} samples")
        return dataloader
        
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        return None


dataloader = create_dataloader("/home/uas-dtu/DTC-Trauma/DTC-Trauma/main/yolo/","/home/uas-dtu/DTC-Trauma/output_sorted_updated.csv")


for batch_idx, (images, labels_batch, indices_batch) in enumerate(dataloader):
    print(f"Batch {batch_idx}:")
    print(f"  Labels: {labels_batch[0]}")  
    print(f"  Patch Indices: {indices_batch[0]}")
    
    if batch_idx >= 1:  
        break
