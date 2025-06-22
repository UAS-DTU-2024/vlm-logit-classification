import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from masks import HuggingFacePoseProcessor


def get_nonzero_patch_indices(image_input, target_size=(336, 336), patch_size=(14, 14)):
    """
    Calculates the indices of patches that contain non-zero pixels.
    Accepts either an image path or a NumPy array.
    """
    image_array = None
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
             # Return empty list if the mask file doesn't exist
            return []
        image = Image.open(image_input)
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        image_array = np.array(image)
 
    elif isinstance(image_input, np.ndarray):
        # Handle numpy array input
        image_array = cv2.resize(image_input, target_size, interpolation=cv2.INTER_LANCZOS4)
        
    else:
        raise ValueError("Input must be image path string or numpy array")
    
    # Convert to grayscale if it's a color image
    if len(image_array.shape) == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    height, width = image_array.shape
    patch_h, patch_w = patch_size
    
    n_patches_h = height // patch_h
    n_patches_w = width // patch_w
    
    nonzero_indices = []
    patch_index = 1  # CLS token is 0, so patches start from 1
    
    for row in range(n_patches_h):
        for col in range(n_patches_w):
            patch = image_array[row*patch_h:(row+1)*patch_h, col*patch_w:(col+1)*patch_w]
            if np.any(patch > 20):
                nonzero_indices.append(patch_index)
            patch_index += 1
    
    return nonzero_indices

def custom_collate_fxn(batch):
    images, labels, indices = zip(*batch)
    return torch.stack(images, 0), list(labels), list(indices)

class VisionFineTuneDataset(Dataset):
    def __init__(self, image_dir, csv_path, pose_processor, transform=None):
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        
        self.pose_processor = pose_processor
        
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.ToTensor()
        ])
        
        print(f"Original dataset size: {len(self.df)}")
        self.df = self._filter_existing_images()
        print(f"Filtered dataset size: {len(self.df)}")

        self.label_mapping = {
            'Normal': '0', 'Wound': '1', 'Amputation': '1', 'Not Testable': '0',
        }

    def _filter_existing_images(self):
        existing_indices = self.df['image_name'].apply(
            lambda name: self._find_image_path(str(name).strip()) is not None
        )
        return self.df[existing_indices].reset_index(drop=True)

    def _find_image_path(self, base_name):
        if base_name.endswith('.json'):
            base_name = base_name.replace('.json', '')
        possible_extensions = ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG']
        for ext in possible_extensions:
            test_path = os.path.join(self.image_dir, base_name + ext)
            if os.path.exists(test_path):
                return test_path
        return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        base_image_name = str(row['image_name']).strip()
        
        image_path = self._find_image_path(base_image_name)
        if image_path is None:
            raise FileNotFoundError(f"Image not found for base name: {base_image_name}")

        image_pil = Image.open(image_path).convert("RGB")
        
        # 1. Generate all body part masks using the updated processor method
        body_masks,_ = self.pose_processor.generate_body_part_masks(image_pil)
        
        # 2. Calculate patch indices for each body part from its mask
        # Head
        head_mask = body_masks.get('head')
        head_patches = get_nonzero_patch_indices(head_mask) if head_mask is not None else []

        # Torso
        torso_mask = body_masks.get('torso')
        torso_patches = get_nonzero_patch_indices(torso_mask) if torso_mask is not None else []

        # Upper Extremities
        upper_ext_patches = []
        if body_masks.get('larm') is not None:
            upper_ext_patches.extend(get_nonzero_patch_indices(body_masks['larm']))
        if body_masks.get('rarm') is not None:
            upper_ext_patches.extend(get_nonzero_patch_indices(body_masks['rarm']))

        # Lower Extremities
        lower_ext_patches = []
        if body_masks.get('lleg') is not None:
            lower_ext_patches.extend(get_nonzero_patch_indices(body_masks['lleg']))
        if body_masks.get('rleg') is not None:
            lower_ext_patches.extend(get_nonzero_patch_indices(body_masks['rleg']))

        patch_indices = {
            "trauma_head": head_patches,
            "trauma_torso": torso_patches,
            "trauma_upper_ext": sorted(list(set(upper_ext_patches))),
            "trauma_lower_ext": sorted(list(set(lower_ext_patches))),
        }
        
        # 3. Prepare labels, safely checking for new columns in the CSV
        # Using row.get('Column', 'DefaultValue') is safe if columns are missing.
        # It will default to 'Normal', which maps to label '0'.
        labels = {
            "trauma_head": self.label_mapping.get(row.get('Head', 'Normal'), '0'),
            "trauma_torso": self.label_mapping.get(row.get('Torso', 'Normal'), '0'),
            "trauma_upper_ext": self.label_mapping.get(row['Upper Extremities'], '0'),
            "trauma_lower_ext": self.label_mapping.get(row['Lower Extremities'], '0'),
        }
        
        # 4. Apply transformations to the original image for the model
        image_tensor = self.transform(image_pil)
        
        return image_tensor, labels, patch_indices

# The create_dataloader function remains the same as the previous answer.
def create_dataloader(image_dir, csv_path, batch_size=1, shuffle=True):
    try:
        print("Initializing HuggingFacePoseProcessor...")
        pose_processor = HuggingFacePoseProcessor()
        print("Processor initialized.")
        
        dataset = VisionFineTuneDataset(
            image_dir=image_dir, 
            csv_path=csv_path,
            pose_processor=pose_processor
        )
        
        if len(dataset) == 0:
            raise ValueError("Dataset is empty! Check image paths and CSV content.")
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=custom_collate_fxn,
            pin_memory=True
        )
        
        print(f"Successfully created dataloader with {len(dataset)} samples.")
        return dataloader
        
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        return None
def visualize_patches(image_input, nonzero_indices, title="Patch Visualization", target_size=(336, 336), patch_size=(14, 14)):
    """Visualizes an image with its patch grid, highlighting specified patches."""
    
    # 1. Handle input and resize
    image_array = None
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
        image_array = np.array(image)
    elif isinstance(image_input, np.ndarray):
        image_array = image_input.copy()
    elif isinstance(image_input, Image.Image):
        image_array = np.array(image_input.convert("RGB"))
    else:
        raise ValueError("Unsupported input type for visualization")
        
    image_array = cv2.resize(image_array, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    # 2. Prepare grid image (ensure it's 3-channel for color drawing)
    grid_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    if len(image_array.shape) == 3:
        grid_image = image_array.copy()
    else: # Grayscale mask
        grid_image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)

    # 3. Draw grid and highlight patches
    height, width, _ = grid_image.shape
    patch_h, patch_w = patch_size
    n_patches_h = height // patch_h
    n_patches_w = width // patch_w
    
    patch_index = 1 # Patches are 1-indexed
    for row in range(n_patches_h):
        for col in range(n_patches_w):
            start_w, start_h = col * patch_w, row * patch_h
            end_w, end_h = start_w + patch_w, start_h + patch_h
            
            # Highlight patches in the list
            if patch_index in nonzero_indices:
                # Add a semi-transparent overlay
                overlay = grid_image.copy()
                cv2.rectangle(overlay, (start_w, start_h), (end_w, end_h), (0, 255, 0), -1) # Green fill
                alpha = 0.4
                cv2.addWeighted(overlay, alpha, grid_image, 1 - alpha, 0, grid_image)
                
                # Add a border
                cv2.rectangle(grid_image, (start_w, start_h), (end_w, end_h), (0, 255, 0), 2) # Green border
                
                # Add patch number
                cv2.putText(grid_image, str(patch_index), 
                           (start_w + 3, start_h + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            
            patch_index += 1
            
    # 4. Display the plot
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_image)
    plt.title(f"{title}\nHighlighted Patches: {len(nonzero_indices)}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_detections(image_path, head_box=None, torso_box=None, pose_keypoints=None, pose_scores=None, keypoint_conf_threshold=0.2):
    """
    Draws all detected bounding boxes and keypoints on an image for verification.
    """
    # 1. Load the image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    # Convert RGB to BGR for OpenCV drawing
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # 2. Draw Head Bounding Box
    if head_box:
        x_min, y_min, x_max, y_max = head_box
        # Draw a blue rectangle for the head
        cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(image_bgr, 'Head', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # 3. Draw Torso Bounding Box
    if torso_box:
        x_min, y_min, x_max, y_max = torso_box
        # Draw a yellow rectangle for the torso
        cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        cv2.putText(image_bgr, 'Torso', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # 4. Draw Pose Keypoints
    if pose_keypoints is not None and pose_scores is not None:
        # Define the MPII keypoint mapping for labeling
        mpii_labels = {
            0: "R-Ankle", 1: "R-Knee", 2: "R-Hip", 3: "L-Hip", 4: "L-Knee", 5: "L-Ankle",
            6: "Pelvis", 7: "Thorax", 8: "Neck", 9: "Head", 10: "R-Wrist",
            11: "R-Elbow", 12: "R-Shoulder", 13: "L-Shoulder", 14: "L-Elbow", 15: "L-Wrist"
        }
        # Define connections (skeleton) for MPII
        skeleton = [
            [0, 1], [1, 2], [2, 6], [3, 6], [3, 4], [4, 5], # Legs
            [6, 7], [7, 8], [8, 9], # Spine
            [7, 12], [12, 11], [11, 10], # Right Arm
            [7, 13], [13, 14], [14, 15]  # Left Arm
        ]

        # Draw skeleton lines
        for p1_idx, p2_idx in skeleton:
            if pose_scores[p1_idx] > keypoint_conf_threshold and pose_scores[p2_idx] > keypoint_conf_threshold:
                pt1 = tuple(pose_keypoints[p1_idx].astype(int))
                pt2 = tuple(pose_keypoints[p2_idx].astype(int))
                cv2.line(image_bgr, pt1, pt2, (0, 255, 0), 2) # Green lines

        # Draw keypoint circles and labels
        for i, (point, score) in enumerate(zip(pose_keypoints, pose_scores)):
            if score > keypoint_conf_threshold:
                x, y = int(point[0]), int(point[1])
                # Draw a red circle for the keypoint
                cv2.circle(image_bgr, (x, y), 5, (0, 0, 255), -1)
                # Put the keypoint index and label
                label = f"{i}:{mpii_labels.get(i, 'Unk')}"
                cv2.putText(image_bgr, label, (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 5. Display the final image
    # Convert BGR back to RGB for matplotlib display
    image_rgb_final = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb_final)
    plt.title("Detection Verification")
    plt.axis('off')
    plt.show()




if __name__ == '__main__':

    
    IMAGE_DIR = "/home/uas-dtu/DTC-Trauma/DTC-Trauma/main/yolo/"
    CSV_PATH = "/home/uas-dtu/DTC-Trauma/output_sorted_updated.csv"
    
    print("Initializing processor for visualization test...")
    pose_processor = HuggingFacePoseProcessor()
    
    dataset = VisionFineTuneDataset(
        image_dir=IMAGE_DIR,
        csv_path=CSV_PATH,
        pose_processor=pose_processor
    )

    if len(dataset) > 0:
        # Get a single sample from the dataset to test
        sample_index = 12 # Choose any valid index
        print(f"\n--- Visualizing Sample Index: {sample_index} ---")
        
        # This will run the full __getitem__ logic
        image_tensor, labels, patch_indices = dataset[sample_index]
        
        # Get the original image path for visualization
        row = dataset.df.iloc[sample_index]
        base_image_name = str(row['image_name']).strip()
        image_path = dataset._find_image_path(base_image_name)

       
        
        print("Initializing processor for visualization test...")
        pose_processor = HuggingFacePoseProcessor()
        
    
        if os.path.exists(image_path):
            
            # Open the image
            image_pil = Image.open(image_path)
            
            # Call the processor to get masks AND debug info
            body_masks, debug_data = pose_processor.generate_body_part_masks(image_pil)
            
            # Now, call the new visualization function with the debug data
            visualize_detections(
                image_path=image_path,
                head_box=debug_data.get("head_box"),
                torso_box=debug_data.get("torso_box"),
                pose_keypoints=debug_data.get("pose_keypoints"),
                pose_scores=debug_data.get("pose_scores")
            )
            
        else:
            print(f"Test image not found at: {image_path}")

        print("Labels:", labels)
        print("Head Patches:", patch_indices["trauma_head"])
        print("Torso Patches:", patch_indices["trauma_torso"])
        print("Upper Ext Patches:", patch_indices["trauma_upper_ext"])
        print("Lower Ext Patches:", patch_indices["trauma_lower_ext"])
        
        # --- Use the visualization function ---
    
        visualize_patches(image_path, patch_indices["trauma_head"], title="upper Patches")
        visualize_patches(image_path, patch_indices["trauma_torso"], title="Torso Patches")
        visualize_patches(image_path, patch_indices["trauma_upper_ext"], title="lower Extremity Patches")
        visualize_patches(image_path, patch_indices["trauma_lower_ext"], title="head Extremity Patches")
        body_masks = pose_processor.generate_body_part_masks(Image.open(image_path))
    
    else:
        print("Dataset is empty, cannot run visualization.")