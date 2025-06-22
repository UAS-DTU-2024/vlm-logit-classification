from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
  

def extract_line_region(image, pt1, pt2, thickness=50):
  
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Draw line with specified thickness
    cv2.line(mask, pt1, pt2, color=255, thickness=thickness)
    
    # Use more appropriate kernel size based on thickness
    kernel_size = max(thickness // 4, 5)  # Adaptive kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Apply Gaussian blur for smoother edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Apply mask to image
    result = cv2.bitwise_and(image, image, mask=mask)
    return result, mask

def create_arm_mask(image, shoulder_pt, elbow_pt, wrist_pt, thickness=50):

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Calculate adaptive thickness based on limb length
    upper_arm_length = np.linalg.norm(np.array(elbow_pt) - np.array(shoulder_pt))
    forearm_length = np.linalg.norm(np.array(wrist_pt) - np.array(elbow_pt))
    
    # Adaptive thickness based on limb proportions
    upper_thickness = max(int(upper_arm_length * 0.3), thickness)
    lower_thickness = max(int(forearm_length * 0.25), thickness - 10)
    
    # Draw upper arm (shoulder to elbow)
    cv2.line(mask, shoulder_pt, elbow_pt, color=255, thickness=upper_thickness)
    
    # Draw forearm (elbow to wrist)
    cv2.line(mask, elbow_pt, wrist_pt, color=255, thickness=lower_thickness)
    
    # Add joint regions for better connectivity
    joint_radius = max(thickness // 3, 15)
    cv2.circle(mask, shoulder_pt, joint_radius, 255, -1)
    cv2.circle(mask, elbow_pt, joint_radius, 255, -1)
    cv2.circle(mask, wrist_pt, joint_radius, 255, -1)
    
    # Smooth the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    
    # Apply mask to image
    result = cv2.bitwise_and(image, image, mask=mask)
    return result, mask

def validate_keypoint(keypoint, image_shape):

    x, y = keypoint[:2]
    height, width = image_shape[:2]
    
    # Check if coordinates are valid and within bounds
    if x <= 0 or y <= 0 or x >= width or y >= height:
        return False
    
    if len(keypoint) > 2 and keypoint[2] < 0.5:  # Low confidence threshold
        return False
    
    return True

def extract_leg_segment(image, keypoints, leg_side='left', thickness=50): 
    
    if keypoints.numel() == 0:
        print("No keypoints detected!")
        return None, None
    
    # Define keypoint indices for YOLO pose model
    if leg_side == 'left':
        hip_idx, knee_idx, ankle_idx = 11, 13, 15  # Left hip, knee, ankle
    else:
        hip_idx, knee_idx, ankle_idx = 12, 14, 16  # Right hip, knee, ankle
    
    try:
        # Extract keypoints
        hip = keypoints[0, hip_idx].cpu().numpy()
        knee = keypoints[0, knee_idx].cpu().numpy()
        ankle = keypoints[0, ankle_idx].cpu().numpy()
        
        # # Validate keypoints
        # if not all(validate_keypoint(kp, image.shape) for kp in [hip, knee, ankle]):
        #     print(f"Invalid keypoints detected for {leg_side} leg")
        #     return None, None
        
        hip_pt = tuple(map(int, hip[:2]))
        knee_pt = tuple(map(int, knee[:2]))
        ankle_pt = tuple(map(int, ankle[:2]))
        
        # Create leg mask
        result, mask = create_arm_mask(image, hip_pt, knee_pt, ankle_pt, thickness)
        
        return result, mask
        
    except Exception as e:
        print(f"Error processing {leg_side} leg: {str(e)}")
        return None, None

def extract_arm_segment(image, keypoints, arm_side='left', thickness=50):
   
    if keypoints.numel() == 0:
        print("No keypoints detected!")
        return None, None
    
    # Define keypoint indices for YOLO pose model
    if arm_side == 'left':
        shoulder_idx, elbow_idx, wrist_idx = 5, 7, 9  # Left shoulder, elbow, wrist
    else:
        shoulder_idx, elbow_idx, wrist_idx = 6, 8, 10  # Right shoulder, elbow, wrist
    
    try:
        # Extract keypoints
        shoulder = keypoints[0, shoulder_idx].cpu().numpy()
        elbow = keypoints[0, elbow_idx].cpu().numpy()
        wrist = keypoints[0, wrist_idx].cpu().numpy()
        
        # # Validate keypoints
        # if not all(validate_keypoint(kp, image.shape) for kp in [shoulder, elbow, wrist]):
        #     print(f"Invalid keypoints detected for {arm_side} arm")
        #     return None, None
        

        shoulder_pt = tuple(map(int, shoulder[:2]))
        elbow_pt = tuple(map(int, elbow[:2]))
        wrist_pt = tuple(map(int, wrist[:2]))
        # Create arm mask
        result, mask = create_arm_mask(image, shoulder_pt, elbow_pt, wrist_pt, thickness)
        
        return result, mask
        
    except Exception as e:
        print(f"Error processing {arm_side} arm: {str(e)}")
        return None, None

def get_nonzero_patch_indices(image_path, target_size=(336, 336), patch_size=(14, 14)):
    
    if isinstance(image_path, str):
        
        image = Image.open(image_path)
        
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        image_array = np.array(image)
 
    elif isinstance(image_path, np.ndarray):
        # Handle numpy array input - use cv2.resize for better control
        if len(image_path.shape) == 3:
            # For color images, resize directly
            image_array = cv2.resize(image_path, target_size, interpolation=cv2.INTER_LANCZOS4)
        else:
            # For grayscale images
            image_array = cv2.resize(image_path, target_size, interpolation=cv2.INTER_LANCZOS4)
        
    else:
        raise ValueError("Input must be image path string or numpy array")
    
    
    if len(image_array.shape) == 3:
        if image_array.shape[2] == 4:  # RGBA
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)
        elif image_array.shape[2] == 3:  # RGB
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    height, width = image_array.shape
    patch_h, patch_w = patch_size
    
    # Calculate number of patches
    n_patches_h = height // patch_h
    n_patches_w = width // patch_w
    
    nonzero_indices = []
    patch_index = 1  # Start from 1 as specified
    
    # Iterate through patches row-wise
    for row in range(n_patches_h):
        for col in range(n_patches_w):
            # Extract patch coordinates
            start_h = row * patch_h
            end_h = start_h + patch_h
            start_w = col * patch_w
            end_w = start_w + patch_w
            
            # Extract patch
            patch = image_array[start_h:end_h, start_w:end_w]
            
            # Check if patch has any non-zero pixels
            if np.any(patch > 0):
                nonzero_indices.append(patch_index)
            
            patch_index += 1
    
    return nonzero_indices


def visualize_patches(image_path,nonzero_indices, target_size=(336, 336), patch_size=(14, 14)):
    
    if isinstance(image_path, str):
        image = Image.open(image_path)
  
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    
        image_array = np.array(image)
    elif isinstance(image_path, np.ndarray):

        image_array = cv2.resize(image_path, target_size, interpolation=cv2.INTER_LANCZOS4)

    else:
        image_array = image_path
    
   
    if len(image_array.shape) == 3:
        actual_height, actual_width = image_array.shape[:2]
    else:
        actual_height, actual_width = image_array.shape
    
    print(f"Final image dimensions: {actual_width}x{actual_height}")
    
    if actual_height != target_size[1] or actual_width != target_size[0]:
       
        image_array = cv2.resize(image_array, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original resized image
    if len(image_array.shape) == 3:
        ax1.imshow(image_array)
    else:
        ax1.imshow(image_array, cmap='gray')
    ax1.set_title(f'Resized Image ({image_array.shape[1]}x{image_array.shape[0]})')
    ax1.axis('off')
    
    # Patch grid overlay
    height, width = image_array.shape[:2]
    patch_h, patch_w = patch_size
    n_patches_h = height // patch_h
    n_patches_w = width // patch_w
    
    print(f"Grid dimensions: {n_patches_w}x{n_patches_h} patches")
    print(f"Total patches: {n_patches_h * n_patches_w}")
    
    
    grid_image = np.zeros((height, width, 3), dtype=np.uint8)
    if len(image_array.shape) == 3:
        grid_image = image_array.copy()
    else:
        grid_image = np.stack([image_array] * 3, axis=-1)
    
    # Draw grid lines and highlight non-zero patches
    patch_index = 1
    for row in range(n_patches_h):
        for col in range(n_patches_w):
            start_h = row * patch_h
            end_h = start_h + patch_h
            start_w = col * patch_w
            end_w = start_w + patch_w
            
            # Draw rectangle border
            cv2.rectangle(grid_image, (start_w, start_h), (end_w-1, end_h-1), (255, 255, 255), 1)
            
            # Highlight non-zero patches
            if patch_index in nonzero_indices:
                cv2.rectangle(grid_image, (start_w, start_h), (end_w-1, end_h-1), (0, 255, 0), 2)
                # Add patch number
                cv2.putText(grid_image, str(patch_index), 
                           (start_w + 2, start_h + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            
            patch_index += 1
    
    ax2.imshow(grid_image)
    ax2.set_title(f'Patch Grid (Non-zero patches: {len(nonzero_indices)})')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return nonzero_indices


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


if __name__ == "__main__":
        
    model = YOLO('yolo11n-pose.pt')
    results = model("/Users/arjuntomar/Desktop/cropped/IMG_2289_crop1.jpg")

    for result in results:
        keypoints = result.keypoints.data
        
        if keypoints.numel() > 0:
            
            img = cv2.imread("/Users/arjuntomar/Desktop/cropped/IMG_2289_crop1.jpg")
            if img is None:
                print("Error: Could not load image")
                continue
                
            height, width = img.shape[:2]
            print(f"Image dimensions: {width}x{height}")
            
            # Extract left arm
            left_arm, left_mask = extract_leg_segment(img, keypoints, 'left', thickness=100)
            
            # Extract right arm
            right_arm, right_mask = extract_leg_segment(img, keypoints, 'right', thickness=100)
            
    
            if left_arm is not None and right_arm is not None:
                combined_mask = cv2.bitwise_or(left_mask, right_mask)
                combined_result = cv2.bitwise_and(img, img, mask=combined_mask)
                cv2.imshow("Upper Ext.", combined_result)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No keypoints detected in the image")
    
    nonzero_patches = get_nonzero_patch_indices(combined_result)
    print(f"Non-zero patch indices: {nonzero_patches}")
    print(f"Total patches with content: {len(nonzero_patches)}")
    
    
    visualize_patches("/Users/arjuntomar/Desktop/cropped/IMG_2289_crop1.jpg",nonzero_patches)