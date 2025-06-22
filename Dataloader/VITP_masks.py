import torch
import requests
import numpy as np
import cv2
import os
from collections import deque
from PIL import Image
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)
from ultralytics import YOLO

class HuggingFacePoseProcessor:
    """
    A processor to detect persons, estimate pose, and generate masks for various
    body parts (head, torso, limbs) from an image.

    All processing is done in-memory, making it suitable for integration into
    a data loading pipeline (e.g., a PyTorch Dataset).
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Initializing HuggingFacePoseProcessor on device: {self.device}")
        
        # Models for person detection
        self.person_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        self.person_model = RTDetrForObjectDetection.from_pretrained(
            "PekingU/rtdetr_r50vd_coco_o365"
        ).to(self.device)
        
        # Models for pose estimation
        self.pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(
            "usyd-community/vitpose-base-simple"
        ).to(self.device)
        
        # Model for face detection
        self.face_detector = YOLO("yolov8n-face.pt")
        print("All models loaded successfully.")

    # This method goes inside your HuggingFacePoseProcessor class

    def generate_body_part_masks(self, image: Image.Image, conf_threshold=0.2):
        """
        Processes a PIL image to generate head, torso, and limb masks in-memory.
        This version uses a robust fallback for head detection and simple, reliable
        "sausage" masks for limbs.

        Args:
            image (PIL.Image.Image): The input image.
            conf_threshold (float): The confidence score needed for a keypoint to be considered valid.

        Returns:
            dict: A dictionary mapping part names ('head', 'torso', 'larm', etc.)
                to their corresponding mask as a NumPy array.
        """
        image_np_rgb = np.array(image.convert("RGB"))
        image_np = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)
        h, w = image_np.shape[:2]

        body_part_masks = {}
        debug_info = {
            "head_box": None,
            "torso_box": None,
            "pose_keypoints": None,
            "pose_scores": None
            }
        head_mask_generated = False

        # 1. Primary Head Detection (using YOLO)
        # Lower the confidence to catch more faces, e.g., in profile.
        _, face_box = self.detect_face_yolo(image_np_rgb, conf_threshold=0.15)
        if face_box:
            x_min, y_min, x_max, y_max = face_box
            head_mask = np.zeros((h, w), dtype=np.uint8)
            head_mask[y_min:y_max, x_min:x_max] = 255  # White rectangle for the mask
            body_part_masks['head'] = head_mask
            debug_info['head_box'] = face_box
            head_mask_generated = True

        # 2. Process Full Body Pose
        # Use a slightly lower threshold for person detection to be more inclusive
        person_boxes = self.detect_persons(image, threshold=0.25)
        if person_boxes is None:
            return body_part_masks # Return whatever we have (maybe a head mask)

        # Find the person with the largest bounding box
        areas = person_boxes[:, 2] * person_boxes[:, 3]
        max_area_idx = np.argmax(areas)
        person_box = person_boxes[max_area_idx:max_area_idx+1]

        pose_results = self.estimate_pose(image, person_box)
        if not pose_results or not pose_results[0]:
            return body_part_masks

        person_pose = pose_results[0][0]
        keypoints, scores = person_pose["keypoints"], person_pose["scores"]

        # 3. Fallback Head Detection (using Pose Keypoints)
        # If YOLO failed, try to create a head mask from pose data.
        if not head_mask_generated:
            head_top_idx, neck_idx = 9, 8 # MPII format
            if scores[head_top_idx] > conf_threshold and scores[neck_idx] > conf_threshold:
                head_top_pt = keypoints[head_top_idx]
                neck_pt = keypoints[neck_idx]
                
                # Estimate a bounding box for the head
                center_x = int((head_top_pt[0] + neck_pt[0]) / 2)
                head_height = np.linalg.norm(head_top_pt - neck_pt)
                head_width = head_height * 0.9 # A reasonable aspect ratio for a head
                
                x_min = max(0, int(center_x - head_width / 2))
                x_max = min(w, int(center_x + head_width / 2))
                # Get y-bounds and ensure correct order
                y_min_raw, y_max_raw = int(head_top_pt[1]), int(neck_pt[1])
                y_min = max(0, min(y_min_raw, y_max_raw))
                y_max = min(h, max(y_min_raw, y_max_raw))

                head_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.rectangle(head_mask, (x_min, y_min), (x_max, y_max), 255, -1)
                body_part_masks['head'] = head_mask
                
        # 4. Generate Torso Mask (from pose keypoints)
        _, chest_box = self.extract_chest_region(image_np, keypoints, scores)
        if chest_box:
            x_min, y_min, x_max, y_max = chest_box
            torso_mask = np.zeros((h, w), dtype=np.uint8)
            torso_mask[y_min:y_max, x_min:x_max] = 255
            body_part_masks['torso'] = torso_mask
        
        # 5. Generate Limb Masks (using the new "sausage" method)
        limb_keypoints = self.extract_limb_keypoints(keypoints, scores)
        
        # Generate sausage masks for all limbs at once
        # Pass the shape of the original image, the extracted keypoints, and desired thickness
        sausage_limb_masks = self.generate_sausage_masks(image_np.shape, limb_keypoints, thickness=40)
        
        # Update the main dictionary with the new limb masks
        body_part_masks.update(sausage_limb_masks)
                
        return body_part_masks,debug_info

    # --- Core Detection and Extraction Methods ---

    def detect_persons(self, image, threshold=0.3):
        inputs = self.person_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.person_model(**inputs)
        results = self.person_processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=threshold
        )
        result = results[0]
        person_boxes = result["boxes"][result["labels"] == 0]
        if len(person_boxes) == 0:
            return None
        person_boxes = person_boxes.cpu().numpy()
        person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
        person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]
        return person_boxes

    def detect_face_yolo(self, image_np, conf_threshold=0.25):
        """Detects the largest face and returns the crop and its bounding box."""
        results = self.face_detector(image_np,conf=conf_threshold, verbose=False)
        if not results or not results[0].boxes:
            return None, None
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        if len(boxes) == 0:
            return None, None
        best_box = boxes[np.argmax(confs)]
        x_min, y_min, x_max, y_max = map(int, best_box)
        h, w = image_np.shape[:2]
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)
        if x_max <= x_min or y_max <= y_min:
            return None, None
        face_crop = image_np[y_min:y_max, x_min:x_max]
        face_box = (x_min, y_min, x_max, y_max)
        return face_crop, face_box

    def estimate_pose(self, image, person_boxes, threshold=0.5):
        inputs = self.pose_processor(image, boxes=[person_boxes], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.pose_model(**inputs)
        return self.pose_processor.post_process_pose_estimation(
            outputs, boxes=[person_boxes], threshold=threshold
        )
    
    def extract_chest_region(self, image_np, keypoints, scores):
        """Extracts the torso region and returns the crop and its bounding box."""
        indices = [5, 6, 11, 12] # Shoulders and hips
        chest_keypoints = np.array([keypoints[i] for i in indices if scores[i] > 0.2])
        if len(chest_keypoints) < 2:
            return None, None
        x_min, y_min = np.min(chest_keypoints, axis=0).astype(int)
        x_max, y_max = np.max(chest_keypoints, axis=0).astype(int)
        h, w = image_np.shape[:2]
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)
        if x_max <= x_min or y_max <= y_min:
            return None, None
        chest_crop = image_np[y_min:y_max, x_min:x_max]
        chest_box = (x_min, y_min, x_max, y_max)
        return chest_crop, chest_box

    # --- Helper Methods for Limb Mask Generation ---
    
    def extract_limb_keypoints(self, keypoints, scores):
        """
        Extracts limb keypoints using the MPII 16-point format.
        """

        limbs = {
            # 'limb_name': [Shoulder/Hip, Elbow/Knee, Wrist/Ankle]
            'rarm': [12, 11, 10], # R Shoulder, R Elbow, R Wrist
            'larm': [13, 14, 15], # L Shoulder, L Elbow, L Wrist
            'rleg': [2, 1, 0],    # R Hip, R Knee, R Ankle
            'lleg': [3, 4, 5],    # L Hip, L Knee, L Ankle
        }
        
        limb_keypoints_map = {}
        for limb_name, indices in limbs.items():
            limb_points = []
            for idx in indices:
                # Defensive check, although the indices should now be correct
                if idx < len(keypoints):
                    x, y = keypoints[idx]
                    score = scores[idx] if idx < len(scores) else 0.0
                    conf = 1.0 if score > 0.2 else 0.0
                    limb_points.append([x.item(), y.item(), score.item(), conf])
                else:
                    # This case should not be hit now, but it's good practice
                    limb_points.append([0.0, 0.0, 0.0, 0.0])
            limb_keypoints_map[limb_name] = limb_points

        # Ensure a consistent order for the output array for the rest of the pipeline
        # The original order was larm, rarm, lleg, rleg. Let's maintain that.
        ordered_limbs = ['larm', 'rarm', 'lleg', 'rleg']
        final_limb_keypoints = [limb_keypoints_map[name] for name in ordered_limbs]
        
        # Transpose to the format expected by calculate_sigma
        return np.transpose(np.array(final_limb_keypoints), (1, 2, 0))
    
    def gaussian(self, x, y, sigma):
        return np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    
    def calculate_sigma(self, keypoints, k=0.24):
        sigma = np.ones([1, 3, 4], dtype=np.float16)
        for i in range(keypoints.shape[2]): 
            sigma[0, 0, i] = np.linalg.norm(keypoints[0, :2, i] - keypoints[1, :2, i]) * k * keypoints[0,3,i] * keypoints[1,3,i]
            sigma[0, 1, i] = np.linalg.norm(keypoints[1, :2, i] - keypoints[2, :2, i]) * k * keypoints[1,3,i] * keypoints[2,3,i]
            sigma[0, 2, i] = np.linalg.norm(keypoints[2, :2, i] - keypoints[0, :2, i]) * k * keypoints[2,3,i] * keypoints[0,3,i]
        return sigma
    
    def bfs(self, image, startrow, startcol, visited):
        q = deque([(startrow, startcol)])
        visited[startrow][startcol] = True
        rows, cols = image.shape
        max_x, min_x, max_y, min_y = 0, rows, 0, cols
        
        while q:
            r, c = q.popleft()
            if image[r, c] != 0:
                max_x, min_x = max(max_x, r), min(min_x, r)
                max_y, min_y = max(max_y, c), min(min_y, c)
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = r - dr, c - dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                        visited[nr][nc] = True
                        q.append((nr, nc))
        return max_x, max_y, min_x, min_y
    
    def get_gaussian_and_bfs_limbs(self, image, keypoints, sigmas, conf_thres=0.0):
        limb_images_list = []
        height, width = image.shape[:2]
        splatted_images = [np.zeros((height, width, 3), dtype=np.float64) for _ in range(4)]
        sigmas_mean = np.mean(sigmas, axis=1)
        radius = (2 * sigmas_mean).astype(np.int16)
        
        # Pre-calculate Gaussian weights
        gaussian_weights_list = []
        for j in range(sigmas_mean.shape[1]):
            r = radius[0,j]
            x, y = np.arange(-r, r + 1), np.arange(-r, r + 1)
            sigma = sigmas_mean[0,j]
            X, Y = np.meshgrid(x, y)
            weights = self.gaussian(X, Y, sigma)
            gaussian_weights_list.append(weights / np.sum(weights) if np.sum(weights) > 0 else weights)
    
        # Interpolate points along limbs
        midpoints = []
        for i in range(keypoints.shape[0] - 1):
            for r1, r2 in [(4, 1), (3, 2), (2, 3), (1, 4)]:
                midpoint = (
                    (r1 * keypoints[i, 0, :] + r2 * keypoints[i+1, 0, :]) / 5,
                    (r1 * keypoints[i, 1, :] + r2 * keypoints[i+1, 1, :]) / 5,
                    (r1 * keypoints[i, 2, :] + r2 * keypoints[i+1, 2, :]) / 5,
                    keypoints[i, 3, :] * keypoints[i+1, 3, :]
                )
                midpoints.append(np.expand_dims(np.stack(midpoint), axis=0))
        
        all_points = np.concatenate([keypoints] + midpoints, axis=0) if midpoints else keypoints

        # Vectorized calculations for ROI and indices
        conf_mask = all_points[:, -2, :] >= conf_thres
        all_points[:, -2, :] = conf_mask
        calc = np.zeros([all_points.shape[0], 8, all_points.shape[2]], dtype=np.int16)
        calc[:, 0, :] = np.maximum(0, all_points[:, 0, :] - radius)
        calc[:, 1, :] = np.minimum(width, all_points[:, 0, :] + radius + 1)
        calc[:, 2, :] = np.maximum(0, all_points[:, 1, :] - radius)
        calc[:, 3, :] = np.minimum(height, all_points[:, 1, :] + radius + 1)
        calc[:, 4, :] = np.maximum(0, radius - all_points[:, 0, :])
        calc[:, 5, :] = np.minimum(2 * radius + 1, radius - all_points[:, 0, :] + width)
        calc[:, 6, :] = np.maximum(0, radius - all_points[:, 1, :])
        calc[:, 7, :] = np.minimum(2 * radius + 1, radius - all_points[:, 1, :] + height)

        for i in range(all_points.shape[2]): # For each limb
            for j in range(all_points.shape[0]): # For each point
                if all_points[j, 2, i] == 0 or all_points[j, 3, i] == 0: continue
                
                c = calc[j, :, i]
                roi_img = image[c[2]:c[3], c[0]:c[1]]
                roi_splat = splatted_images[i][c[2]:c[3], c[0]:c[1]]
                roi_gauss = gaussian_weights_list[i][c[6]:c[7], c[4]:c[5]]
                
                # Ensure shapes match before broadcasting
                h_min, w_min = min(roi_img.shape[0], roi_gauss.shape[0]), min(roi_img.shape[1], roi_gauss.shape[1])
                roi_splat[:h_min, :w_min] += roi_gauss[:h_min, :w_min, np.newaxis] * roi_img[:h_min, :w_min]
            
            # Normalize and process final limb mask
            splatted_image = splatted_images[i]
            max_val = np.max(splatted_image)
            limb_image = np.array(splatted_image / max_val * 255, dtype=np.uint8) if max_val > 0 else np.zeros_like(splatted_image, dtype=np.uint8)
            
            try:
                img_gray = cv2.cvtColor(limb_image, cv2.COLOR_BGR2GRAY)
                if np.any(img_gray > 0):
                    img_resized = cv2.resize(img_gray, (img_gray.shape[1]//8, img_gray.shape[0]//8))
                    visited = np.zeros(img_resized.shape, dtype=bool)
                    max_x, max_y, min_x, min_y = self.bfs(img_resized, 0, 0, visited)
                    if max_x > min_x and max_y > min_y:
                        image_cropped = cv2.resize(limb_image[8*min_x:8*max_x, 8*min_y:8*max_y], (256, 256))
                        limb_images_list.append(image_cropped)
            except Exception as e:
                # print(f"Warning: Could not process limb mask {i}. Error: {e}")
                continue
                
        return limb_images_list
    def generate_sausage_masks(self, image_shape, limb_keypoints, thickness=30):
        h, w = image_shape[:2]
        masks = {}
        
        # NOTE: The limb_keypoints array is indexed as [joint, info, limb]
        # Limbs are ordered: rarm, larm, rleg, lleg based on our new MPII-based keypoint extraction
        # Let's create a mapping for clarity
        limb_map = {
            'rarm': limb_keypoints[:, :, 0],
            'larm': limb_keypoints[:, :, 1],
            'rleg': limb_keypoints[:, :, 2],
            'lleg': limb_keypoints[:, :, 3],
        }

        for name, points in limb_map.items():
            mask = np.zeros((h, w), dtype=np.uint8)
            # points shape is (3, 4) -> [joint_idx][x, y, score, conf]
            
            # Define the two segments for each limb
            # Segment 1: Shoulder/Hip to Elbow/Knee
            # Segment 2: Elbow/Knee to Wrist/Ankle
            for i in range(2): # 0 for segment 1, 1 for segment 2
                pt1 = (int(points[i, 0]), int(points[i, 1]))
                pt2 = (int(points[i+1, 0]), int(points[i+1, 1]))
                
                # Check confidence of both endpoints before drawing
                conf1 = points[i, 3]
                conf2 = points[i+1, 3]
                
                if conf1 > 0 and conf2 > 0:
                    cv2.line(mask, pt1, pt2, color=255, thickness=thickness)

            masks[name] = mask
            
        return masks
    