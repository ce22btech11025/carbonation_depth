# """Segmentation Module using SAM 2 - WITH GPU ACCELERATION

# Performs precise object segmentation with automatic GPU detection and acceleration

# """

# import cv2
# import numpy as np
# from typing import Dict, List, Tuple, Optional
# import torch

# try:
#     from ultralytics import SAM
# except ImportError:
#     print("WARNING: ultralytics not installed. Install with: pip install ultralytics")
#     SAM = None


# class SegmentationModule:
#     def __init__(self, model_path: str = "sam2_b.pt"):
#         """Initialize SAM 2 with automatic GPU detection

#         Args:
#             model_path: Path to SAM 2 model weights
#         """
#         self.model_path = model_path
#         self.model = None
#         self.segmentation_results = {}

#         # Detect available device (GPU or CPU)
#         self.device = self._detect_device()
#         print(f"🖥️  Device detected: {self.device}")

#     def _detect_device(self) -> str:
#         """Detect and return available computation device

#         Returns:
#             Device string: 'cuda', 'mps', or 'cpu'
#         """
#         # Check for NVIDIA GPU (CUDA)
#         if torch.cuda.is_available():
#             device = 'cuda'
#             print(f"✓ CUDA GPU available: {torch.cuda.get_device_name(0)}")
#             print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
#         # Check for Apple Silicon GPU (MPS)
#         elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#             device = 'mps'
#             print("✓ Apple Silicon GPU (MPS) available")
#         # Fallback to CPU
#         else:
#             device = 'cpu'
#             print("⚠ No GPU available, using CPU (slower)")

#         return device

#     def initialize_sam_model(self) -> None:
#         """Load and initialize SAM 2 model with GPU acceleration"""
#         if SAM is None:
#             raise ImportError("ultralytics not installed. Run: pip install ultralytics")

#         try:
#             print(f"\n[SAM] Loading model on {self.device}...")
#             self.model = SAM(self.model_path)

#             # Explicitly set device
#             if self.device == 'cuda':
#                 self.model.to('cuda')
#                 print(f"✓ SAM 2 model loaded on GPU (CUDA)")
#             elif self.device == 'mps':
#                 self.model.to('mps')
#                 print(f"✓ SAM 2 model loaded on Apple GPU (MPS)")
#             else:
#                 self.model.to('cpu')
#                 print(f"✓ SAM 2 model loaded on CPU")

#         except Exception as e:
#             print(f"⚠ Error loading model: {e}")
#             print("  Downloading model automatically...")
#             self.model = SAM("sam2_b.pt")

#             # Set device after download
#             if self.device == 'cuda':
#                 self.model.to('cuda')
#             elif self.device == 'mps':
#                 self.model.to('mps')

#     def segment_objects(self, image: np.ndarray,
#                         points: Optional[List[List[int]]] = None,
#                         labels: Optional[List[int]] = None) -> Dict:
#         """Segment objects using SAM 2 with GPU acceleration

#         Args:
#             image: Input preprocessed image
#             points: Optional prompt points [[x, y], ...]
#             labels: Optional point labels (1=foreground, 0=background)

#         Returns:
#             Segmentation results
#         """
#         if self.model is None:
#             self.initialize_sam_model()

#         print(f"\n[SAM] Running segmentation on {self.device}...")

#         # Automatic segmentation if no points provided
#         if points is None:
#             try:
#                 # Use device parameter for GPU acceleration
#                 results = self.model.predict(
#                     source=image, 
#                     device=self.device,  # GPU acceleration here!
#                     save=False, 
#                     verbose=False
#                 )
#                 print(f"✓ Automatic segmentation complete: {len(results)} result(s)")

#             except Exception as e:
#                 print(f"⚠ Error during segmentation: {e}")
#                 results = []
#         else:
#             # Guided segmentation with prompts
#             results = self.model(
#                 image, 
#                 points=points, 
#                 labels=labels, 
#                 device=self.device,  # GPU acceleration
#                 save=False, 
#                 verbose=False
#             )
#             print(f"✓ Guided segmentation complete with {len(points)} prompts")

#         return results

#     def extract_masks(self, results) -> Dict[str, np.ndarray]:
#         """Extract individual masks for different objects

#         Args:
#             results: Segmentation results from SAM 2

#         Returns:
#             Dictionary of masks for different objects
#         """
#         masks = {}

#         if hasattr(results, 'masks') and results.masks is not None:
#             # Move masks from GPU to CPU if needed
#             if hasattr(results.masks.data, 'cpu'):
#                 all_masks = results.masks.data.cpu().numpy()
#             else:
#                 all_masks = results.masks.data.numpy()

#             # Sort masks by area (largest first)
#             mask_areas = [np.sum(mask) for mask in all_masks]
#             sorted_indices = np.argsort(mask_areas)[::-1]

#             # Assume largest = concrete block, second = scale
#             if len(sorted_indices) >= 1:
#                 masks['concrete_block'] = all_masks[sorted_indices[0]].astype(np.uint8) * 255
#                 print(f"✓ Concrete block: {mask_areas[sorted_indices[0]]:.0f} pixels")

#             if len(sorted_indices) >= 2:
#                 masks['scale'] = all_masks[sorted_indices[1]].astype(np.uint8) * 255
#                 print(f"✓ Scale: {mask_areas[sorted_indices[1]]:.0f} pixels")

#             # Additional objects
#             for i in range(2, min(len(sorted_indices), 5)):
#                 masks[f'object_{i-1}'] = all_masks[sorted_indices[i]].astype(np.uint8) * 255

#         return masks

#     def detect_scale_boundaries(self, scale_mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
#         """Detect precise boundaries of the scale/ruler

#         Args:
#             scale_mask: Binary mask of the scale

#         Returns:
#             Tuple of (edge image, boundary info dict)
#         """
#         edges = cv2.Canny(scale_mask, 50, 150)
#         contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         if not contours:
#             return edges, {}

#         largest_contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest_contour)

#         boundary_info = {
#             'x': x,
#             'y': y,
#             'width': w,
#             'height': h,
#             'contour': largest_contour,
#             'area': cv2.contourArea(largest_contour)
#         }

#         print(f"✓ Scale boundaries: {w}x{h} pixels")
#         return edges, boundary_info

#     def detect_concrete_boundaries(self, concrete_mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
#         """Detect precise boundaries of concrete block

#         Args:
#             concrete_mask: Binary mask of concrete block

#         Returns:
#             Tuple of (edge image, boundary info dict)
#         """
#         edges = cv2.Canny(concrete_mask, 50, 150)
#         contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         if not contours:
#             return edges, {}

#         largest_contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest_contour)

#         boundary_info = {
#             'x': x,
#             'y': y,
#             'width': w,
#             'height': h,
#             'contour': largest_contour,
#             'area': cv2.contourArea(largest_contour),
#             'perimeter': cv2.arcLength(largest_contour, True)
#         }

#         print(f"✓ Concrete boundaries: {w}x{h} pixels")
#         return edges, boundary_info

#     def segment_and_extract(self, image: np.ndarray) -> Dict:
#         """Complete segmentation pipeline with GPU acceleration

#         Args:
#             image: Preprocessed input image

#         Returns:
#             Dictionary containing all segmentation results
#         """
#         # Initialize model if needed
#         if self.model is None:
#             self.initialize_sam_model()

#         # Perform GPU-accelerated segmentation
#         results = self.segment_objects(image)

#         # Extract masks
#         masks = self.extract_masks(results[0] if isinstance(results, list) else results)

#         # Detect boundaries
#         segmentation_data = {'masks': masks}

#         if 'concrete_block' in masks:
#             edges, boundary_info = self.detect_concrete_boundaries(masks['concrete_block'])
#             segmentation_data['concrete_boundaries'] = boundary_info
#             segmentation_data['concrete_edges'] = edges

#         if 'scale' in masks:
#             edges, boundary_info = self.detect_scale_boundaries(masks['scale'])
#             segmentation_data['scale_boundaries'] = boundary_info
#             segmentation_data['scale_edges'] = edges

#         self.segmentation_results = segmentation_data

#         # Clear GPU cache if using CUDA
#         if self.device == 'cuda':
#             torch.cuda.empty_cache()

#         return segmentation_data

#     def visualize_segmentation(self, image: np.ndarray, 
#                               masks: Dict[str, np.ndarray],
#                               output_path: Optional[str] = None) -> np.ndarray:
#         """Visualize segmentation results with color overlays

#         Args:
#             image: Original image
#             masks: Dictionary of masks
#             output_path: Optional save path

#         Returns:
#             Visualization image
#         """
#         vis_image = image.copy()

#         colors = {
#             'concrete_block': (0, 255, 0),    # Green
#             'scale': (255, 0, 0),              # Blue
#             'object_1': (0, 255, 255),         # Yellow
#             'object_2': (255, 0, 255),         # Magenta
#         }

#         for name, mask in masks.items():
#             if name in colors:
#                 color = colors[name]
#                 colored_mask = np.zeros_like(vis_image)
#                 colored_mask[mask > 0] = color
#                 vis_image = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)

#         if output_path:
#             cv2.imwrite(output_path, vis_image)
#             print(f"✓ Visualization saved to {output_path}")

#         return vis_image

#     def get_device_info(self) -> Dict:
#         """Get detailed device information

#         Returns:
#             Dictionary with device details
#         """
#         info = {
#             'device': self.device,
#             'device_name': 'CPU',
#             'memory_gb': 0,
#             'cuda_available': torch.cuda.is_available(),
#             'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
#         }

#         if self.device == 'cuda':
#             info['device_name'] = torch.cuda.get_device_name(0)
#             info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
#             info['cuda_version'] = torch.version.cuda

#         return info


# if __name__ == "__main__":
#     # Example usage with GPU acceleration
#     print("="*60)
#     print("SAM 2 Segmentation Module - GPU Accelerated")
#     print("="*60)

#     segmenter = SegmentationModule()

#     # Print device info
#     device_info = segmenter.get_device_info()
#     print(f"\nDevice: {device_info['device']}")
#     print(f"Name: {device_info['device_name']}")
#     if device_info['memory_gb'] > 0:
#         print(f"Memory: {device_info['memory_gb']:.2f} GB")

#     # Load and segment image
#     image = cv2.imread("preprocessed_output.jpg")
#     if image is not None:
#         results = segmenter.segment_and_extract(image)

#         if results.get('masks'):
#             segmenter.visualize_segmentation(
#                 image,
#                 results['masks'],
#                 output_path="segmentation_output.jpg"
#             )

#         print("\n✓ Segmentation complete!")

"""Segmentation Module using SAM 2 - WITH GPU ACCELERATION

Enhanced with automatic GPU detection, memory optimization, and faster inference
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import time

try:
    from ultralytics import SAM
    from ultralytics.models.sam import Predictor as SAMPredictor
except ImportError:
    print("WARNING: ultralytics not installed. Install with: pip install ultralytics")
    SAM = None


class SegmentationModule:
    def __init__(self, model_path: str = "sam2_b.pt"):
        """Initialize SAM 2 with automatic GPU detection and optimization"""
        self.model_path = model_path
        self.model = None
        self.predictor = None
        self.segmentation_results = {}

        # Detect available device (GPU or CPU)
        self.device = self._detect_device()
        self.device_info = self._get_detailed_device_info()
        
        print(f"🖥️  Device detected: {self.device_info['device_name']}")
        if self.device == 'cuda':
            print(f"  GPU Memory: {self.device_info['memory_gb']:.2f} GB")
            print(f"  CUDA Version: {self.device_info['cuda_version']}")

    def _detect_device(self) -> str:
        """Detect and return available computation device with optimization"""
        # Check for NVIDIA GPU (CUDA)
        if torch.cuda.is_available():
            device = 'cuda'
            # Set CUDA optimization flags
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        # Check for Apple Silicon GPU (MPS)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        # Fallback to CPU
        else:
            device = 'cpu'
            
        return device

    def _get_detailed_device_info(self) -> Dict:
        """Get detailed device information"""
        info = {
            'device': self.device,
            'device_name': 'CPU',
            'memory_gb': 0,
            'cuda_available': torch.cuda.is_available(),
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        }

        if self.device == 'cuda':
            info['device_name'] = torch.cuda.get_device_name(0)
            info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            info['cuda_version'] = torch.version.cuda
            info['current_memory_gb'] = torch.cuda.memory_allocated(0) / 1e9
        elif self.device == 'mps':
            info['device_name'] = 'Apple Silicon GPU'

        return info

    def initialize_sam_model(self) -> None:
        """Load and initialize SAM 2 model with GPU acceleration and optimization"""
        if SAM is None:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

        try:
            print(f"\n[SAM] Loading model on {self.device} with optimization...")
            
            # Clear GPU cache if using CUDA
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            start_time = time.time()
            
            # Load model with device specification
            self.model = SAM(self.model_path)
            
            # Move model to appropriate device
            self.model.to(self.device)
            
            # Initialize predictor for faster inference
            self.predictor = SAMPredictor(self.model.model)
            
            load_time = time.time() - start_time
            
            print(f"✓ SAM 2 model loaded in {load_time:.2f}s on {self.device.upper()}")
            
            # Print model information
            if hasattr(self.model, 'model'):
                param_count = sum(p.numel() for p in self.model.model.parameters())
                print(f"  Model parameters: {param_count:,}")
                
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            print("  Downloading model automatically...")
            
            # Fallback with automatic download
            self.model = SAM("sam2_b.pt")
            self.model.to(self.device)
            self.predictor = SAMPredictor(self.model.model)

    def optimize_image_for_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Optimize image size for faster segmentation while maintaining quality"""
        h, w = image.shape[:2]
        
        # Calculate optimal size (maintain aspect ratio)
        max_dim = 1024  # Good balance between speed and quality
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            optimized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"  Image optimized: {w}x{h} -> {new_w}x{new_h}")
            return optimized_image
        
        return image

    def segment_objects(self, image: np.ndarray,
                        points: Optional[List[List[int]]] = None,
                        labels: Optional[List[int]] = None) -> Dict:
        """Segment objects using SAM 2 with GPU acceleration and optimization"""
        if self.model is None:
            self.initialize_sam_model()

        print(f"\n[SAM] Running segmentation on {self.device}...")
        start_time = time.time()

        # Optimize image size for faster processing
        optimized_image = self.optimize_image_for_segmentation(image)
        
        # Automatic segmentation if no points provided
        if points is None:
            try:
                # Use optimized settings for faster inference
                results = self.model.predict(
                    source=optimized_image, 
                    device=self.device,
                    save=False, 
                    verbose=False,
                    conf=0.25,  # Lower confidence for more detections
                    imgsz=1024,  # Standard size for better performance
                )
                
                inference_time = time.time() - start_time
                print(f"✓ Automatic segmentation complete in {inference_time:.2f}s")
                print(f"  Detected {len(results[0]) if len(results) > 0 else 0} objects")

            except Exception as e:
                print(f"⚠ Error during segmentation: {e}")
                # Fallback to CPU if GPU fails
                if self.device != 'cpu':
                    print("  Falling back to CPU...")
                    self.device = 'cpu'
                    self.model.to('cpu')
                    return self.segment_objects(image, points, labels)
                results = []
        else:
            # Guided segmentation with prompts
            results = self.model(
                optimized_image, 
                points=points, 
                labels=labels, 
                device=self.device,
                save=False, 
                verbose=False
            )
            inference_time = time.time() - start_time
            print(f"✓ Guided segmentation complete in {inference_time:.2f}s")

        return results

    def extract_masks(self, results) -> Dict[str, np.ndarray]:
        """Extract individual masks for different objects with area-based filtering"""
        masks = {}

        if hasattr(results, 'masks') and results.masks is not None:
            # Move masks from GPU to CPU if needed
            if hasattr(results.masks.data, 'cpu'):
                all_masks = results.masks.data.cpu().numpy()
            else:
                all_masks = results.masks.data.numpy()

            # Get confidence scores if available
            confidences = None
            if hasattr(results, 'boxes') and results.boxes is not None:
                if hasattr(results.boxes.conf, 'cpu'):
                    confidences = results.boxes.conf.cpu().numpy()
                else:
                    confidences = results.boxes.conf.numpy()

            # Sort masks by area (largest first) or confidence
            mask_areas = [np.sum(mask) for mask in all_masks]
            sorted_indices = np.argsort(mask_areas)[::-1]

            # Filter masks by minimum area (remove very small detections)
            min_area = 1000  # Minimum pixels for valid mask
            valid_indices = [idx for idx in sorted_indices if mask_areas[idx] > min_area]

            if not valid_indices:
                print("⚠ No valid masks found after area filtering")
                return masks

            # Assume largest = concrete block, look for scale in remaining masks
            concrete_candidate = valid_indices[0]
            masks['concrete_block'] = all_masks[concrete_candidate].astype(np.uint8) * 255
            print(f"✓ Concrete block: {mask_areas[concrete_candidate]:.0f} pixels")

            # Look for scale (typically rectangular, moderate size)
            scale_candidate = None
            for idx in valid_indices[1:]:
                mask = all_masks[idx]
                # Calculate aspect ratio and solidity to identify scale/ruler
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                    
                    # Scale typically has high aspect ratio (long and narrow)
                    if aspect_ratio > 3 and mask_areas[idx] > 5000:
                        scale_candidate = idx
                        break

            if scale_candidate is not None:
                masks['scale'] = all_masks[scale_candidate].astype(np.uint8) * 255
                print(f"✓ Scale: {mask_areas[scale_candidate]:.0f} pixels")
            else:
                print("⚠ Scale not automatically detected")

            # Additional objects
            other_objects = [idx for idx in valid_indices if idx not in [concrete_candidate, scale_candidate]]
            for i, idx in enumerate(other_objects[:3]):  # Limit to 3 additional objects
                masks[f'object_{i+1}'] = all_masks[idx].astype(np.uint8) * 255

        else:
            print("⚠ No masks found in segmentation results")

        return masks

    def detect_scale_boundaries(self, scale_mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Detect precise boundaries of the scale/ruler with orientation detection"""
        # Clean mask first
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_mask = cv2.morphologyEx(scale_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

        edges = cv2.Canny(cleaned_mask, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return edges, {}

        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get rotated rectangle for better orientation
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate orientation angle
        angle = rect[2]
        if w < h:
            angle += 90

        boundary_info = {
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'contour': largest_contour,
            'rotated_rect': rect,
            'bounding_box': box,
            'angle': angle,
            'area': cv2.contourArea(largest_contour)
        }

        print(f"✓ Scale boundaries: {w}x{h} pixels, angle: {angle:.1f}°")

        return edges, boundary_info

    def detect_concrete_boundaries(self, concrete_mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Detect precise boundaries of concrete block with subpixel accuracy"""
        # Enhance mask for better boundary detection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        enhanced_mask = cv2.morphologyEx(concrete_mask, cv2.MORPH_CLOSE, kernel)
        
        edges = cv2.Canny(enhanced_mask, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return edges, {}

        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate contour to reduce noise
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        x, y, w, h = cv2.boundingRect(approx_contour)

        boundary_info = {
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'contour': approx_contour,
            'area': cv2.contourArea(approx_contour),
            'perimeter': cv2.arcLength(approx_contour, True),
            'solidity': cv2.contourArea(approx_contour) / (w * h) if w * h > 0 else 0
        }

        print(f"✓ Concrete boundaries: {w}x{h} pixels, solidity: {boundary_info['solidity']:.2f}")

        return edges, boundary_info

    def segment_and_extract(self, image: np.ndarray) -> Dict:
        """Complete segmentation pipeline with GPU acceleration and optimization"""
        # Initialize model if needed
        if self.model is None:
            self.initialize_sam_model()

        start_time = time.time()
        
        # Perform GPU-accelerated segmentation
        results = self.segment_objects(image)

        total_time = time.time() - start_time
        print(f"✓ Total segmentation time: {total_time:.2f}s")

        # Extract masks
        masks = self.extract_masks(results[0] if isinstance(results, list) else results)

        # Detect boundaries
        segmentation_data = {'masks': masks}

        if 'concrete_block' in masks:
            edges, boundary_info = self.detect_concrete_boundaries(masks['concrete_block'])
            segmentation_data['concrete_boundaries'] = boundary_info
            segmentation_data['concrete_edges'] = edges

        if 'scale' in masks:
            edges, boundary_info = self.detect_scale_boundaries(masks['scale'])
            segmentation_data['scale_boundaries'] = boundary_info
            segmentation_data['scale_edges'] = edges

        self.segmentation_results = segmentation_data

        # Clear GPU cache if using CUDA
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return segmentation_data

    def visualize_segmentation(self, image: np.ndarray, 
                              masks: Dict[str, np.ndarray],
                              output_path: Optional[str] = None) -> np.ndarray:
        """Enhanced visualization with better color coding and labels"""
        vis_image = image.copy()

        colors = {
            'concrete_block': (0, 255, 0),    # Green
            'scale': (255, 0, 0),             # Blue
            'object_1': (0, 255, 255),        # Yellow
            'object_2': (255, 0, 255),        # Magenta
        }

        labels = {
            'concrete_block': 'Concrete Block',
            'scale': 'Scale/Ruler',
            'object_1': 'Object 1',
            'object_2': 'Object 2'
        }

        for name, mask in masks.items():
            if name in colors:
                color = colors[name]
                colored_mask = np.zeros_like(vis_image)
                colored_mask[mask > 0] = color
                vis_image = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)
                
                # Add label at centroid
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    M = cv2.moments(contours[0])
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        label = labels.get(name, name)
                        cv2.putText(vis_image, label, (cx-50, cy), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"✓ Enhanced visualization saved to {output_path}")

        return vis_image

    def get_device_info(self) -> Dict:
        """Get detailed device information with memory usage"""
        info = self.device_info.copy()
        
        if self.device == 'cuda':
            info['current_memory_gb'] = torch.cuda.memory_allocated(0) / 1e9
            info['max_memory_gb'] = torch.cuda.max_memory_allocated(0) / 1e9
            info['memory_cached_gb'] = torch.cuda.memory_reserved(0) / 1e9
            
        return info


if __name__ == "__main__":
    # Example usage with GPU acceleration
    print("="*60)
    print("SAM 2 Segmentation Module - GPU Accelerated")
    print("="*60)

    segmenter = SegmentationModule()

    # Print device info
    device_info = segmenter.get_device_info()
    print(f"\nDevice: {device_info['device']}")
    print(f"Name: {device_info['device_name']}")
    if device_info['memory_gb'] > 0:
        print(f"Memory: {device_info['memory_gb']:.2f} GB")

    # Load and segment image
    image = cv2.imread("preprocessed_output.jpg")
    if image is not None:
        results = segmenter.segment_and_extract(image)

        if results.get('masks'):
            segmenter.visualize_segmentation(
                image,
                results['masks'],
                output_path="segmentation_output.jpg"
            )

        print("\n✓ Segmentation complete!")