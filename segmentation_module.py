"""Segmentation Module using SAM 2
Performs precise object segmentation for concrete block, scale, and other objects
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
try:
    from ultralytics import SAM
except ImportError:
    print("WARNING: ultralytics not installed. Install with: pip install ultralytics")
    SAM = None

class SegmentationModule:
    def __init__(self, model_path: str = "sam2_b.pt"):
        """Initialize the SAM 2 segmentation model
        
        Args:
            model_path: Path to SAM 2 model weights
        """
        self.model_path = model_path
        self.model = None
        self.segmentation_results = {}
        
    def initialize_sam_model(self) -> None:
        """Load and initialize the SAM 2 model"""
        if SAM is None:
            raise ImportError("ultralytics package not installed. Install with: pip install ultralytics")
        
        try:
            self.model = SAM(self.model_path, device="cpu")
            print(f"✓ SAM 2 model loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Model will be downloaded automatically on first use.")
            self.model = SAM("sam2_b.pt")
    
    def segment_objects(self, image: np.ndarray, 
                       points: Optional[List[List[int]]] = None,
                       labels: Optional[List[int]] = None) -> Dict:
        """Segment objects using SAM 2
        
        Args:
            image: Input image (preprocessed)
            points: Optional list of prompt points [[x, y], ...]
            labels: Optional list of point labels (1 for foreground, 0 for background)
            
        Returns:
            Dictionary containing segmentation results
        """
        if self.model is None:
            self.initialize_sam_model()
        print("6")
        
        # Run automatic segmentation if no points provided
        # if points is None:
        #     print('8')
        #     results = self.model(image, save=False, verbose=False)
        #     print("SAM Results object:", results)
        #     print("Type of results:", type(results))
        #     print("Results content:", dir(results))

        #     print(f"✓ Automatic segmentation complete: {len(results)} objects detected")
        if points is None:
            print('8')
            try:
                results = self.model.predict(source=image, device="cpu", save=False, verbose=False)
            except Exception as e:
                print("Error running SAM:", e)
                results = []
            print("SAM Results object:", results)
            print("Type of results:", type(results))
            print("Results content:", dir(results))
            print(f"✓ Automatic segmentation complete: {len(results)} objects detected")
            
        else:
            print("9")
            # Use point prompts for guided segmentation
            results = self.model(image, points=points, labels=labels, save=False, verbose=False)
            print(f"✓ Guided segmentation complete with {len(points)} prompts")
        print("7")
        return results
    
    def extract_masks(self, results) -> Dict[str, np.ndarray]:
        """Extract individual masks for different objects
        
        Args:
            results: Segmentation results from SAM 2
            
        Returns:
            Dictionary of masks for different objects
        """
        masks = {}
        
        if hasattr(results, 'masks') and results.masks is not None:
            all_masks = results.masks.data.cpu().numpy()
            
            # Sort masks by area (largest first)
            mask_areas = [np.sum(mask) for mask in all_masks]
            sorted_indices = np.argsort(mask_areas)[::-1]
            
            # Assume largest mask is concrete block, second largest is scale
            if len(sorted_indices) >= 1:
                masks['concrete_block'] = all_masks[sorted_indices[0]].astype(np.uint8) * 255
                print(f"✓ Concrete block mask extracted: {mask_areas[sorted_indices[0]]:.0f} pixels")
            
            if len(sorted_indices) >= 2:
                masks['scale'] = all_masks[sorted_indices[1]].astype(np.uint8) * 255
                print(f"✓ Scale mask extracted: {mask_areas[sorted_indices[1]]:.0f} pixels")
            
            # Additional masks for other objects
            for i in range(2, min(len(sorted_indices), 5)):
                masks[f'object_{i-1}'] = all_masks[sorted_indices[i]].astype(np.uint8) * 255
        
        return masks
    
    def detect_scale_boundaries(self, scale_mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Detect precise boundaries of the scale/ruler
        
        Args:
            scale_mask: Binary mask of the scale
            
        Returns:
            Tuple of (edge image, boundary info dict)
        """
        # Apply Canny edge detection
        edges = cv2.Canny(scale_mask, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return edges, {}
        
        # Get the largest contour (should be the scale)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate scale dimensions
        boundary_info = {
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'contour': largest_contour,
            'area': cv2.contourArea(largest_contour)
        }
        
        print(f"✓ Scale boundaries detected: {w}x{h} pixels")
        return edges, boundary_info
    
    def detect_concrete_boundaries(self, concrete_mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Detect precise boundaries of the concrete block
        
        Args:
            concrete_mask: Binary mask of the concrete block
            
        Returns:
            Tuple of (edge image, boundary info dict)
        """
        # Apply Canny edge detection
        edges = cv2.Canny(concrete_mask, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return edges, {}
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        boundary_info = {
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'contour': largest_contour,
            'area': cv2.contourArea(largest_contour),
            'perimeter': cv2.arcLength(largest_contour, True)
        }
        
        print(f"✓ Concrete block boundaries detected: {w}x{h} pixels")
        return edges, boundary_info
    
    def segment_and_extract(self, image: np.ndarray) -> Dict:
        """Complete segmentation pipeline
        
        Args:
            image: Preprocessed input image
            
        Returns:
            Dictionary containing all segmentation results and masks
        """
        # Initialize model if needed
        if self.model is None:
            self.initialize_sam_model()
        
        # Perform segmentation
        print("5")
        results = self.segment_objects(image)
        print(results)
        # Extract masks
        masks = self.extract_masks(results[0] if isinstance(results, list) else results)
        print(masks)
        # Detect boundaries
        segmentation_data = {'masks': masks}
        
        if 'concrete_block' in masks:
            edges, boundary_info = self.detect_concrete_boundaries(masks['concrete_block'])
            segmentation_data['concrete_boundaries'] = boundary_info
            segmentation_data['concrete_edges'] = edges
        print("5")
        if 'scale' in masks:
            edges, boundary_info = self.detect_scale_boundaries(masks['scale'])
            segmentation_data['scale_boundaries'] = boundary_info
            segmentation_data['scale_edges'] = edges
        print("6")
        self.segmentation_results = segmentation_data
        return segmentation_data
    
    def visualize_segmentation(self, image: np.ndarray, masks: Dict[str, np.ndarray], 
                              output_path: Optional[str] = None) -> np.ndarray:
        """Visualize segmentation results with color overlays
        
        Args:
            image: Original image
            masks: Dictionary of masks
            output_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        
        # Color map for different objects
        colors = {
            'concrete_block': (0, 255, 0),    # Green
            'scale': (255, 0, 0),              # Blue
            'object_1': (0, 255, 255),         # Yellow
            'object_2': (255, 0, 255),         # Magenta
        }
        
        # Overlay masks with transparency
        for name, mask in masks.items():
            if name in colors:
                color = colors[name]
                colored_mask = np.zeros_like(vis_image)
                colored_mask[mask > 0] = color
                vis_image = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)
        
        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"✓ Segmentation visualization saved to {output_path}")
        
        return vis_image


if __name__ == "__main__":
    # Example usage
    segmenter = SegmentationModule()
    
    # Load preprocessed image
    image = cv2.imread("preprocessed_output.jpg")
    
    if image is not None:
        # Perform segmentation
        results = segmenter.segment_and_extract(image)
        
        # Visualize
        if results.get('masks'):
            segmenter.visualize_segmentation(
                image, 
                results['masks'], 
                output_path="segmentation_output.jpg"
            )
        
        print("\n✓ Segmentation complete!")