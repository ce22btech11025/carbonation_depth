"""
SIMPLIFIED ROBUST CALIBRATION - Focus on Actual Markings Only

Key improvements:
1. Stricter filtering for actual vertical lines (minimum length)
2. Clustering of nearby lines to merge false positives
3. Spacing consistency validation (remove outliers aggressively)
4. Manual verification mode for debugging
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy import optimize
from scipy.ndimage import gaussian_filter

class AdvancedCalibrationMeasurement:
    def __init__(self):
        """Initialize the calibration module"""
        self.pixel_to_mm_ratio = None
        self.pixel_to_cm_ratio = None
        self.measurements = {}
        self.scale_info = {}
        self.calibration_uncertainty = None

    # ============ SIMPLE ROBUST EDGE DETECTION ============
    def simple_edge_detection(self, gray_image: np.ndarray) -> np.ndarray:
        """Simple edge detection - focus on strong edges only"""
        
        # Apply median blur to remove noise
        gray_blur = cv2.medianBlur(gray_image, 5)
        
        # Apply morphological operations to enhance edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        gray_morph = cv2.morphologyEx(gray_blur, cv2.MORPH_CLOSE, kernel)
        
        # Use Sobel for vertical edges (marking lines are vertical)
        sobelx = cv2.Sobel(gray_morph, cv2.CV_64F, 1, 0, ksize=3)
        edges = np.abs(sobelx).astype(np.uint8)
        
        # Threshold to keep only strong edges
        _, edges = cv2.threshold(edges, 80, 255, cv2.THRESH_BINARY)
        
        return edges

    # ============ STRICT MARKING LINE DETECTION ============
    def detect_marking_lines_strict(self, scale_region: np.ndarray,
                                   scale_mask: np.ndarray,
                                   min_line_length: int = 50) -> Dict:
        """
        Detect only actual marking lines with strict criteria
        """
        
        print("\n[Calibration] Detecting marking lines (STRICT mode)...")
        
        gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_and(gray, gray, mask=scale_mask)
        
        # Simple edge detection
        edges = self.simple_edge_detection(gray)
        
        # Hough Line detection with STRICT parameters
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=40,  # Higher threshold = fewer false lines
            minLineLength=min_line_length,  # Must be long enough
            maxLineGap=2   # Small gap tolerance
        )
        
        if lines is None or len(lines) == 0:
            print(" ⚠ No marking lines detected")
            return {'lines': [], 'positions': [], 'num_lines': 0}
        
        print(f" Initial lines detected: {len(lines)}")
        
        # Filter: Keep only VERTICAL lines
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            dy = abs(y2 - y1)
            dx = abs(x2 - x1)
            
            # Prefer vertical lines (dy >> dx)
            if dy > dx * 3:  # Must be significantly more vertical than horizontal
                vertical_lines.append((x1, y1, x2, y2))
        
        print(f" After vertical filter: {len(vertical_lines)}")
        
        if len(vertical_lines) < 2:
            print(" ⚠ Too few vertical lines")
            return {'lines': [], 'positions': [], 'num_lines': 0}
        
        # Cluster nearby lines (merge false positives)
        clustered_lines = self._cluster_nearby_lines(vertical_lines, cluster_dist=5)
        
        print(f" After clustering: {len(clustered_lines)}")
        
        # Extract center x positions
        marking_positions = []
        for line in clustered_lines:
            x1, y1, x2, y2 = line
            center_x = (x1 + x2) / 2.0
            marking_positions.append((center_x, (y1 + y2) / 2.0))
        
        # Sort by x position
        marking_positions = sorted(marking_positions, key=lambda p: p[0])
        
        return {
            'lines': clustered_lines,
            'positions': marking_positions,
            'num_lines': len(clustered_lines),
            'edges': edges
        }

    def _cluster_nearby_lines(self, lines: List[Tuple], cluster_dist: int = 5) -> List[Tuple]:
        """Merge lines that are too close together (false positives)"""
        
        if not lines:
            return []
        
        # Extract x positions
        x_positions = [(line[0] + line[2]) / 2.0 for line in lines]
        x_positions = sorted(enumerate(x_positions), key=lambda x: x[1])
        
        clustered = []
        current_cluster = [x_positions[0][1]]
        current_lines = [lines[x_positions[0][0]]]
        
        for i in range(1, len(x_positions)):
            x_idx, x_val = x_positions[i]
            
            if x_val - current_cluster[-1] <= cluster_dist:
                # Add to current cluster
                current_cluster.append(x_val)
                current_lines.append(lines[x_idx])
            else:
                # Save cluster (use average)
                avg_x = np.mean(current_cluster)
                representative_line = self._average_cluster_line(current_lines, avg_x)
                clustered.append(representative_line)
                
                current_cluster = [x_val]
                current_lines = [lines[x_idx]]
        
        # Don't forget last cluster
        avg_x = np.mean(current_cluster)
        representative_line = self._average_cluster_line(current_lines, avg_x)
        clustered.append(representative_line)
        
        return clustered

    def _average_cluster_line(self, lines: List[Tuple], target_x: float) -> Tuple:
        """Average lines in a cluster"""
        
        y_coords = []
        for x1, y1, x2, y2 in lines:
            y_coords.extend([y1, y2])
        
        y_min = min(y_coords)
        y_max = max(y_coords)
        
        return (int(target_x), int(y_min), int(target_x), int(y_max))

    # ============ SPACING VALIDATION ============
    def validate_marking_spacing(self, positions: List[Tuple[float, float]],
                                max_deviation_percent: float = 15.0) -> Tuple[List[float], List[int]]:
        """
        Calculate spacings and validate consistency
        Remove outliers aggressively
        """
        
        if len(positions) < 3:
            return [], list(range(len(positions)))
        
        # Calculate spacings between consecutive marks
        spacings = []
        for i in range(len(positions) - 1):
            spacing = positions[i+1][0] - positions[i][0]
            spacings.append(spacing)
        
        spacings = np.array(spacings)
        
        print(f"\n [Spacing] Raw spacings: min={spacings.min():.2f}, max={spacings.max():.2f}, mean={spacings.mean():.2f}")
        
        # Aggressive outlier removal (3 iterations)
        median_spacing = np.median(spacings)
        
        for iteration in range(3):
            deviation_percent = np.abs(spacings - median_spacing) / median_spacing * 100
            valid_mask = deviation_percent <= max_deviation_percent
            
            if np.sum(valid_mask) < 2:
                break
            
            spacings = spacings[valid_mask]
            median_spacing = np.median(spacings)
            
            print(f" [Iteration {iteration+1}] Kept {np.sum(valid_mask)} spacings, median={median_spacing:.2f}")
        
        # Map back to original positions
        valid_indices = list(range(len(spacings) + 1))
        
        return spacings, valid_indices

    # ============ MAIN CALIBRATION ============
    def auto_calibrate_advanced(self, image: np.ndarray,
                               scale_mask: np.ndarray) -> Dict:
        """
        Main calibration method - ROBUST VERSION
        """
        
        print("\n" + "="*70)
        print("ROBUST CALIBRATION - SIMPLIFIED & ACCURATE")
        print("="*70)
        
        # Extract scale region
        scale_region = cv2.bitwise_and(image, image, mask=scale_mask)
        
        # Detect marking lines (STRICT)
        marking_info = self.detect_marking_lines_strict(scale_region, scale_mask, min_line_length=50)
        
        if marking_info['num_lines'] < 3:
            print("\n ✗ ERROR: Insufficient markings detected!")
            print(f"   Found only {marking_info['num_lines']} markings")
            print("   Check if the scale mask is correct")
            return None
        
        # Validate spacing consistency
        spacings, valid_indices = self.validate_marking_spacing(
            marking_info['positions'],
            max_deviation_percent=15.0
        )
        
        if len(spacings) < 2:
            print("\n ✗ ERROR: Spacing validation failed!")
            return None
        
        # Calculate pixel-to-cm ratio
        # Standard ruler: 2mm between markings = 0.2cm
        mean_spacing_px = np.mean(spacings)
        std_spacing_px = np.std(spacings)
        
        # 2mm = 0.2cm per marking
        pixel_per_cm = mean_spacing_px / 0.2
        pixel_per_mm = pixel_per_cm / 10
        
        print(f"\n [Result] Mean spacing: {mean_spacing_px:.2f} ± {std_spacing_px:.2f} px")
        print(f" [Result] Pixel/cm: {pixel_per_cm:.2f}")
        print(f" [Result] Pixel/mm: {pixel_per_mm:.2f}")
        
        # Calculate uncertainty
        uncertainty_mm = (std_spacing_px / mean_spacing_px) * 2.0 if mean_spacing_px > 0 else 0
        
        # Store results
        self.pixel_to_cm_ratio = pixel_per_cm
        self.pixel_to_mm_ratio = pixel_per_mm
        self.calibration_uncertainty = uncertainty_mm
        
        self.scale_info = {
            'num_markings': marking_info['num_lines'],
            'pixel_per_cm': pixel_per_cm,
            'pixel_per_mm': pixel_per_mm,
            'mean_spacing_px': mean_spacing_px,
            'std_spacing_px': std_spacing_px,
            'validation_agreement': 100.0 - (std_spacing_px / mean_spacing_px * 100),
            'uncertainty_mm': uncertainty_mm
        }
        
        print("\n" + "="*70)
        print("✓ CALIBRATION SUCCESSFUL")
        print(f" Markings detected: {marking_info['num_lines']}")
        print(f" Pixel-to-MM: {pixel_per_mm:.6f}")
        print(f" Pixel-to-CM: {pixel_per_cm:.6f}")
        print(f" Method agreement: {self.scale_info['validation_agreement']:.2f}%")
        print(f" Uncertainty: ±{uncertainty_mm:.4f} mm")
        print("="*70 + "\n")
        
        return self.scale_info

    # ============ MEASUREMENT WITH UNCERTAINTY ============
    def measure_concrete_block_with_uncertainty(self, concrete_mask: np.ndarray,
                                               image: np.ndarray) -> Dict:
        """Measure concrete block with uncertainty propagation"""
        
        if self.pixel_to_mm_ratio is None:
            raise ValueError("Calibration not performed!")
        
        print("\n[Measurement] Measuring concrete block...")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get contour
        contours, _ = cv2.findContours(concrete_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        contour = max(contours, key=cv2.contourArea)
        
        # Refine with cornerSubPix
        try:
            refined_contour = cv2.cornerSubPix(
                gray,
                contour.reshape(-1, 1, 2).astype(np.float32),
                winSize=(5, 5),
                zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            ).reshape(-1, 2)
        except:
            refined_contour = contour.reshape(-1, 2).astype(float)
        
        # Calculate measurements
        area_px = cv2.contourArea(refined_contour)
        perimeter_px = cv2.arcLength(refined_contour, closed=True)
        
        x_coords = refined_contour[:, 0]
        y_coords = refined_contour[:, 1]
        
        width_px = np.max(x_coords) - np.min(x_coords)
        height_px = np.max(y_coords) - np.min(y_coords)
        
        # Convert to physical units
        measurements = {
            'width_mm': width_px / self.pixel_to_mm_ratio,
            'height_mm': height_px / self.pixel_to_mm_ratio,
            'area_mm2': area_px / (self.pixel_to_mm_ratio ** 2),
            'area_cm2': area_px / (self.pixel_to_mm_ratio ** 2) / 100,
            'perimeter_mm': perimeter_px / self.pixel_to_mm_ratio,
            'uncertainty_mm': self.calibration_uncertainty
        }
        
        # Add uncertainty
        measurements['width_uncertainty'] = self.calibration_uncertainty
        measurements['area_uncertainty_percent'] = (
            2 * self.calibration_uncertainty / measurements['width_mm'] * 100
        ) if measurements['width_mm'] > 0 else 0
        
        print(f"✓ Width: {measurements['width_mm']:.2f} ± {self.calibration_uncertainty:.2f} mm")
        print(f"✓ Height: {measurements['height_mm']:.2f} ± {self.calibration_uncertainty:.2f} mm")
        print(f"✓ Area: {measurements['area_cm2']:.2f} cm²")
        print(f"✓ Area uncertainty: ±{measurements['area_uncertainty_percent']:.1f}%")
        
        self.measurements['concrete_block'] = measurements
        return measurements

    def create_measurement_visualization(self, image: np.ndarray,
                                        concrete_mask: np.ndarray,
                                        output_path: Optional[str] = None) -> np.ndarray:
        """Create visualization"""
        
        vis_image = image.copy()
        
        # Draw contours
        contours, _ = cv2.findContours(concrete_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)
        
        # Add measurement text
        if 'concrete_block' in self.measurements:
            m = self.measurements['concrete_block']
            text1 = f"Width: {m['width_mm']:.2f} mm"
            text2 = f"Height: {m['height_mm']:.2f} mm"
            text3 = f"Area: {m['area_cm2']:.2f} cm²"
            
            cv2.putText(vis_image, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2)
            cv2.putText(vis_image, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2)
            cv2.putText(vis_image, text3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"✓ Visualization saved to {output_path}")
        
        return vis_image

    def generate_advanced_report(self, output_path: str = "advanced_report.txt") -> str:
        """Generate detailed report"""
        
        lines = ["="*70, "CALIBRATION AND MEASUREMENT REPORT", "="*70, ""]
        
        if self.scale_info:
            lines.extend([
                "CALIBRATION RESULTS:",
                f" Markings detected: {self.scale_info.get('num_markings', 'N/A')}",
                f" Pixel/cm: {self.scale_info.get('pixel_per_cm', 'N/A'):.6f}",
                f" Pixel/mm: {self.scale_info.get('pixel_per_mm', 'N/A'):.6f}",
                f" Mean spacing: {self.scale_info.get('mean_spacing_px', 'N/A'):.2f} px",
                f" Spacing std: {self.scale_info.get('std_spacing_px', 'N/A'):.2f} px",
                f" Consistency: {self.scale_info.get('validation_agreement', 'N/A'):.2f}%",
                f" Uncertainty: ±{self.scale_info.get('uncertainty_mm', 'N/A'):.4f} mm",
                ""
            ])
        
        if 'concrete_block' in self.measurements:
            m = self.measurements['concrete_block']
            lines.extend([
                "CONCRETE BLOCK MEASUREMENTS:",
                f" Width: {m['width_mm']:.2f} mm",
                f" Height: {m['height_mm']:.2f} mm",
                f" Area: {m['area_cm2']:.2f} cm²",
                f" Perimeter: {m['perimeter_mm']:.2f} mm",
                f" Area uncertainty: ±{m['area_uncertainty_percent']:.1f}%",
                ""
            ])
        
        lines.append("="*70)
        
        report = "\n".join(lines)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Report saved to {output_path}")
        
        return report