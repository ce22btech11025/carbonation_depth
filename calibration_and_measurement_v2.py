"""Calibration and Measurement Module - AUTOMATIC SCALE DETECTION

Automatically detects yellow scale with black markings and calculates pixel-to-mm ratio
No manual input required!
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List

class CalibrationMeasurement:
    
    def __init__(self):
        """Initialize the calibration and measurement module"""
        self.pixel_to_mm_ratio = None
        self.pixel_to_cm_ratio = None
        self.measurements = {}
        self.scale_info = {}
    
    def detect_yellow_scale(self, image: np.ndarray, scale_mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Detect yellow scale region with black markings
        
        Args:
            image: Original image
            scale_mask: Binary mask of the scale region
            
        Returns:
            Tuple of (scale_region_image, detection_info)
        """
        print("\n[Auto-Calibration] Detecting yellow scale...")
        
        # Extract scale region using mask
        scale_region = cv2.bitwise_and(image, image, mask=scale_mask)
        
        # Convert to HSV for yellow color detection
        hsv = cv2.cvtColor(scale_region, cv2.COLOR_BGR2HSV)
        
        # Define yellow color range
        # Yellow in HSV: H=20-30, S=100-255, V=100-255
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Apply original scale mask
        yellow_mask = cv2.bitwise_and(yellow_mask, yellow_mask, mask=scale_mask)
        
        yellow_pixels = np.sum(yellow_mask > 0)
        total_scale_pixels = np.sum(scale_mask > 0)
        yellow_percentage = (yellow_pixels / total_scale_pixels * 100) if total_scale_pixels > 0 else 0
        
        detection_info = {
            'yellow_mask': yellow_mask,
            'yellow_percentage': yellow_percentage,
            'is_yellow_scale': yellow_percentage > 30  # At least 30% yellow
        }
        
        print(f"  ✓ Yellow scale detection: {yellow_percentage:.1f}% yellow pixels")
        print(f"  ✓ Scale type: {'YELLOW' if detection_info['is_yellow_scale'] else 'OTHER'}")
        
        return scale_region, detection_info
    
    def detect_black_markings(self, scale_region: np.ndarray, scale_mask: np.ndarray) -> Dict:
        """Detect black marking lines on the scale (cm graduations)
        
        Args:
            scale_region: Cropped scale region
            scale_mask: Binary mask of scale
            
        Returns:
            Dictionary with detected marking lines and positions
        """
        print("\n[Auto-Calibration] Detecting black markings (cm graduations)...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        
        # Apply mask
        gray = cv2.bitwise_and(gray, gray, mask=scale_mask)
        
        # Threshold to get black markings (dark lines)
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        
        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Apply edge detection
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=30,
            minLineLength=15,  # Short lines for markings
            maxLineGap=3
        )
        
        if lines is None or len(lines) == 0:
            print("  ⚠ Warning: No marking lines detected!")
            return {'lines': [], 'positions': [], 'num_lines': 0}
        
        # Filter lines: keep perpendicular ones (marking lines are perpendicular to scale)
        marking_lines = []
        marking_positions = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line angle
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Keep mostly vertical lines (70-110 degrees) for horizontal scale
            # Or mostly horizontal lines (0-20 or 160-180 degrees) for vertical scale
            is_perpendicular = (70 <= angle <= 110) or (angle <= 20) or (angle >= 160)
            
            if is_perpendicular:
                marking_lines.append((x1, y1, x2, y2))
                # Store midpoint position
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                marking_positions.append((mid_x, mid_y))
        
        # Sort positions by x-coordinate (for horizontal scale)
        if marking_positions:
            marking_positions = sorted(marking_positions, key=lambda p: p[0])
        
        marking_info = {
            'lines': marking_lines,
            'positions': marking_positions,
            'num_lines': len(marking_lines),
            'edges': edges,
            'binary': binary
        }
        
        print(f"  ✓ Detected {len(marking_lines)} black marking lines")
        
        return marking_info
    
    def calculate_cm_spacings(self, marking_positions: List[Tuple[float, float]], 
                             num_samples: int = 4) -> Dict:
        """Calculate pixel spacing for 1cm intervals using multiple samples
        
        Args:
            marking_positions: List of (x, y) positions of markings
            num_samples: Number of 1cm intervals to sample
            
        Returns:
            Dictionary with spacing measurements and average
        """
        print(f"\n[Auto-Calibration] Calculating pixel-to-cm ratio from {num_samples} samples...")
        
        if len(marking_positions) < 2:
            print("  ⚠ Warning: Not enough markings detected!")
            return {'spacings': [], 'average_px_per_cm': 0, 'std_dev': 0}
        
        # Calculate distances between consecutive markings (assuming 1cm intervals)
        spacings = []
        spacing_pairs = []
        
        # Sample multiple consecutive pairs
        max_samples = min(num_samples, len(marking_positions) - 1)
        
        for i in range(max_samples):
            if i + 1 < len(marking_positions):
                pos1 = marking_positions[i]
                pos2 = marking_positions[i + 1]
                
                # Calculate Euclidean distance
                distance_px = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                
                spacings.append(distance_px)
                spacing_pairs.append((i, i+1, distance_px))
                
                print(f"  Sample {i+1}: Mark {i} to Mark {i+1} = {distance_px:.2f} pixels (1 cm)")
        
        if not spacings:
            print("  ⚠ Warning: Could not calculate spacings!")
            return {'spacings': [], 'average_px_per_cm': 0, 'std_dev': 0}
        
        # Calculate statistics
        average_px_per_cm = np.mean(spacings)
        std_dev = np.std(spacings)
        median_px_per_cm = np.median(spacings)
        
        # Filter outliers (remove measurements > 1.5 * std dev from mean)
        filtered_spacings = [s for s in spacings if abs(s - average_px_per_cm) <= 1.5 * std_dev]
        
        if filtered_spacings:
            final_px_per_cm = np.mean(filtered_spacings)
        else:
            final_px_per_cm = average_px_per_cm
        
        spacing_info = {
            'spacings': spacings,
            'spacing_pairs': spacing_pairs,
            'average_px_per_cm': average_px_per_cm,
            'median_px_per_cm': median_px_per_cm,
            'std_dev': std_dev,
            'filtered_px_per_cm': final_px_per_cm,
            'num_samples': len(spacings)
        }
        
        print(f"\n  ✓ Pixel spacing analysis:")
        print(f"    Average: {average_px_per_cm:.2f} pixels/cm")
        print(f"    Median:  {median_px_per_cm:.2f} pixels/cm")
        print(f"    Std Dev: {std_dev:.2f} pixels")
        print(f"    Final (filtered): {final_px_per_cm:.2f} pixels/cm")
        
        return spacing_info
    
    def auto_calibrate_from_scale(self, image: np.ndarray, scale_mask: np.ndarray) -> Dict:
        """MAIN METHOD: Automatically calibrate from yellow scale with black markings
        
        Args:
            image: Original preprocessed image
            scale_mask: Binary mask of the detected scale
            
        Returns:
            Dictionary with calibration results
        """
        print("\n" + "="*60)
        print("AUTOMATIC SCALE CALIBRATION")
        print("="*60)
        
        # Step 1: Detect yellow scale
        scale_region, yellow_info = self.detect_yellow_scale(image, scale_mask)
        
        if not yellow_info['is_yellow_scale']:
            print("\n  ⚠ Warning: Scale does not appear to be yellow!")
            print("  Continuing with detection anyway...")
        
        # Step 2: Detect black marking lines
        marking_info = self.detect_black_markings(scale_region, scale_mask)
        
        if marking_info['num_lines'] < 2:
            print("\n  ✗ ERROR: Not enough marking lines detected for calibration!")
            print("  Please ensure scale markings are visible and clear.")
            return None
        
        # Step 3: Calculate pixel-to-cm ratio from multiple samples
        spacing_info = self.calculate_cm_spacings(
            marking_info['positions'], 
            num_samples=min(4, marking_info['num_lines'] - 1)
        )
        
        if spacing_info['filtered_px_per_cm'] == 0:
            print("\n  ✗ ERROR: Could not calculate pixel-to-cm ratio!")
            return None
        
        # Store calibration results
        self.pixel_to_cm_ratio = spacing_info['filtered_px_per_cm']
        self.pixel_to_mm_ratio = self.pixel_to_cm_ratio / 10  # Convert cm to mm
        
        self.scale_info = {
            'scale_type': 'Yellow ruler with black cm markings',
            'detection_method': 'Automatic',
            'yellow_percentage': yellow_info['yellow_percentage'],
            'num_markings_detected': marking_info['num_lines'],
            'samples_used': spacing_info['num_samples'],
            'pixel_per_cm': self.pixel_to_cm_ratio,
            'pixel_per_mm': self.pixel_to_mm_ratio,
            'std_deviation': spacing_info['std_dev'],
            'all_spacings': spacing_info['spacings']
        }
        
        print("\n" + "="*60)
        print("CALIBRATION COMPLETE")
        print("="*60)
        print(f"✓ Scale Type: Yellow ruler with black markings")
        print(f"✓ Method: Automatic detection")
        print(f"✓ Markings detected: {marking_info['num_lines']}")
        print(f"✓ Samples analyzed: {spacing_info['num_samples']}")
        print(f"✓ Pixel-to-cm ratio: {self.pixel_to_cm_ratio:.4f} pixels/cm")
        print(f"✓ Pixel-to-mm ratio: {self.pixel_to_mm_ratio:.4f} pixels/mm")
        print(f"✓ Measurement accuracy: ±{spacing_info['std_dev']:.2f} pixels")
        print("="*60 + "\n")
        
        return self.scale_info
    
    def measure_concrete_block(self, concrete_boundaries: Dict, 
                               concrete_mask: np.ndarray) -> Dict:
        """Measure concrete block dimensions and area
        
        Args:
            concrete_boundaries: Dictionary with concrete boundary information
            concrete_mask: Binary mask of concrete block
            
        Returns:
            Dictionary with measurements in pixels and mm/cm
        """
        if self.pixel_to_mm_ratio is None:
            raise ValueError("Calibration not performed. Call auto_calibrate_from_scale first.")
        
        print("\n[Measurement] Measuring concrete block...")
        
        # Pixel measurements
        width_px = concrete_boundaries['width']
        height_px = concrete_boundaries['height']
        area_px = concrete_boundaries['area']
        perimeter_px = concrete_boundaries.get('perimeter', 0)
        
        # Convert to millimeters
        width_mm = width_px / self.pixel_to_mm_ratio
        height_mm = height_px / self.pixel_to_mm_ratio
        area_mm2 = area_px / (self.pixel_to_mm_ratio ** 2)
        perimeter_mm = perimeter_px / self.pixel_to_mm_ratio
        
        measurements = {
            'width_pixels': width_px,
            'height_pixels': height_px,
            'area_pixels': area_px,
            'perimeter_pixels': perimeter_px,
            'width_mm': width_mm,
            'height_mm': height_mm,
            'width_cm': width_mm / 10,
            'height_cm': height_mm / 10,
            'area_mm2': area_mm2,
            'area_cm2': area_mm2 / 100,
            'perimeter_mm': perimeter_mm,
            'perimeter_cm': perimeter_mm / 10,
            'calibration_used': f'{self.pixel_to_cm_ratio:.2f} px/cm'
        }
        
        print(f"✓ Concrete block measured:")
        print(f"  Dimensions: {width_mm:.2f} x {height_mm:.2f} mm")
        print(f"  Dimensions: {width_mm/10:.2f} x {height_mm/10:.2f} cm")
        print(f"  Area: {area_mm2:.2f} mm² ({area_mm2/100:.2f} cm²)")
        print(f"  Calibration: {self.pixel_to_cm_ratio:.2f} pixels/cm")
        
        self.measurements['concrete_block'] = measurements
        return measurements
    
    def get_affected_area(self, image: np.ndarray, 
                         concrete_mask: np.ndarray) -> Dict:
        """Calculate phenophthalein affected area (magenta regions)
        
        Args:
            image: Original image
            concrete_mask: Binary mask of concrete block
            
        Returns:
            Dictionary with affected and unaffected areas
        """
        if self.pixel_to_mm_ratio is None:
            raise ValueError("Calibration not performed. Call auto_calibrate_from_scale first.")
        
        print("\n[Measurement] Analyzing phenophthalein coverage...")
        
        # Extract concrete block region
        concrete_region = cv2.bitwise_and(image, image, mask=concrete_mask)
        
        # Convert to HSV for magenta detection
        hsv = cv2.cvtColor(concrete_region, cv2.COLOR_BGR2HSV)
        
        # Define magenta color range (phenophthalein indicator)
        # Magenta in HSV: H=140-170, S=50-255, V=50-255
        lower_magenta = np.array([140, 50, 50])
        upper_magenta = np.array([170, 255, 255])
        
        # Create mask for magenta regions
        magenta_mask = cv2.inRange(hsv, lower_magenta, upper_magenta)
        
        # Apply concrete mask to ensure we only count within concrete
        magenta_mask = cv2.bitwise_and(magenta_mask, magenta_mask, mask=concrete_mask)
        
        # Count pixels
        affected_pixels = np.sum(magenta_mask > 0)
        total_concrete_pixels = np.sum(concrete_mask > 0)
        unaffected_pixels = total_concrete_pixels - affected_pixels
        
        # Calculate areas in physical units
        affected_area_mm2 = affected_pixels / (self.pixel_to_mm_ratio ** 2)
        unaffected_area_mm2 = unaffected_pixels / (self.pixel_to_mm_ratio ** 2)
        total_area_mm2 = total_concrete_pixels / (self.pixel_to_mm_ratio ** 2)
        
        # Calculate percentages
        affected_percentage = (affected_pixels / total_concrete_pixels * 100) if total_concrete_pixels > 0 else 0
        unaffected_percentage = 100 - affected_percentage
        
        area_analysis = {
            'affected_pixels': affected_pixels,
            'unaffected_pixels': unaffected_pixels,
            'total_pixels': total_concrete_pixels,
            'affected_mm2': affected_area_mm2,
            'unaffected_mm2': unaffected_area_mm2,
            'total_mm2': total_area_mm2,
            'affected_cm2': affected_area_mm2 / 100,
            'unaffected_cm2': unaffected_area_mm2 / 100,
            'total_cm2': total_area_mm2 / 100,
            'affected_percentage': affected_percentage,
            'unaffected_percentage': unaffected_percentage,
            'magenta_mask': magenta_mask
        }
        
        print(f"✓ Phenophthalein analysis complete:")
        print(f"  Affected area (magenta): {affected_area_mm2:.2f} mm² ({affected_area_mm2/100:.2f} cm²)")
        print(f"  Unaffected area: {unaffected_area_mm2:.2f} mm² ({unaffected_area_mm2/100:.2f} cm²)")
        print(f"  Coverage: {affected_percentage:.2f}% affected, {unaffected_percentage:.2f}% unaffected")
        
        self.measurements['phenophthalein_analysis'] = area_analysis
        return area_analysis
    
    def create_measurement_visualization(self, image: np.ndarray,
                                        concrete_mask: np.ndarray,
                                        magenta_mask: np.ndarray,
                                        output_path: Optional[str] = None) -> np.ndarray:
        """Create visualization showing measurements and affected areas
        
        Args:
            image: Original image
            concrete_mask: Binary mask of concrete block
            magenta_mask: Binary mask of magenta regions
            output_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        
        # Create colored overlay for affected areas
        magenta_overlay = np.zeros_like(vis_image)
        magenta_overlay[magenta_mask > 0] = [255, 0, 255]  # Magenta
        
        # Blend with original image
        vis_image = cv2.addWeighted(vis_image, 0.7, magenta_overlay, 0.3, 0)
        
        # Draw concrete block contour
        contours, _ = cv2.findContours(concrete_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)
        
        # Add measurement text
        if 'concrete_block' in self.measurements:
            m = self.measurements['concrete_block']
            text1 = f"Dimensions: {m['width_cm']:.1f} x {m['height_cm']:.1f} cm"
            text2 = f"Scale: {m['calibration_used']}"
            cv2.putText(vis_image, text1, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_image, text2, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if 'phenophthalein_analysis' in self.measurements:
            pa = self.measurements['phenophthalein_analysis']
            text = f"Affected: {pa['affected_percentage']:.1f}%"
            cv2.putText(vis_image, text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"✓ Measurement visualization saved to {output_path}")
        
        return vis_image
    
    def generate_report(self, output_path: str = "measurement_report.txt") -> str:
        """Generate a detailed text report of all measurements
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Report as string
        """
        report_lines = [
            "="*60,
            "CONCRETE BLOCK ANALYSIS REPORT",
            "Automatic Scale Calibration & Phenophthalein Detection",
            "="*60,
            "",
            "AUTOMATIC CALIBRATION:",
        ]
        
        if self.scale_info:
            report_lines.extend([
                f" Scale Type: {self.scale_info['scale_type']}",
                f" Detection: {self.scale_info['detection_method']}",
                f" Markings Found: {self.scale_info['num_markings_detected']}",
                f" Samples Used: {self.scale_info['samples_used']}",
                f" Pixel/cm ratio: {self.scale_info['pixel_per_cm']:.4f} px/cm",
                f" Pixel/mm ratio: {self.scale_info['pixel_per_mm']:.4f} px/mm",
                f" Accuracy (±): {self.scale_info['std_deviation']:.2f} pixels",
                ""
            ])
        
        if 'concrete_block' in self.measurements:
            m = self.measurements['concrete_block']
            report_lines.extend([
                "CONCRETE BLOCK DIMENSIONS:",
                f" Width: {m['width_mm']:.2f} mm ({m['width_cm']:.2f} cm)",
                f" Height: {m['height_mm']:.2f} mm ({m['height_cm']:.2f} cm)",
                f" Area: {m['area_mm2']:.2f} mm² ({m['area_cm2']:.2f} cm²)",
                f" Perimeter: {m['perimeter_mm']:.2f} mm ({m['perimeter_cm']:.2f} cm)",
                ""
            ])
        
        if 'phenophthalein_analysis' in self.measurements:
            pa = self.measurements['phenophthalein_analysis']
            report_lines.extend([
                "PHENOPHTHALEIN ANALYSIS:",
                f" Total area: {pa['total_mm2']:.2f} mm² ({pa['total_cm2']:.2f} cm²)",
                f" Affected area: {pa['affected_mm2']:.2f} mm² ({pa['affected_cm2']:.2f} cm²)",
                f" Unaffected area: {pa['unaffected_mm2']:.2f} mm² ({pa['unaffected_cm2']:.2f} cm²)",
                f" Affected percentage: {pa['affected_percentage']:.2f}%",
                f" Unaffected percentage: {pa['unaffected_percentage']:.2f}%",
                ""
            ])
        
        report_lines.append("="*60)
        
        report = "\n".join(report_lines)
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Report saved to {output_path}")
        
        return report


if __name__ == "__main__":
    print("Automatic Calibration and Measurement Module")
    print("Detects yellow scale with black markings automatically!")