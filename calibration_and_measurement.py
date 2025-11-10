"""Calibration and Measurement Module
Calculates pixel-to-physical unit ratios and performs measurements on concrete block
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional

class CalibrationMeasurement:
    def __init__(self):
        """Initialize the calibration and measurement module"""
        self.pixel_to_mm_ratio = None
        self.measurements = {}
    
    def detect_scale_markings(self, scale_mask: np.ndarray, 
                             scale_image: np.ndarray) -> Dict:
        """Detect scale markings/graduations using edge detection
        
        Args:
            scale_mask: Binary mask of the scale
            scale_image: Original scale region image
            
        Returns:
            Dictionary containing detected markings information
        """
        # Apply edge detection on the scale region
        gray = cv2.cvtColor(scale_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Apply Hough Line Transform to detect scale lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=20, maxLineGap=5)
        
        marking_info = {
            'edges': edges,
            'lines': lines,
            'num_lines': len(lines) if lines is not None else 0
        }
        
        # Analyze line patterns to detect graduations
        if lines is not None:
            # Calculate average line spacing (this represents scale divisions)
            line_positions = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Use the midpoint of each line
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                line_positions.append((mid_x, mid_y))
            
            marking_info['line_positions'] = line_positions
            
            # Calculate spacing between consecutive markings
            if len(line_positions) > 1:
                spacings = []
                sorted_positions = sorted(line_positions, key=lambda p: p[0])  # Sort by x
                for i in range(len(sorted_positions) - 1):
                    dist = np.sqrt(
                        (sorted_positions[i+1][0] - sorted_positions[i][0])**2 +
                        (sorted_positions[i+1][1] - sorted_positions[i][1])**2
                    )
                    spacings.append(dist)
                
                marking_info['spacings'] = spacings
                marking_info['avg_spacing'] = np.mean(spacings) if spacings else 0
        
        print(f"✓ Scale markings detected: {marking_info['num_lines']} lines found")
        return marking_info
    
    def calculate_pixel_ratio(self, scale_boundaries: Dict, 
                             actual_scale_length_mm: float) -> float:
        """Calculate pixel to millimeter ratio using scale dimensions
        
        Args:
            scale_boundaries: Dictionary with scale boundary information
            actual_scale_length_mm: Actual physical length of scale in mm
            
        Returns:
            Pixel to millimeter ratio
        """
        # Use the width or height of the scale (whichever is larger)
        scale_pixel_length = max(scale_boundaries['width'], scale_boundaries['height'])
        
        # Calculate ratio: pixels per mm
        self.pixel_to_mm_ratio = scale_pixel_length / actual_scale_length_mm
        
        print(f"✓ Calibration complete: {self.pixel_to_mm_ratio:.4f} pixels/mm")
        print(f"  Scale length: {scale_pixel_length} pixels = {actual_scale_length_mm} mm")
        
        return self.pixel_to_mm_ratio
    
    def measure_concrete_block(self, concrete_boundaries: Dict, 
                               concrete_mask: np.ndarray) -> Dict:
        """Measure concrete block dimensions and area
        
        Args:
            concrete_boundaries: Dictionary with concrete boundary information
            concrete_mask: Binary mask of concrete block
            
        Returns:
            Dictionary with measurements in pixels and mm
        """
        if self.pixel_to_mm_ratio is None:
            raise ValueError("Calibration not performed. Call calculate_pixel_ratio first.")
        
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
            'perimeter_cm': perimeter_mm / 10
        }
        
        print(f"✓ Concrete block measured:")
        print(f"  Dimensions: {width_mm:.2f} x {height_mm:.2f} mm")
        print(f"  Dimensions: {width_mm/10:.2f} x {height_mm/10:.2f} cm")
        print(f"  Area: {area_mm2:.2f} mm² ({area_mm2/100:.2f} cm²)")
        
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
            raise ValueError("Calibration not performed. Call calculate_pixel_ratio first.")
        
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
        
        print(f"\n✓ Phenophthalein analysis complete:")
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
            text = f"Dimensions: {m['width_cm']:.1f} x {m['height_cm']:.1f} cm"
            cv2.putText(vis_image, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if 'phenophthalein_analysis' in self.measurements:
            pa = self.measurements['phenophthalein_analysis']
            text = f"Affected: {pa['affected_percentage']:.1f}%"
            cv2.putText(vis_image, text, (10, 60), 
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
            "Phenophthalein Detection & Measurement",
            "="*60,
            "",
            "CALIBRATION:",
            f"  Pixel to mm ratio: {self.pixel_to_mm_ratio:.4f} pixels/mm",
            ""
        ]
        
        if 'concrete_block' in self.measurements:
            m = self.measurements['concrete_block']
            report_lines.extend([
                "CONCRETE BLOCK DIMENSIONS:",
                f"  Width:  {m['width_mm']:.2f} mm ({m['width_cm']:.2f} cm)",
                f"  Height: {m['height_mm']:.2f} mm ({m['height_cm']:.2f} cm)",
                f"  Area:   {m['area_mm2']:.2f} mm² ({m['area_cm2']:.2f} cm²)",
                f"  Perimeter: {m['perimeter_mm']:.2f} mm ({m['perimeter_cm']:.2f} cm)",
                ""
            ])
        
        if 'phenophthalein_analysis' in self.measurements:
            pa = self.measurements['phenophthalein_analysis']
            report_lines.extend([
                "PHENOPHTHALEIN ANALYSIS:",
                f"  Total area:      {pa['total_mm2']:.2f} mm² ({pa['total_cm2']:.2f} cm²)",
                f"  Affected area:   {pa['affected_mm2']:.2f} mm² ({pa['affected_cm2']:.2f} cm²)",
                f"  Unaffected area: {pa['unaffected_mm2']:.2f} mm² ({pa['unaffected_cm2']:.2f} cm²)",
                f"  Affected percentage:   {pa['affected_percentage']:.2f}%",
                f"  Unaffected percentage: {pa['unaffected_percentage']:.2f}%",
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
    # Example usage
    calibrator = CalibrationMeasurement()
    
    # Load necessary data (this would come from segmentation module)
    image = cv2.imread("preprocessed_output.jpg")
    
    # Example: simulate scale boundaries
    scale_boundaries = {
        'width': 500,  # pixels
        'height': 50,
        'x': 100,
        'y': 100
    }
    
    # User input: actual scale length
    actual_length_mm = float(input("Enter actual scale length in mm: "))
    
    # Calculate calibration
    calibrator.calculate_pixel_ratio(scale_boundaries, actual_length_mm)
    
    print("\n✓ Calibration and measurement module ready!")