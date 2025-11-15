"""
ADAPTIVE CALIBRATION v3 - DETECTS INDIVIDUAL MARKINGS NOT LINES

Key insight: We don't need perfect lines, we need marking POSITIONS
Strategy: Find contours/blobs of edges that represent marking strokes
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List

class AdvancedCalibrationMeasurement:
    def __init__(self):
        """Initialize the calibration module"""
        self.pixel_to_mm_ratio = None
        self.pixel_to_cm_ratio = None
        self.measurements = {}
        self.scale_info = {}
        self.calibration_uncertainty = None

    # ============ EXTRACT INDIVIDUAL MARKING POSITIONS ============
    def extract_marking_positions(self, gray_image: np.ndarray) -> List[float]:
        """
        Extract X positions of individual markings from edge map
        Uses contour/blob detection instead of line detection
        """
        
        print("\n[Marking Detection] Extracting individual marking positions...")
        
        # Preprocessing
        gray_blur = cv2.medianBlur(gray_image, 3)
        
        # Vertical edge detection (Sobel)
        sobelx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
        edges = np.abs(sobelx).astype(np.uint8)
        
        # Threshold - more permissive than before
        _, edges_bin = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
        
        # Dilate to make markings more visible (small dilation)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        edges_dilated = cv2.dilate(edges_bin, kernel, iterations=1)
        
        # Find contours (each marking stroke = one contour)
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        print(f" Found {len(contours)} potential markings")
        
        if len(contours) < 3:
            print(" ⚠ Too few contours detected")
            return []
        
        # Extract center X position of each contour
        marking_positions = []
        
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size: marking strokes should be tall/long
            if h < 20:  # Must be at least 20 pixels tall
                continue
            
            if w > 15:  # But not too wide (should be thin lines)
                continue
            
            # Center X position
            center_x = x + w / 2.0
            marking_positions.append(center_x)
        
        print(f" After size filtering: {len(marking_positions)} valid markings")
        
        if len(marking_positions) < 3:
            print(" ⚠ Still too few markings after filtering")
            return []
        
        # Sort by X position
        marking_positions.sort()
        
        # Remove duplicates (same marking detected twice)
        final_positions = []
        for pos in marking_positions:
            if not final_positions or abs(pos - final_positions[-1]) > 3:
                final_positions.append(pos)
        
        print(f" After duplicate removal: {len(final_positions)} markings")
        
        return final_positions

    # ============ ALTERNATIVE: PROJECTION-BASED DETECTION ============
    def extract_markings_by_projection(self, gray_image: np.ndarray) -> List[float]:
        """
        Alternative approach: Use horizontal projection to find marking positions
        Works better when markings form clear peaks
        """
        
        print("\n[Marking Detection] Using projection-based method...")
        
        gray_blur = cv2.medianBlur(gray_image, 5)
        
        # Vertical edge detection
        sobelx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
        edges = np.abs(sobelx).astype(np.uint8)
        _, edges_bin = cv2.threshold(edges, 40, 255, cv2.THRESH_BINARY)
        
        # Horizontal projection (sum of edges in each column)
        projection = np.sum(edges_bin, axis=0)
        
        # Find peaks in projection (marking positions)
        # Use local maxima detection
        marking_positions = self._find_projection_peaks(projection, min_distance=8)
        
        print(f" Found {len(marking_positions)} markings via projection peaks")
        
        return marking_positions

    def _find_projection_peaks(self, projection: np.ndarray, 
                              min_distance: int = 8) -> List[float]:
        """Find peaks in 1D projection array"""
        
        # Smooth projection
        projection_smooth = cv2.GaussianBlur(projection.reshape(-1, 1), 
                                            (5, 1), 0).flatten()
        
        # Find local maxima
        peaks = []
        for i in range(min_distance, len(projection_smooth) - min_distance):
            if projection_smooth[i] > projection_smooth[i-1] and \
               projection_smooth[i] > projection_smooth[i+1] and \
               projection_smooth[i] > projection_smooth.mean() * 0.5:
                peaks.append(float(i))
        
        return peaks

    # ============ MANUAL MARKING EXTRACTION (LAST RESORT) ============
    def extract_markings_manual_threshold(self, gray_image: np.ndarray,
                                         threshold: int = 30) -> List[float]:
        """
        Ultra-simple approach: threshold + connected components
        Can adjust threshold manually
        """
        
        print(f"\n[Marking Detection] Using manual threshold method (threshold={threshold})...")
        
        gray_blur = cv2.medianBlur(gray_image, 3)
        sobelx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
        edges = np.abs(sobelx).astype(np.uint8)
        
        _, edges_bin = cv2.threshold(edges, threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        edges_clean = cv2.morphologyEx(edges_bin, cv2.MORPH_CLOSE, kernel_v)
        
        # Connected components
        num_labels, labels = cv2.connectedComponents(edges_clean)
        
        print(f" Found {num_labels} connected components")
        
        # Extract center of each component
        marking_positions = []
        
        for label in range(1, num_labels):  # Skip background (0)
            component_mask = (labels == label).astype(np.uint8) * 255
            
            # Get size
            component_size = np.sum(component_mask) / 255
            
            # Filter by size
            if component_size < 15 or component_size > 1000:
                continue
            
            # Find center X
            cols = np.where(component_mask > 0)[1]  # Column indices
            if len(cols) > 0:
                center_x = np.mean(cols)
                marking_positions.append(center_x)
        
        print(f" Valid markings: {len(marking_positions)}")
        
        if len(marking_positions) > 0:
            marking_positions.sort()
        
        return marking_positions

    # ============ VALIDATE AND REFINE POSITIONS ============
    def validate_marking_positions(self, positions: List[float],
                                  max_deviation_percent: float = 25.0) -> Tuple[List[float], List[float]]:
        """Validate marking positions and return spacings"""
        
        if len(positions) < 3:
            return positions, []
        
        # Calculate spacings
        spacings = []
        for i in range(len(positions) - 1):
            spacing = positions[i+1] - positions[i]
            spacings.append(spacing)
        
        spacings = np.array(spacings)
        
        print(f"\n [Spacing Analysis]")
        print(f" Detected {len(spacings)} spacings")
        print(f" Min: {spacings.min():.2f} px, Max: {spacings.max():.2f} px")
        print(f" Mean: {spacings.mean():.2f} px, Std: {spacings.std():.2f} px")
        
        # Validate consistency (with tolerance)
        median_spacing = np.median(spacings)
        print(f" Median: {median_spacing:.2f} px")
        
        valid_spacings = []
        for spacing in spacings:
            deviation_percent = abs(spacing - median_spacing) / median_spacing * 100
            if deviation_percent <= max_deviation_percent:
                valid_spacings.append(spacing)
        
        print(f" Valid spacings (±{max_deviation_percent}%): {len(valid_spacings)}/{len(spacings)}")
        
        if len(valid_spacings) < 2:
            print(" ⚠ Not enough consistent spacings!")
            return positions, spacings
        
        return positions, valid_spacings

    # ============ MAIN CALIBRATION ============
    def auto_calibrate_advanced(self, image: np.ndarray,
                               scale_mask: np.ndarray) -> Dict:
        """
        MAIN: Adaptive calibration with multiple detection strategies
        """
        
        print("\n" + "="*70)
        print("ADAPTIVE CALIBRATION v3 - MULTIPLE STRATEGIES")
        print("="*70)
        
        # Extract scale region
        scale_region = cv2.bitwise_and(image, image, mask=scale_mask)
        gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)
        
        # Try multiple detection strategies
        marking_positions = []
        
        # Strategy 1: Contour-based
        print("\n[Strategy 1/3] Contour-based detection...")
        positions_1 = self.extract_marking_positions(gray)
        
        if len(positions_1) >= 10:
            print(f" ✓ Strategy 1 successful: {len(positions_1)} markings")
            marking_positions = positions_1
        else:
            # Strategy 2: Projection-based
            print("\n[Strategy 2/3] Projection-based detection...")
            positions_2 = self.extract_markings_by_projection(gray)
            
            if len(positions_2) >= 10:
                print(f" ✓ Strategy 2 successful: {len(positions_2)} markings")
                marking_positions = positions_2
            else:
                # Strategy 3: Manual threshold (more permissive)
                print("\n[Strategy 3/3] Manual threshold detection...")
                positions_3 = self.extract_markings_manual_threshold(gray, threshold=30)
                
                if len(positions_3) >= 10:
                    print(f" ✓ Strategy 3 successful: {len(positions_3)} markings")
                    marking_positions = positions_3
                else:
                    # Try even more permissive
                    print("\n[Strategy 3b] Ultra-permissive threshold...")
                    positions_3b = self.extract_markings_manual_threshold(gray, threshold=15)
                    
                    if len(positions_3b) >= 10:
                        print(f" ✓ Strategy 3b successful: {len(positions_3b)} markings")
                        marking_positions = positions_3b
        
        if len(marking_positions) < 10:
            print("\n ✗ ERROR: Could not detect enough markings with any strategy!")
            print(f"   Best result: {len(marking_positions)} markings")
            print("\n   Troubleshooting:")
            print("   - Check if scale mask is correct")
            print("   - Verify markings are vertical or near-vertical")
            print("   - Try running debug_calibration.py to visualize")
            print("   - Check image contrast/clarity")
            return None
        
        # Validate positions
        positions, valid_spacings = self.validate_marking_positions(
            marking_positions,
            max_deviation_percent=30.0  # More permissive
        )
        
        if len(valid_spacings) < 2:
            print("\n ✗ ERROR: Spacing validation failed!")
            return None
        
        # Calculate calibration
        mean_spacing = np.mean(valid_spacings)
        std_spacing = np.std(valid_spacings)
        
        # Standard ruler: 2mm between markings = 0.2cm
        pixel_per_cm = mean_spacing / 0.2
        pixel_per_mm = pixel_per_cm / 10
        
        uncertainty_mm = (std_spacing / mean_spacing) * 2.0 if mean_spacing > 0 else 0
        
        # Store results
        self.pixel_to_cm_ratio = pixel_per_cm
        self.pixel_to_mm_ratio = pixel_per_mm
        self.calibration_uncertainty = uncertainty_mm
        
        self.scale_info = {
            'num_markings': len(positions),
            'pixel_per_cm': pixel_per_cm,
            'pixel_per_mm': pixel_per_mm,
            'mean_spacing_px': mean_spacing,
            'std_spacing_px': std_spacing,
            'valid_spacings': len(valid_spacings),
            'uncertainty_mm': uncertainty_mm
        }
        
        print("\n" + "="*70)
        print("✓ CALIBRATION SUCCESSFUL")
        print(f" Markings detected: {len(positions)}")
        print(f" Valid spacings: {len(valid_spacings)}")
        print(f" Mean spacing: {mean_spacing:.2f} px")
        print(f" Pixel-to-MM: {pixel_per_mm:.6f}")
        print(f" Pixel-to-CM: {pixel_per_cm:.6f}")
        print(f" Uncertainty: ±{uncertainty_mm:.4f} mm")
        print("="*70 + "\n")
        
        return self.scale_info

    # ============ MEASUREMENT ============
    def measure_concrete_block_with_uncertainty(self, concrete_mask: np.ndarray,
                                               image: np.ndarray) -> Dict:
        """Measure concrete block"""
        
        if self.pixel_to_mm_ratio is None:
            raise ValueError("Calibration not performed!")
        
        print("\n[Measurement] Measuring concrete block...")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        contours, _ = cv2.findContours(concrete_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        contour = max(contours, key=cv2.contourArea)
        
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
        
        area_px = cv2.contourArea(refined_contour)
        perimeter_px = cv2.arcLength(refined_contour, closed=True)
        
        x_coords = refined_contour[:, 0]
        y_coords = refined_contour[:, 1]
        
        width_px = np.max(x_coords) - np.min(x_coords)
        height_px = np.max(y_coords) - np.min(y_coords)
        
        measurements = {
            'width_mm': width_px / self.pixel_to_mm_ratio,
            'height_mm': height_px / self.pixel_to_mm_ratio,
            'area_mm2': area_px / (self.pixel_to_mm_ratio ** 2),
            'area_cm2': area_px / (self.pixel_to_mm_ratio ** 2) / 100,
            'perimeter_mm': perimeter_px / self.pixel_to_mm_ratio,
            'uncertainty_mm': self.calibration_uncertainty
        }
        
        measurements['width_uncertainty'] = self.calibration_uncertainty
        measurements['area_uncertainty_percent'] = (
            2 * self.calibration_uncertainty / measurements['width_mm'] * 100
        ) if measurements['width_mm'] > 0 else 0
        
        print(f"✓ Width: {measurements['width_mm']:.2f} ± {self.calibration_uncertainty:.2f} mm")
        print(f"✓ Height: {measurements['height_mm']:.2f} ± {self.calibration_uncertainty:.2f} mm")
        print(f"✓ Area: {measurements['area_cm2']:.2f} cm²")
        
        self.measurements['concrete_block'] = measurements
        return measurements

    def create_measurement_visualization(self, image: np.ndarray,
                                        concrete_mask: np.ndarray,
                                        output_path: Optional[str] = None) -> np.ndarray:
        """Create visualization"""
        
        vis_image = image.copy()
        
        contours, _ = cv2.findContours(concrete_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)
        
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
        """Generate report"""
        
        lines = ["="*70, "CALIBRATION AND MEASUREMENT REPORT", "="*70, ""]
        
        if self.scale_info:
            lines.extend([
                "CALIBRATION RESULTS:",
                f" Markings detected: {self.scale_info.get('num_markings', 'N/A')}",
                f" Pixel/cm: {self.scale_info.get('pixel_per_cm', 'N/A'):.6f}",
                f" Pixel/mm: {self.scale_info.get('pixel_per_mm', 'N/A'):.6f}",
                f" Mean spacing: {self.scale_info.get('mean_spacing_px', 'N/A'):.2f} px",
                f" Valid spacings used: {self.scale_info.get('valid_spacings', 'N/A')}",
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
                ""
            ])
        
        lines.append("="*70)
        
        report = "\n".join(lines)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Report saved to {output_path}")
        
        return report