"""Calibration and Measurement Module - SPACING PATTERN DETECTION

FIXED APPROACH: Uses spacing pattern to identify 1cm vs 0.2cm markings
- Detect ALL line positions (x-coordinates)
- Group by spacing: large gaps (~420px) = 1cm markings, small gaps (~84px) = 0.2cm
- Primary pattern: 1cm mark, then 4 small marks (0.2cm each), then 1cm mark
- Pattern: [BIG] [small] [small] [small] [small] [BIG] [small] ...

"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy import optimize
from sklearn.cluster import DBSCAN


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
            'is_yellow_scale': yellow_percentage > 30
        }

        print(f"  ✓ Yellow scale detection: {yellow_percentage:.1f}% yellow pixels")
        print(f"  ✓ Scale type: {'YELLOW' if detection_info['is_yellow_scale'] else 'OTHER'}")

        return scale_region, detection_info

    def detect_marking_positions(self, scale_region: np.ndarray, 
                                  scale_mask: np.ndarray) -> Dict:
        """Detect marking positions using edge-based approach

        Finds all vertical black line positions (x-coordinates)

        Args:
            scale_region: Cropped scale region
            scale_mask: Binary mask of scale

        Returns:
            Dictionary with marking positions
        """
        print("\n[Auto-Calibration] Detecting all marking positions...")

        # Convert to grayscale
        gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_and(gray, gray, mask=scale_mask)

        # Threshold to get black markings
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find vertical edges
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)

        # Use Hough to find lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=10,
            minLineLength=5,  # SHORT: detect all marking points
            maxLineGap=5
        )

        if lines is None:
            print("  ⚠ No lines detected!")
            return {'positions': [], 'spacings': []}

        # Extract x-positions of vertical lines
        marking_positions = []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            # Vertical or nearly vertical
            if (70 <= angle <= 110) or (angle <= 20) or (angle >= 160):
                mid_x = (x1 + x2) / 2
                marking_positions.append(mid_x)

        if not marking_positions:
            print("  ⚠ No vertical markings found!")
            return {'positions': [], 'spacings': []}

        # Cluster nearby x-positions (same marking detected multiple times)
        marking_positions = np.array(marking_positions)
        marking_positions = np.sort(marking_positions)

        # Use DBSCAN to cluster nearby positions
        clustering = DBSCAN(eps=5, min_samples=1).fit(marking_positions.reshape(-1, 1))
        labels = clustering.labels_

        # Get cluster centers (unique marking positions)
        unique_positions = []
        for label in np.unique(labels):
            cluster_positions = marking_positions[labels == label]
            center = np.mean(cluster_positions)
            unique_positions.append(center)

        unique_positions = sorted(unique_positions)

        print(f"  ✓ Detected {len(marking_positions)} line segments")
        print(f"  ✓ Clustered into {len(unique_positions)} unique marking positions")

        # Calculate all consecutive spacings
        spacings = []
        for i in range(len(unique_positions) - 1):
            spacing = unique_positions[i + 1] - unique_positions[i]
            spacings.append(spacing)

        spacings = np.array(spacings)

        return {
            'positions': unique_positions,
            'spacings': spacings,
            'num_positions': len(unique_positions)
        }

    def identify_1cm_spacing_pattern(self, spacings: np.ndarray) -> Dict:
        """Identify 1cm vs 0.2cm spacing pattern

        Scale pattern: [1cm mark] [0.2cm] [0.2cm] [0.2cm] [0.2cm] [1cm mark]
        Spacing: [BIG ~420px] [small ~84px] [small] [small] [small] [BIG ~420px]

        Args:
            spacings: Array of consecutive spacings (pixel distances)

        Returns:
            Dictionary with identified 1cm spacing
        """
        print("\n[Pattern Analysis] Identifying 1cm spacing pattern...")

        if len(spacings) == 0:
            return {'spacing_1cm': 0, 'spacing_02cm': 0}

        # Find the two most common spacing values
        # BIG gaps = 1cm marks, small gaps = 0.2cm marks
        # Ratio should be ~5:1 (420/84 ≈ 5)

        # Use histogram to find peaks
        hist, bins = np.histogram(spacings, bins=50)

        # Find peaks
        peak_indices = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peak_indices.append(i)

        if len(peak_indices) < 2:
            # Only one spacing detected - assume it's all one type
            print("  ⚠ Only one spacing pattern detected")
            print(f"    Mean spacing: {np.mean(spacings):.2f} pixels")
            print(f"    Std dev: {np.std(spacings):.2f} pixels")
            return {'spacing_1cm': np.median(spacings), 'spacing_02cm': 0}

        # Get the two largest peaks (biggest gaps first)
        peak_values = [(bins[i] + bins[i+1]) / 2 for i in peak_indices[:2]]
        peak_counts = [hist[i] for i in peak_indices[:2]]

        # Sort by value (descending)
        sorted_peaks = sorted(zip(peak_values, peak_counts), reverse=True)

        large_spacing = sorted_peaks[0][0]  # Should be 1cm
        small_spacing = sorted_peaks[1][0] if len(sorted_peaks) > 1 else large_spacing / 5

        # Verify ratio (should be ~5:1)
        ratio = large_spacing / small_spacing if small_spacing > 0 else 5

        print(f"  [Spacing Detection]:")
        print(f"    Large spacing (1cm): {large_spacing:.2f} pixels (count: {sorted_peaks[0][1]})")
        if len(sorted_peaks) > 1:
            print(f"    Small spacing (0.2cm): {small_spacing:.2f} pixels (count: {sorted_peaks[1][1]})")
        print(f"    Ratio: {ratio:.2f}:1 (expected ~5:1)")

        return {
            'spacing_1cm': large_spacing,
            'spacing_02cm': small_spacing,
            'ratio': ratio
        }

    def validate_and_extract_1cm_markings(self, positions: np.ndarray, 
                                          spacings: np.ndarray,
                                          pattern_info: Dict) -> Tuple[List, float]:
        """Extract ONLY 1cm marking positions using pattern

        Uses the identified spacing to pick every 6th position (1 + 5 small = 6 positions)

        Args:
            positions: All marking x-positions
            spacings: Consecutive spacing values
            pattern_info: Dictionary with identified spacings

        Returns:
            Tuple of (primary_1cm_positions, average_1cm_spacing_pixels)
        """
        print("\n[Pattern Extraction] Extracting 1cm primary markings...")

        spacing_1cm = pattern_info['spacing_1cm']
        spacing_02cm = pattern_info['spacing_02cm']

        if spacing_02cm == 0:
            # No secondary marks detected, use all marks assuming they're 1cm
            print("  ! No 0.2cm marks detected, using all marks as 1cm marks")
            return list(positions), spacing_1cm

        # Identify which spacings are "big" (1cm) vs "small" (0.2cm)
        threshold = (spacing_1cm + spacing_02cm) / 2

        primary_markings = [positions[0]]  # Start with first marking

        for i in range(len(spacings)):
            if spacings[i] > threshold:  # This is a 1cm spacing
                # Next position after big spacing is a 1cm marking
                primary_markings.append(positions[i + 1])

        # Calculate average 1cm spacing
        if len(primary_markings) > 1:
            primary_spacings = []
            for i in range(len(primary_markings) - 1):
                sp = primary_markings[i + 1] - primary_markings[i]
                primary_spacings.append(sp)

            avg_1cm_spacing = np.median(primary_spacings)
        else:
            avg_1cm_spacing = spacing_1cm

        print(f"  ✓ Extracted {len(primary_markings)} primary (1cm) markings")
        print(f"  ✓ Average 1cm spacing: {avg_1cm_spacing:.4f} pixels")

        return primary_markings, avg_1cm_spacing

    def calculate_calibration_from_pattern(self, primary_markings: List, 
                                           avg_spacing_px: float) -> Dict:
        """Calculate final calibration from extracted 1cm markings

        Args:
            primary_markings: List of x-positions of 1cm markings
            avg_spacing_px: Average pixel spacing between 1cm markings

        Returns:
            Dictionary with calibration info
        """
        print("\n[Calibration] Calculating from extracted primary markings...")

        if len(primary_markings) < 2:
            print("  ⚠ Not enough primary markings!")
            return {'px_per_cm': 0, 'spacings': []}

        # Calculate all consecutive 1cm spacings
        spacings = []
        for i in range(len(primary_markings) - 1):
            sp = primary_markings[i + 1] - primary_markings[i]
            spacings.append(sp)
            print(f"  Mark {i} → {i+1}: {sp:.4f} pixels (= 1cm)")

        spacings = np.array(spacings)

        # Robust statistics
        median_spacing = np.median(spacings)
        mean_spacing = np.mean(spacings)
        std_dev = np.std(spacings)

        # IQR filtering
        q1 = np.percentile(spacings, 25)
        q3 = np.percentile(spacings, 75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr if iqr > 0 else median_spacing - 20
        upper = q3 + 1.5 * iqr if iqr > 0 else median_spacing + 20

        inliers = (spacings >= lower) & (spacings <= upper)
        final_spacing = np.median(spacings[inliers]) if np.sum(inliers) > 0 else median_spacing
        final_std = np.std(spacings[inliers]) if np.sum(inliers) > 0 else std_dev

        print(f"\n  [Statistics]:")
        print(f"    Measurements: {len(spacings)}")
        print(f"    Mean: {mean_spacing:.4f} px/cm")
        print(f"    Median: {median_spacing:.4f} px/cm")
        print(f"    Final (IQR-filtered): {final_spacing:.4f} px/cm")
        print(f"    Std Dev: {final_std:.4f} pixels")

        return {
            'px_per_cm': final_spacing,
            'spacings': spacings.tolist(),
            'all_count': len(spacings),
            'inlier_count': np.sum(inliers),
            'std_dev': final_std,
            'mean': mean_spacing,
            'median': median_spacing
        }

    def auto_calibrate_from_scale(self, image: np.ndarray, scale_mask: np.ndarray) -> Dict:
        """MAIN: Auto-calibrate using spacing pattern detection

        Args:
            image: Original preprocessed image
            scale_mask: Binary mask of the detected scale

        Returns:
            Dictionary with calibration results
        """
        print("\n" + "="*70)
        print("AUTOMATIC SCALE CALIBRATION - SPACING PATTERN DETECTION")
        print("="*70)

        # Step 1: Detect yellow scale
        scale_region, yellow_info = self.detect_yellow_scale(image, scale_mask)

        # Step 2: Detect all marking positions
        marking_info = self.detect_marking_positions(scale_region, scale_mask)

        if marking_info['num_positions'] < 3:
            print("\n  ✗ ERROR: Not enough markings detected!")
            return None

        # Step 3: Identify spacing pattern (1cm vs 0.2cm)
        pattern_info = self.identify_1cm_spacing_pattern(marking_info['spacings'])

        # Step 4: Extract only 1cm primary markings
        primary_markings, avg_spacing = self.validate_and_extract_1cm_markings(
            np.array(marking_info['positions']),
            marking_info['spacings'],
            pattern_info
        )

        if len(primary_markings) < 2:
            print("\n  ✗ ERROR: Could not extract primary 1cm markings!")
            return None

        # Step 5: Calculate calibration
        calib_info = self.calculate_calibration_from_pattern(primary_markings, avg_spacing)

        if calib_info['px_per_cm'] == 0:
            print("\n  ✗ ERROR: Calibration failed!")
            return None

        # Store results
        self.pixel_to_cm_ratio = calib_info['px_per_cm']
        self.pixel_to_mm_ratio = self.pixel_to_cm_ratio / 10

        self.scale_info = {
            'scale_type': 'Yellow ruler (cm side): 1cm marks + 5×0.2cm marks',
            'detection_method': 'Spacing pattern detection',
            'all_markings_detected': marking_info['num_positions'],
            'primary_1cm_markings': len(primary_markings),
            'pixel_per_cm': self.pixel_to_cm_ratio,
            'pixel_per_mm': self.pixel_to_mm_ratio,
            'std_deviation': calib_info['std_dev'],
            'all_spacings': calib_info['spacings'],
            'inlier_count': calib_info['inlier_count'],
            'total_samples': calib_info['all_count']
        }

        print("\n" + "="*70)
        print("CALIBRATION COMPLETE - SPACING PATTERN")
        print("="*70)
        print(f"✓ Scale: Yellow ruler with cm markings")
        print(f"✓ All markings detected: {marking_info['num_positions']}")
        print(f"✓ Primary 1cm markings extracted: {len(primary_markings)}")
        print(f"✓ Pattern: 1×1cm + 5×0.2cm marks")
        print(f"✓ Valid measurements: {calib_info['inlier_count']}/{calib_info['all_count']}")
        print(f"✓ Pixel-to-cm ratio: {self.pixel_to_cm_ratio:.6f} pixels/cm")
        print(f"✓ Pixel-to-mm ratio: {self.pixel_to_mm_ratio:.6f} pixels/mm")
        print(f"✓ Precision: ±{calib_info['std_dev']:.4f} pixels")
        print("="*70 + "\n")

        return self.scale_info

    def refine_contour_subpixel(self, contour: np.ndarray, 
                                gray_image: np.ndarray) -> np.ndarray:
        """Refine contour to sub-pixel accuracy using cornerSubPix"""
        corners = contour.reshape(-1, 1, 2).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

        refined_corners = cv2.cornerSubPix(
            gray_image, 
            corners, 
            winSize=(5, 5),
            zeroZone=(-1, -1),
            criteria=criteria
        )

        return refined_corners.reshape(-1, 2)

    def measure_concrete_block(self, concrete_boundaries: Dict,
                                concrete_mask: np.ndarray,
                                image: np.ndarray) -> Dict:
        """Measure concrete block using calibrated ratio"""
        if self.pixel_to_mm_ratio is None:
            raise ValueError("Calibration not performed!")

        print("\n[Measurement] Measuring concrete block...")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contour = concrete_boundaries['contour']
        refined_contour = self.refine_contour_subpixel(contour, gray)

        area_px = cv2.contourArea(refined_contour)
        perimeter_px = cv2.arcLength(refined_contour, closed=True)

        x_coords = refined_contour[:, 0]
        y_coords = refined_contour[:, 1]
        width_px = np.max(x_coords) - np.min(x_coords)
        height_px = np.max(y_coords) - np.min(y_coords)

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
            'calibration_used': f'{self.pixel_to_cm_ratio:.4f} px/cm',
            'measurement_precision': 'Sub-pixel (cornerSubPix)'
        }

        print(f"✓ Concrete block measured:")
        print(f"  Dimensions: {width_mm:.3f} x {height_mm:.3f} mm ({width_mm/10:.2f} x {height_mm/10:.2f} cm)")
        print(f"  Area: {area_mm2:.3f} mm² ({area_mm2/100:.3f} cm²)")
        print(f"  Perimeter: {perimeter_mm:.3f} mm")

        self.measurements['concrete_block'] = measurements
        return measurements

    def get_affected_area(self, image: np.ndarray, concrete_mask: np.ndarray) -> Dict:
        """Calculate phenophthalein affected area"""
        if self.pixel_to_mm_ratio is None:
            raise ValueError("Calibration not performed!")

        print("\n[Measurement] Analyzing phenophthalein coverage...")

        concrete_region = cv2.bitwise_and(image, image, mask=concrete_mask)
        hsv = cv2.cvtColor(concrete_region, cv2.COLOR_BGR2HSV)

        lower_magenta = np.array([140, 50, 50])
        upper_magenta = np.array([170, 255, 255])
        magenta_mask = cv2.inRange(hsv, lower_magenta, upper_magenta)
        magenta_mask = cv2.bitwise_and(magenta_mask, magenta_mask, mask=concrete_mask)

        affected_pixels = np.sum(magenta_mask > 0)
        total_concrete_pixels = np.sum(concrete_mask > 0)
        unaffected_pixels = total_concrete_pixels - affected_pixels

        affected_area_mm2 = affected_pixels / (self.pixel_to_mm_ratio ** 2)
        unaffected_area_mm2 = unaffected_pixels / (self.pixel_to_mm_ratio ** 2)
        total_area_mm2 = total_concrete_pixels / (self.pixel_to_mm_ratio ** 2)

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
        print(f"  Affected: {affected_area_mm2:.3f} mm² ({affected_percentage:.2f}%)")
        print(f"  Unaffected: {unaffected_area_mm2:.3f} mm²")

        self.measurements['phenophthalein_analysis'] = area_analysis
        return area_analysis

    def create_measurement_visualization(self, image: np.ndarray,
                                        concrete_mask: np.ndarray,
                                        magenta_mask: np.ndarray,
                                        output_path: Optional[str] = None) -> np.ndarray:
        """Create visualization"""
        vis_image = image.copy()

        magenta_overlay = np.zeros_like(vis_image)
        magenta_overlay[magenta_mask > 0] = [255, 0, 255]
        vis_image = cv2.addWeighted(vis_image, 0.7, magenta_overlay, 0.3, 0)

        contours, _ = cv2.findContours(concrete_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)

        if 'concrete_block' in self.measurements:
            m = self.measurements['concrete_block']
            text1 = f"Dims: {m['width_cm']:.2f} x {m['height_cm']:.2f} cm"
            text2 = f"Area: {m['area_cm2']:.2f} cm²"
            cv2.putText(vis_image, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_image, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if 'phenophthalein_analysis' in self.measurements:
            pa = self.measurements['phenophthalein_analysis']
            text = f"Affected: {pa['affected_percentage']:.2f}%"
            cv2.putText(vis_image, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"✓ Visualization saved to {output_path}")

        return vis_image

    def generate_report(self, output_path: str = "measurement_report.txt") -> str:
        """Generate detailed report"""
        report_lines = [
            "="*70,
            "CONCRETE BLOCK ANALYSIS REPORT",
            "Spacing Pattern-Based Calibration",
            "="*70,
            "",
            "CALIBRATION (SPACING PATTERN DETECTION):",
        ]

        if self.scale_info:
            report_lines.extend([
                f"  Scale Type: {self.scale_info['scale_type']}",
                f"  Detection: {self.scale_info['detection_method']}",
                f"  All Markings Detected: {self.scale_info['all_markings_detected']}",
                f"  Primary 1cm Markings: {self.scale_info['primary_1cm_markings']}",
                f"  Valid Samples: {self.scale_info['inlier_count']}/{self.scale_info['total_samples']}",
                f"  Pixel/cm ratio: {self.scale_info['pixel_per_cm']:.6f} px/cm",
                f"  Pixel/mm ratio: {self.scale_info['pixel_per_mm']:.6f} px/mm",
                f"  Precision (±): {self.scale_info['std_deviation']:.4f} pixels",
                ""
            ])

        if 'concrete_block' in self.measurements:
            m = self.measurements['concrete_block']
            report_lines.extend([
                "CONCRETE BLOCK DIMENSIONS:",
                f"  Width: {m['width_mm']:.3f} mm ({m['width_cm']:.3f} cm)",
                f"  Height: {m['height_mm']:.3f} mm ({m['height_cm']:.3f} cm)",
                f"  Area: {m['area_mm2']:.3f} mm² ({m['area_cm2']:.3f} cm²)",
                f"  Perimeter: {m['perimeter_mm']:.3f} mm ({m['perimeter_cm']:.3f} cm)",
                f"  Method: {m['measurement_precision']}",
                ""
            ])

        if 'phenophthalein_analysis' in self.measurements:
            pa = self.measurements['phenophthalein_analysis']
            report_lines.extend([
                "PHENOPHTHALEIN ANALYSIS:",
                f"  Total area: {pa['total_mm2']:.3f} mm² ({pa['total_cm2']:.3f} cm²)",
                f"  Affected: {pa['affected_mm2']:.3f} mm² ({pa['affected_percentage']:.2f}%)",
                f"  Unaffected: {pa['unaffected_mm2']:.3f} mm²",
                ""
            ])

        report_lines.append("="*70)
        report = "\n".join(report_lines)

        with open(output_path, 'w') as f:
            f.write(report)

        print(f"✓ Report saved to {output_path}")
        return report


if __name__ == "__main__":
    print("Spacing Pattern-Based Calibration Module")