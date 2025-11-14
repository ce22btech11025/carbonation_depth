# """Smart Adaptive Ruler Calibration - Pattern Learning

# Instead of hardcoding major/minor classification, let the data tell us:

# 1. Detect ALL marking positions
# 2. Analyze spacing pattern to find the repeating unit
# 3. Use spacing regularity to infer calibration
# 4. Adapt to any ruler format (1cm, 0.5cm, etc.)

# Key insight: Ruler markings have PERIODIC spacing pattern!
# If you see many equally-spaced lines, find the smallest regular interval.
# """

# import cv2
# import numpy as np
# from typing import Dict, List, Tuple, Optional
# from scipy import signal


# class SmartRulerCalibration:
#     """Adaptive calibration - learns pattern from marking positions"""

#     def __init__(self):
#         self.pixel_to_mm_ratio = None
#         self.pixel_to_cm_ratio = None
#         self.measurements = {}
#         self.calibration_info = {}

#     def detect_all_markings(self, scale_region: np.ndarray) -> List[int]:
#         """Detect ALL ruler marking positions (no classification)

#         Returns: List of x-positions where markings are detected
#         """
#         print("\n[Marking Detection] Finding all marking lines...")

#         # Convert to grayscale
#         if len(scale_region.shape) == 3:
#             gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = scale_region.copy()

#         # Find vertical edges
#         sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#         abs_sobelx = np.abs(sobelx)
#         _, edges = cv2.threshold(abs_sobelx, 30, 255, cv2.THRESH_BINARY)
#         edges = edges.astype(np.uint8)

#         # Find columns with vertical lines
#         line_strength = np.sum(edges, axis=0)

#         # Find peaks with smaller minimum distance (catch ALL markings)
#         peaks, _ = signal.find_peaks(line_strength, distance=3, height=20)

#         print(f"  ✓ Detected {len(peaks)} marking positions")

#         return sorted(peaks)

#     def analyze_spacing_pattern(self, positions: List[int]) -> Dict:
#         """Analyze spacing pattern to find repeating interval

#         Smart approach: Look at all consecutive spacing differences
#         Find the most common small spacing = calibration unit!
#         """
#         print("\n[Pattern Analysis] Finding repeating spacing pattern...")

#         if len(positions) < 3:
#             print("  ⚠ Too few markings to analyze pattern")
#             return {}

#         # Calculate all consecutive spacings
#         spacings = np.diff(positions)

#         print(f"  Spacings: min={spacings.min():.1f}px, "
#               f"max={spacings.max():.1f}px, "
#               f"mean={spacings.mean():.1f}px")

#         # Find the fundamental spacing unit
#         # Key insight: Minor markings have smallest consistent spacing
#         # Find histogram peak in spacing values

#         hist, bin_edges = np.histogram(spacings, bins=30)
#         most_common_idx = np.argmax(hist)
#         fundamental_spacing = (bin_edges[most_common_idx] + bin_edges[most_common_idx + 1]) / 2

#         print(f"  Fundamental spacing: {fundamental_spacing:.1f} pixels")

#         # Analyze multiples of fundamental spacing
#         # e.g., some markings might be 2x, 3x, 5x, 10x the fundamental
#         multiples = spacings / fundamental_spacing
#         unique_multiples = np.unique(np.round(multiples, 1))

#         print(f"  Found multiples: {unique_multiples}")

#         # The actual calibration unit is likely the GCD or most common multiple
#         # For standard rulers: 1cm = 5 × 0.2cm

#         # Find the metric that makes sense
#         # Standard ruler: major every 1cm (= 5 or 10 minor markings)
#         major_period = None

#         # Look for periodicity - ruler usually has major marks every N minors
#         if len(unique_multiples) > 1:
#             # Common patterns: 5 (0.2cm each), 10 (0.1cm each)
#             for test_period in [5, 10, 4, 2]:
#                 if test_period in unique_multiples:
#                     major_period = test_period
#                     break

#         if major_period is None:
#             # Default assumption: most frequent non-unit multiple is major
#             major_period = int(np.median(unique_multiples[unique_multiples > 1.5]))

#         print(f"  Major marking period: every {major_period} minor markings")

#         # So: minor = fundamental_spacing pixels
#         # And: major = major_period × fundamental_spacing pixels

#         return {
#             'fundamental_spacing': fundamental_spacing,
#             'major_period': major_period,
#             'all_spacings': spacings,
#             'unique_multiples': unique_multiples
#         }

#     def infer_calibration(self, pattern_info: Dict, positions: List[int]) -> Optional[Dict]:
#         """Infer calibration from pattern

#         Strategy: Assume fundamental spacing = 0.2cm or 0.1cm
#         Calculate backward to find actual dimension
#         """
#         print("\n[Calibration Inference] Computing pixel-to-mm ratio...")

#         if not pattern_info:
#             return None

#         fund_spacing = pattern_info['fundamental_spacing']
#         major_period = pattern_info['major_period']

#         # Major marking interval in pixels
#         major_spacing = fund_spacing * major_period

#         # Now we assume: major markings are 1cm apart (standard ruler)
#         # This gives us the calibration!

#         px_per_cm = major_spacing
#         px_per_mm = px_per_cm / 10.0

#         print(f"  Fundamental spacing: {fund_spacing:.2f} px")
#         print(f"  Major period: {major_period}")
#         print(f"  Major spacing: {major_spacing:.2f} px")
#         print(f"  ✓ Inferred calibration: {px_per_cm:.2f} px/cm")

#         # Validate: check if this makes sense
#         # Measure actual positions vs expected positions
#         expected_positions = np.arange(len(positions)) * fund_spacing + positions[0]
#         actual_positions = np.array(positions)

#         error = np.mean(np.abs(expected_positions - actual_positions))
#         error_pct = (error / fund_spacing) * 100

#         print(f"  Pattern fit error: {error_pct:.1f}%")

#         if error_pct > 15:
#             print(f"  ⚠ Warning: Pattern fit is poor, ruler might not be level")

#         calibration = {
#             'method': 'Smart Adaptive Pattern Learning',
#             'num_markings': len(positions),
#             'fundamental_spacing_px': fund_spacing,
#             'major_period': major_period,
#             'major_spacing_px': major_spacing,
#             'pixel_per_cm': px_per_cm,
#             'pixel_per_mm': px_per_mm,
#             'pattern_fit_error_pct': error_pct
#         }

#         return calibration

#     def auto_calibrate(self, image: np.ndarray, scale_mask: np.ndarray) -> Optional[Dict]:
#         """Main calibration"""
#         print("\n" + "="*70)
#         print("SMART ADAPTIVE RULER CALIBRATION")
#         print("(Pattern Learning - Works with any ruler!)")
#         print("="*70)

#         scale_region = cv2.bitwise_and(image, image, mask=scale_mask)

#         # Step 1: Detect ALL markings
#         positions = self.detect_all_markings(scale_region)

#         if len(positions) < 3:
#             print("✗ Calibration failed: Not enough markings")
#             return None

#         # Step 2: Analyze pattern
#         pattern_info = self.analyze_spacing_pattern(positions)

#         if not pattern_info:
#             print("✗ Calibration failed: Could not analyze pattern")
#             return None

#         # Step 3: Infer calibration
#         calibration = self.infer_calibration(pattern_info, positions)

#         if calibration is None:
#             print("\n✗ Calibration failed")
#             return None

#         self.pixel_to_cm_ratio = calibration['pixel_per_cm']
#         self.pixel_to_mm_ratio = calibration['pixel_per_mm']
#         self.calibration_info = calibration

#         print("\n" + "="*70)
#         print("✓ CALIBRATION SUCCESSFUL")
#         print("="*70)

#         return calibration

#     def measure_concrete_block(self, concrete_boundaries: Dict,
#                                 concrete_mask: np.ndarray,
#                                 image: np.ndarray) -> Dict:
#         """Measure concrete block"""
#         if self.pixel_to_mm_ratio is None:
#             raise ValueError("Calibration not performed!")

#         print("\n[Measurement] Measuring concrete block...")

#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         contour = concrete_boundaries['contour']

#         corners = contour.reshape(-1, 1, 2).astype(np.float32)
#         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
#         refined_contour = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
#         refined_contour = refined_contour.reshape(-1, 2)

#         area_px = cv2.contourArea(refined_contour)
#         perimeter_px = cv2.arcLength(refined_contour, closed=True)

#         x_coords = refined_contour[:, 0]
#         y_coords = refined_contour[:, 1]
#         width_px = np.max(x_coords) - np.min(x_coords)
#         height_px = np.max(y_coords) - np.min(y_coords)

#         width_mm = width_px / self.pixel_to_mm_ratio
#         height_mm = height_px / self.pixel_to_mm_ratio
#         area_mm2 = area_px / (self.pixel_to_mm_ratio ** 2)
#         perimeter_mm = perimeter_px / self.pixel_to_mm_ratio

#         measurements = {
#             'width_mm': width_mm,
#             'height_mm': height_mm,
#             'width_cm': width_mm / 10,
#             'height_cm': height_mm / 10,
#             'area_mm2': area_mm2,
#             'area_cm2': area_mm2 / 100,
#             'perimeter_mm': perimeter_mm,
#             'perimeter_cm': perimeter_mm / 10,
#             'calibration_used': f'{self.pixel_to_cm_ratio:.4f} px/cm'
#         }

#         print(f"✓ Dimensions: {width_mm:.2f} × {height_mm:.2f} mm")
#         print(f"✓ Area: {area_mm2:.2f} mm² ({area_mm2/100:.2f} cm²)")

#         self.measurements['concrete_block'] = measurements
#         return measurements

#     def get_affected_area(self, image: np.ndarray, concrete_mask: np.ndarray) -> Dict:
#         """Calculate phenophthalein coverage"""
#         if self.pixel_to_mm_ratio is None:
#             raise ValueError("Calibration not performed!")

#         print("\n[Measurement] Analyzing phenophthalein coverage...")

#         concrete_region = cv2.bitwise_and(image, image, mask=concrete_mask)
#         hsv = cv2.cvtColor(concrete_region, cv2.COLOR_BGR2HSV)

#         lower_magenta = np.array([140, 50, 50])
#         upper_magenta = np.array([170, 255, 255])
#         magenta_mask = cv2.inRange(hsv, lower_magenta, upper_magenta)
#         magenta_mask = cv2.bitwise_and(magenta_mask, magenta_mask, mask=concrete_mask)

#         affected_pixels = np.sum(magenta_mask > 0)
#         total_pixels = np.sum(concrete_mask > 0)

#         affected_area_mm2 = affected_pixels / (self.pixel_to_mm_ratio ** 2)
#         total_area_mm2 = total_pixels / (self.pixel_to_mm_ratio ** 2)
#         affected_pct = (affected_pixels / total_pixels * 100) if total_pixels > 0 else 0

#         area_analysis = {
#             'affected_pixels': affected_pixels,
#             'total_pixels': total_pixels,
#             'affected_mm2': affected_area_mm2,
#             'total_mm2': total_area_mm2,
#             'affected_cm2': affected_area_mm2 / 100,
#             'total_cm2': total_area_mm2 / 100,
#             'affected_percentage': affected_pct,
#             'magenta_mask': magenta_mask
#         }

#         print(f"✓ Affected: {affected_area_mm2:.2f} mm² ({affected_pct:.2f}%)")

#         self.measurements['phenophthalein_analysis'] = area_analysis
#         return area_analysis

#     def create_measurement_visualization(self, image: np.ndarray,
#                                         concrete_mask: np.ndarray,
#                                         magenta_mask: np.ndarray,
#                                         output_path: Optional[str] = None) -> np.ndarray:
#         """Create visualization"""
#         vis_image = image.copy()

#         magenta_overlay = np.zeros_like(vis_image)
#         magenta_overlay[magenta_mask > 0] = [255, 0, 255]
#         vis_image = cv2.addWeighted(vis_image, 0.7, magenta_overlay, 0.3, 0)

#         contours, _ = cv2.findContours(concrete_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:
#             cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)

#         if 'concrete_block' in self.measurements:
#             m = self.measurements['concrete_block']
#             text1 = f"Dims: {m['width_cm']:.2f} x {m['height_cm']:.2f} cm"
#             text2 = f"Area: {m['area_cm2']:.2f} cm²"
#             cv2.putText(vis_image, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             cv2.putText(vis_image, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         if output_path:
#             cv2.imwrite(output_path, vis_image)
#             print(f"✓ Visualization saved to {output_path}")

#         return vis_image

#     def generate_report(self, output_path: str = "calibration_report.txt") -> str:
#         """Generate report"""
#         report_lines = [
#             "="*70,
#             "SMART ADAPTIVE RULER CALIBRATION REPORT",
#             "="*70,
#             ""
#         ]

#         if self.calibration_info:
#             report_lines.extend([
#                 "CALIBRATION:",
#                 f"  Method: {self.calibration_info['method']}",
#                 f"  Markings detected: {self.calibration_info['num_markings']}",
#                 f"  Fundamental spacing: {self.calibration_info['fundamental_spacing_px']:.2f} px",
#                 f"  Major period: {self.calibration_info['major_period']}",
#                 f"  Major spacing: {self.calibration_info['major_spacing_px']:.2f} px",
#                 f"  Pixel/cm: {self.calibration_info['pixel_per_cm']:.2f}",
#                 f"  Pixel/mm: {self.calibration_info['pixel_per_mm']:.6f}",
#                 f"  Pattern fit error: {self.calibration_info['pattern_fit_error_pct']:.1f}%",
#                 ""
#             ])

#         if 'concrete_block' in self.measurements:
#             m = self.measurements['concrete_block']
#             report_lines.extend([
#                 "MEASUREMENTS:",
#                 f"  Width: {m['width_mm']:.2f} mm ({m['width_cm']:.2f} cm)",
#                 f"  Height: {m['height_mm']:.2f} mm ({m['height_cm']:.2f} cm)",
#                 f"  Area: {m['area_cm2']:.2f} cm²",
#                 ""
#             ])

#         report_lines.append("="*70)
#         report = "\n".join(report_lines)

#         with open(output_path, 'w') as f:
#             f.write(report)

#         return report

"""Smart Adaptive Ruler Calibration - Pattern Learning

Enhanced with precise yellow scale detection and 2mm least count support
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal
import scipy.cluster.hierarchy as hcluster


class SmartRulerCalibration:
    """Adaptive calibration - learns pattern from marking positions with enhanced detection"""

    def __init__(self):
        self.pixel_to_mm_ratio = None
        self.pixel_to_cm_ratio = None
        self.measurements = {}
        self.calibration_info = {}

    def detect_yellow_scale_region(self, scale_region: np.ndarray) -> np.ndarray:
        """Enhanced yellow scale detection for better marking visibility"""
        print("\n[Yellow Scale Detection] Isolating yellow ruler area...")
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(scale_region, cv2.COLOR_BGR2HSV)
        
        # Define yellow color range in HSV
        lower_yellow1 = np.array([20, 50, 50])
        upper_yellow1 = np.array([30, 255, 255])
        lower_yellow2 = np.array([15, 50, 50])  # Wider range for yellow
        upper_yellow2 = np.array([35, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_yellow1, upper_yellow1)
        mask2 = cv2.inRange(hsv, lower_yellow2, upper_yellow2)
        yellow_mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply yellow mask to enhance contrast for markings
        enhanced_scale = cv2.bitwise_and(scale_region, scale_region, mask=yellow_mask)
        
        print(f"✓ Yellow scale region enhanced")
        return enhanced_scale

    def detect_all_markings(self, scale_region: np.ndarray) -> List[int]:
        """Enhanced marking detection for yellow scale with black marks"""
        print("\n[Marking Detection] Finding all marking lines...")

        # First enhance the yellow scale region
        enhanced_scale = self.detect_yellow_scale_region(scale_region)

        # Convert to grayscale with contrast enhancement
        if len(enhanced_scale.shape) == 3:
            gray = cv2.cvtColor(enhanced_scale, cv2.COLOR_BGR2GRAY)
        else:
            gray = enhanced_scale.copy()

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Use adaptive threshold for better marking detection
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)

        # Find vertical edges with multiple methods
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.abs(sobelx)
        
        # Combine edge detection methods
        edges_combined = cv2.Canny(gray, 50, 150)
        
        # Use morphological operations to enhance vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Project and find peaks
        line_strength = np.sum(vertical_lines, axis=0)
        
        # Find peaks with multiple methods
        peaks1, _ = signal.find_peaks(line_strength, distance=5, height=np.mean(line_strength)*0.3)
        peaks2, _ = signal.find_peaks(line_strength, distance=3, prominence=np.std(line_strength)*0.5)
        
        # Combine peaks and remove duplicates
        all_peaks = sorted(set(peaks1) | set(peaks2))
        
        # Filter peaks by minimum strength
        min_strength = np.mean(line_strength) * 0.2
        filtered_peaks = [p for p in all_peaks if line_strength[p] > min_strength]
        
        # Cluster nearby peaks (within 3 pixels)
        if len(filtered_peaks) > 1:
            clusters = hcluster.fclusterdata(
                np.array(filtered_peaks).reshape(-1, 1), 
                3, criterion='distance'
            )
            clustered_peaks = []
            for cluster_id in np.unique(clusters):
                cluster_points = np.array(filtered_peaks)[clusters == cluster_id]
                clustered_peaks.append(int(np.mean(cluster_points)))
            filtered_peaks = clustered_peaks

        print(f"✓ Detected {len(filtered_peaks)} marking positions")
        return sorted(filtered_peaks)

    def classify_major_minor_marks(self, positions: List[int], line_strength: np.ndarray) -> Tuple[List[int], List[int]]:
        """Classify marks as major (1cm) and minor (2mm) based on line strength and pattern"""
        if len(positions) < 5:
            return positions, []  # Not enough marks to classify
            
        strengths = [line_strength[pos] for pos in positions]
        mean_strength = np.mean(strengths)
        std_strength = np.std(strengths)
        
        # Major marks are typically stronger (1cm intervals)
        major_threshold = mean_strength + 0.5 * std_strength
        major_positions = [pos for i, pos in enumerate(positions) if strengths[i] > major_threshold]
        
        # Minor marks (2mm intervals)
        minor_positions = [pos for i, pos in enumerate(positions) if strengths[i] <= major_threshold]
        
        # Verify pattern: should have 4 minor marks between major marks (2mm least count)
        if len(major_positions) >= 2:
            major_spacings = np.diff(major_positions)
            expected_minor_count = 4  # 1cm = 5 intervals of 2mm
            
            print(f"✓ Major marks: {len(major_positions)}, Minor marks: {len(minor_positions)}")
            print(f"✓ Expected pattern: 4 minor marks between each major mark (2mm least count)")
            
        return major_positions, minor_positions

    def analyze_spacing_pattern(self, positions: List[int]) -> Dict:
        """Enhanced pattern analysis for 2mm least count rulers"""
        print("\n[Pattern Analysis] Finding repeating spacing pattern...")

        if len(positions) < 6:  # Need at least 3 major intervals for good analysis
            print("  ⚠ Too few markings to analyze pattern")
            return {}

        # Calculate all consecutive spacings
        spacings = np.diff(positions)
        
        print(f"  Spacings: min={spacings.min():.1f}px, "
              f"max={spacings.max():.1f}px, "
              f"mean={spacings.mean():.1f}px, "
              f"std={spacings.std():.1f}px")

        # Use clustering to find fundamental spacing (2mm intervals)
        spacing_clusters = hcluster.fclusterdata(
            spacings.reshape(-1, 1), 
            max(1, np.std(spacings)*0.5), 
            criterion='distance'
        )
        
        # Find the most common spacing (should be 2mm)
        unique_clusters, counts = np.unique(spacing_clusters, return_counts=True)
        most_common_cluster = unique_clusters[np.argmax(counts)]
        fundamental_spacing = np.mean(spacings[spacing_clusters == most_common_cluster])
        
        print(f"  Fundamental spacing (2mm): {fundamental_spacing:.1f} pixels")

        # Look for major mark pattern (every 5 minor marks = 1cm)
        multiples = spacings / fundamental_spacing
        rounded_multiples = np.round(multiples)
        
        # Count occurrences of each multiple
        unique_multiples, multiple_counts = np.unique(rounded_multiples, return_counts=True)
        
        print(f"  Spacing multiples: {dict(zip(unique_multiples.astype(int), multiple_counts))}")

        # The major period should be 5 (1cm = 5 × 2mm)
        major_period = 5
        
        # Verify this makes sense
        expected_major_spacing = fundamental_spacing * major_period
        actual_major_candidates = spacings[np.abs(multiples - major_period) < 0.5]
        
        if len(actual_major_candidates) > 0:
            actual_major_spacing = np.mean(actual_major_candidates)
            consistency = (1 - np.abs(actual_major_spacing - expected_major_spacing) / expected_major_spacing) * 100
            print(f"  Major spacing consistency: {consistency:.1f}%")
            
            if consistency < 80:
                print("  ⚠ Warning: Major spacing pattern inconsistent")
        
        return {
            'fundamental_spacing': fundamental_spacing,
            'major_period': major_period,
            'major_spacing': fundamental_spacing * major_period,
            'all_spacings': spacings,
            'unique_multiples': unique_multiples
        }

    def infer_calibration_2mm_ruler(self, pattern_info: Dict, positions: List[int]) -> Optional[Dict]:
        """Calibration specifically for 2mm least count rulers"""
        print("\n[2mm Ruler Calibration] Computing precise pixel-to-mm ratio...")

        if not pattern_info:
            return None

        fund_spacing = pattern_info['fundamental_spacing']
        major_period = pattern_info['major_period']
        major_spacing = pattern_info['major_spacing']

        # For 2mm least count rulers:
        # fundamental_spacing = 2mm
        # major_spacing (5 × fundamental) = 10mm = 1cm
        
        px_per_2mm = fund_spacing
        px_per_mm = px_per_2mm / 2.0
        px_per_cm = px_per_mm * 10.0

        print(f"  Fundamental spacing: {fund_spacing:.2f} px = 2mm")
        print(f"  Major spacing: {major_spacing:.2f} px = 1cm")
        print(f"  ✓ Calibration: {px_per_mm:.4f} px/mm, {px_per_cm:.2f} px/cm")

        # Validation using major marks
        if len(positions) >= major_period + 1:
            major_positions = positions[::major_period]
            if len(major_positions) >= 2:
                actual_major_spacings = np.diff(major_positions)
                expected_major_spacing = px_per_cm
                
                errors = np.abs(actual_major_spacings - expected_major_spacing) / expected_major_spacing * 100
                avg_error = np.mean(errors)
                
                print(f"  Validation error: {avg_error:.1f}%")
                
                if avg_error > 10:
                    print("  ⚠ Warning: High validation error, calibration may be inaccurate")

        calibration = {
            'method': 'Enhanced 2mm Ruler Pattern Learning',
            'num_markings': len(positions),
            'fundamental_spacing_px': fund_spacing,
            'major_period': major_period,
            'major_spacing_px': major_spacing,
            'pixel_per_2mm': px_per_2mm,
            'pixel_per_mm': px_per_mm,
            'pixel_per_cm': px_per_cm,
            'ruler_type': '2mm_least_count'
        }

        return calibration

    def auto_calibrate(self, image: np.ndarray, scale_mask: np.ndarray) -> Optional[Dict]:
        """Enhanced calibration for 2mm least count rulers"""
        print("\n" + "="*70)
        print("ENHANCED RULER CALIBRATION - 2mm LEAST COUNT")
        print("="*70)

        scale_region = cv2.bitwise_and(image, image, mask=scale_mask)

        # Step 1: Detect ALL markings with enhanced yellow scale detection
        positions = self.detect_all_markings(scale_region)

        if len(positions) < 6:
            print("✗ Calibration failed: Not enough markings detected")
            print(f"  Found only {len(positions)} marks, need at least 6")
            return None

        # Step 2: Enhanced pattern analysis for 2mm rulers
        pattern_info = self.analyze_spacing_pattern(positions)

        if not pattern_info:
            print("✗ Calibration failed: Could not analyze pattern")
            return None

        # Step 3: 2mm ruler specific calibration
        calibration = self.infer_calibration_2mm_ruler(pattern_info, positions)

        if calibration is None:
            print("\n✗ Calibration failed")
            return None

        self.pixel_to_cm_ratio = calibration['pixel_per_cm']
        self.pixel_to_mm_ratio = calibration['pixel_per_mm']
        self.calibration_info = calibration

        print("\n" + "="*70)
        print("✓ ENHANCED CALIBRATION SUCCESSFUL")
        print("="*70)

        return calibration

    def measure_concrete_block(self, concrete_boundaries: Dict,
                                concrete_mask: np.ndarray,
                                image: np.ndarray) -> Dict:
        """Measure concrete block with enhanced precision"""
        if self.pixel_to_mm_ratio is None:
            raise ValueError("Calibration not performed!")

        print("\n[Measurement] Measuring concrete block with subpixel precision...")

        # Use multiple channels for better corner detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contour = concrete_boundaries['contour']

        # Subpixel corner refinement
        corners = contour.reshape(-1, 1, 2).astype(np.float32)
        
        # Define the criteria to stop refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        
        # Refine corners with larger window for better accuracy
        refined_contour = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
        refined_contour = refined_contour.reshape(-1, 2)

        # Calculate geometric properties
        area_px = cv2.contourArea(refined_contour)
        perimeter_px = cv2.arcLength(refined_contour, closed=True)

        # Get bounding rectangle for dimensions
        x_coords = refined_contour[:, 0]
        y_coords = refined_contour[:, 1]
        width_px = np.max(x_coords) - np.min(x_coords)
        height_px = np.max(y_coords) - np.min(y_coords)

        # Convert to real units
        width_mm = width_px / self.pixel_to_mm_ratio
        height_mm = height_px / self.pixel_to_mm_ratio
        area_mm2 = area_px / (self.pixel_to_mm_ratio ** 2)
        perimeter_mm = perimeter_px / self.pixel_to_mm_ratio

        # Calculate additional metrics
        aspect_ratio = width_mm / height_mm if height_mm > 0 else 0

        measurements = {
            'width_mm': width_mm,
            'height_mm': height_mm,
            'width_cm': width_mm / 10,
            'height_cm': height_mm / 10,
            'area_mm2': area_mm2,
            'area_cm2': area_mm2 / 100,
            'perimeter_mm': perimeter_mm,
            'perimeter_cm': perimeter_mm / 10,
            'aspect_ratio': aspect_ratio,
            'calibration_used': f'{self.pixel_to_cm_ratio:.4f} px/cm',
            'precision_note': 'subpixel_refinement'
        }

        print(f"✓ Dimensions: {width_mm:.2f} × {height_mm:.2f} mm")
        print(f"✓ Area: {area_mm2:.2f} mm² ({area_mm2/100:.2f} cm²)")
        print(f"✓ Perimeter: {perimeter_mm:.2f} mm")

        self.measurements['concrete_block'] = measurements
        return measurements

    def get_affected_area(self, image: np.ndarray, concrete_mask: np.ndarray) -> Dict:
        """Calculate phenophthalein coverage with improved color detection"""
        if self.pixel_to_mm_ratio is None:
            raise ValueError("Calibration not performed!")

        print("\n[Measurement] Analyzing phenophthalein coverage...")

        concrete_region = cv2.bitwise_and(image, image, mask=concrete_mask)
        
        # Convert to multiple color spaces for better magenta detection
        hsv = cv2.cvtColor(concrete_region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(concrete_region, cv2.COLOR_BGR2LAB)
        
        # Enhanced magenta detection in HSV
        lower_magenta1 = np.array([140, 40, 40])
        upper_magenta1 = np.array([170, 255, 255])
        magenta_mask_hsv = cv2.inRange(hsv, lower_magenta1, upper_magenta1)
        
        # Additional range for magenta
        lower_magenta2 = np.array([170, 40, 40])
        upper_magenta2 = np.array([180, 255, 255])
        magenta_mask_hsv2 = cv2.inRange(hsv, lower_magenta2, upper_magenta2)
        
        # Combine magenta masks
        magenta_mask = cv2.bitwise_or(magenta_mask_hsv, magenta_mask_hsv2)
        magenta_mask = cv2.bitwise_and(magenta_mask, magenta_mask, mask=concrete_mask)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        magenta_mask = cv2.morphologyEx(magenta_mask, cv2.MORPH_OPEN, kernel)

        affected_pixels = np.sum(magenta_mask > 0)
        total_pixels = np.sum(concrete_mask > 0)

        affected_area_mm2 = affected_pixels / (self.pixel_to_mm_ratio ** 2)
        total_area_mm2 = total_pixels / (self.pixel_to_mm_ratio ** 2)
        affected_pct = (affected_pixels / total_pixels * 100) if total_pixels > 0 else 0

        area_analysis = {
            'affected_pixels': affected_pixels,
            'total_pixels': total_pixels,
            'affected_mm2': affected_area_mm2,
            'total_mm2': total_area_mm2,
            'affected_cm2': affected_area_mm2 / 100,
            'total_cm2': total_area_mm2 / 100,
            'affected_percentage': affected_pct,
            'magenta_mask': magenta_mask
        }

        print(f"✓ Affected area: {affected_area_mm2:.2f} mm² ({affected_pct:.2f}%)")

        self.measurements['phenophthalein_analysis'] = area_analysis
        return area_analysis

    def create_measurement_visualization(self, image: np.ndarray,
                                        concrete_mask: np.ndarray,
                                        magenta_mask: np.ndarray,
                                        output_path: Optional[str] = None) -> np.ndarray:
        """Enhanced visualization with measurement annotations"""
        vis_image = image.copy()

        # Create magenta overlay for affected areas
        magenta_overlay = np.zeros_like(vis_image)
        magenta_overlay[magenta_mask > 0] = [255, 0, 255]  # Magenta color
        vis_image = cv2.addWeighted(vis_image, 0.7, magenta_overlay, 0.3, 0)

        # Draw concrete block contour
        contours, _ = cv2.findContours(concrete_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 3)  # Green contour

        # Add measurement annotations
        if 'concrete_block' in self.measurements:
            m = self.measurements['concrete_block']
            text_lines = [
                f"Dimensions: {m['width_cm']:.2f} x {m['height_cm']:.2f} cm",
                f"Area: {m['area_cm2']:.2f} cm²",
                f"Perimeter: {m['perimeter_cm']:.2f} cm"
            ]
            
            if 'phenophthalein_analysis' in self.measurements:
                pa = self.measurements['phenophthalein_analysis']
                text_lines.append(f"Affected: {pa['affected_percentage']:.1f}%")
            
            # Add text with background for better visibility
            for i, text in enumerate(text_lines):
                y_position = 30 + i * 25
                # Text background
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(vis_image, (10, y_position - 20), 
                            (10 + text_size[0] + 10, y_position + 5), (0, 0, 0), -1)
                # Text
                cv2.putText(vis_image, text, (15, y_position), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"✓ Enhanced visualization saved to {output_path}")

        return vis_image

    def generate_report(self, output_path: str = "calibration_report.txt") -> str:
        """Generate enhanced report with 2mm ruler specifics"""
        report_lines = [
            "="*70,
            "ENHANCED RULER CALIBRATION REPORT - 2mm LEAST COUNT",
            "="*70,
            ""
        ]

        if self.calibration_info:
            report_lines.extend([
                "CALIBRATION:",
                f"  Method: {self.calibration_info['method']}",
                f"  Ruler Type: {self.calibration_info.get('ruler_type', 'Unknown')}",
                f"  Markings detected: {self.calibration_info['num_markings']}",
                f"  Fundamental spacing (2mm): {self.calibration_info['fundamental_spacing_px']:.2f} px",
                f"  Major period: {self.calibration_info['major_period']}",
                f"  Major spacing (1cm): {self.calibration_info['major_spacing_px']:.2f} px",
                f"  Pixel/2mm: {self.calibration_info.get('pixel_per_2mm', 0):.2f}",
                f"  Pixel/mm: {self.calibration_info['pixel_per_mm']:.6f}",
                f"  Pixel/cm: {self.calibration_info['pixel_per_cm']:.2f}",
                ""
            ])

        if 'concrete_block' in self.measurements:
            m = self.measurements['concrete_block']
            report_lines.extend([
                "CONCRETE BLOCK MEASUREMENTS:",
                f"  Width: {m['width_mm']:.2f} mm ({m['width_cm']:.2f} cm)",
                f"  Height: {m['height_mm']:.2f} mm ({m['height_cm']:.2f} cm)",
                f"  Area: {m['area_mm2']:.2f} mm² ({m['area_cm2']:.2f} cm²)",
                f"  Perimeter: {m['perimeter_mm']:.2f} mm ({m['perimeter_cm']:.2f} cm)",
                f"  Aspect Ratio: {m['aspect_ratio']:.2f}",
                ""
            ])

        if 'phenophthalein_analysis' in self.measurements:
            pa = self.measurements['phenophthalein_analysis']
            report_lines.extend([
                "PHENOPHTHALEIN ANALYSIS:",
                f"  Affected Area: {pa['affected_mm2']:.2f} mm² ({pa['affected_cm2']:.2f} cm²)",
                f"  Total Area: {pa['total_mm2']:.2f} mm² ({pa['total_cm2']:.2f} cm²)",
                f"  Percentage Affected: {pa['affected_percentage']:.2f}%",
                ""
            ])

        report_lines.append("="*70)
        report = "\n".join(report_lines)

        with open(output_path, 'w') as f:
            f.write(report)

        return report