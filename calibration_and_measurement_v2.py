"""Calibration and Measurement Module - WITH SUB-PIXEL PRECISION

Automatically detects yellow scale with black markings and calculates pixel-to-mm ratio
Uses advanced sub-pixel edge detection for high precision measurements

"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy import optimize


class CalibrationMeasurement:
    def __init__(self):
        """Initialize the calibration and measurement module"""
        self.pixel_to_mm_ratio = None
        self.pixel_to_cm_ratio = None
        self.measurements = {}
        self.scale_info = {}

    def gaussian_1d(self, x, k, mu, sigma):
        """1D Gaussian function for edge fitting"""
        return k * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    def subpixel_edge_gaussian_fitting(self, gray_image: np.ndarray, 
                                       edge_points: np.ndarray, 
                                       window_size: int = 5) -> np.ndarray:
        """Refine edge points to sub-pixel accuracy using Gaussian fitting

        Args:
            gray_image: Grayscale image
            edge_points: Integer pixel edge coordinates (Nx2 array)
            window_size: Window size for gradient sampling (default: 5)

        Returns:
            Sub-pixel refined edge coordinates
        """
        refined_points = []

        for point in edge_points:
            x, y = int(point[0]), int(point[1])

            # Skip points too close to border
            if (x < window_size or x >= gray_image.shape[1] - window_size or
                y < window_size or y >= gray_image.shape[0] - window_size):
                refined_points.append([x, y])
                continue

            # Calculate gradient direction
            dx = cv2.Sobel(gray_image[y-1:y+2, x-1:x+2], cv2.CV_64F, 1, 0, ksize=3)[1, 1]
            dy = cv2.Sobel(gray_image[y-1:y+2, x-1:x+2], cv2.CV_64F, 0, 1, ksize=3)[1, 1]

            gradient_mag = np.sqrt(dx**2 + dy**2)
            if gradient_mag < 1e-6:
                refined_points.append([x, y])
                continue

            # Normalize gradient direction
            dx /= gradient_mag
            dy /= gradient_mag

            # Sample along gradient direction
            samples = []
            positions = []
            for i in range(-window_size, window_size + 1):
                sample_x = x + i * dx
                sample_y = y + i * dy

                # Bilinear interpolation
                if (0 <= sample_x < gray_image.shape[1] - 1 and 
                    0 <= sample_y < gray_image.shape[0] - 1):
                    ix, iy = int(sample_x), int(sample_y)
                    fx, fy = sample_x - ix, sample_y - iy

                    val = (gray_image[iy, ix] * (1 - fx) * (1 - fy) +
                           gray_image[iy, ix + 1] * fx * (1 - fy) +
                           gray_image[iy + 1, ix] * (1 - fx) * fy +
                           gray_image[iy + 1, ix + 1] * fx * fy)

                    samples.append(val)
                    positions.append(i)

            if len(samples) < 5:
                refined_points.append([x, y])
                continue

            # Fit Gaussian to gradient profile
            try:
                # Initial parameters: amplitude, mean, std
                p0 = [max(samples) - min(samples), 0, 2.0]
                params, _ = optimize.curve_fit(
                    self.gaussian_1d, 
                    positions, 
                    samples, 
                    p0=p0,
                    maxfev=100
                )

                # Extract sub-pixel offset
                sub_pixel_offset = params[1]  # mu parameter

                # Refine coordinates
                refined_x = x + sub_pixel_offset * dx
                refined_y = y + sub_pixel_offset * dy

                refined_points.append([refined_x, refined_y])
            except:
                # If fitting fails, use original point
                refined_points.append([x, y])

        return np.array(refined_points)

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

    def detect_black_markings_subpixel(self, scale_region: np.ndarray, 
                                       scale_mask: np.ndarray) -> Dict:
        """Detect black marking lines with SUB-PIXEL accuracy

        Args:
            scale_region: Cropped scale region
            scale_mask: Binary mask of scale

        Returns:
            Dictionary with detected marking lines at sub-pixel positions
        """
        print("\n[Auto-Calibration] Detecting black markings with sub-pixel precision...")

        # Convert to grayscale
        gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_and(gray, gray, mask=scale_mask)

        # Threshold to get black markings
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Apply edge detection
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)

        # Detect lines using Hough Transform (initial detection)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=15,
            maxLineGap=3
        )

        if lines is None or len(lines) == 0:
            print("  ⚠ Warning: No marking lines detected!")
            return {'lines': [], 'positions': [], 'num_lines': 0}

        # Extract edge points for sub-pixel refinement
        edge_points = np.column_stack(np.where(edges > 0))[:, ::-1]  # (x, y) format

        # Refine edge points to sub-pixel accuracy
        print("  [Sub-pixel] Refining edge coordinates using Gaussian fitting...")
        refined_edge_points = self.subpixel_edge_gaussian_fitting(gray, edge_points)

        # Filter and cluster lines
        marking_lines = []
        marking_positions = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate line angle
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Keep perpendicular lines
            is_perpendicular = (70 <= angle <= 110) or (angle <= 20) or (angle >= 160)

            if is_perpendicular:
                marking_lines.append((x1, y1, x2, y2))

                # Find refined edge points near this line
                line_center_x = (x1 + x2) / 2
                line_center_y = (y1 + y2) / 2

                # Find nearby refined points
                distances = np.sqrt((refined_edge_points[:, 0] - line_center_x)**2 + 
                                  (refined_edge_points[:, 1] - line_center_y)**2)

                if len(distances) > 0 and np.min(distances) < 10:
                    nearest_idx = np.argmin(distances)
                    refined_pos = refined_edge_points[nearest_idx]
                    marking_positions.append(tuple(refined_pos))
                else:
                    marking_positions.append((line_center_x, line_center_y))

        # Sort positions by x-coordinate
        if marking_positions:
            marking_positions = sorted(marking_positions, key=lambda p: p[0])

        marking_info = {
            'lines': marking_lines,
            'positions': marking_positions,
            'num_lines': len(marking_lines),
            'edges': edges,
            'binary': binary,
            'refined_edge_points': refined_edge_points
        }

        print(f"  ✓ Detected {len(marking_lines)} black marking lines")
        print(f"  ✓ Sub-pixel refinement applied to {len(refined_edge_points)} edge points")

        return marking_info

    def calculate_cm_spacings_ransac(self, marking_positions: List[Tuple[float, float]],
                                     num_samples: int = 4, 
                                     ransac_threshold: float = 2.0) -> Dict:
        """Calculate pixel spacing with RANSAC outlier rejection

        Args:
            marking_positions: List of (x, y) sub-pixel positions
            num_samples: Number of samples to use
            ransac_threshold: RANSAC inlier threshold in pixels

        Returns:
            Dictionary with robust spacing measurements
        """
        print(f"\n[Auto-Calibration] Calculating pixel-to-cm ratio with RANSAC...")

        if len(marking_positions) < 2:
            print("  ⚠ Warning: Not enough markings detected!")
            return {'spacings': [], 'average_px_per_cm': 0, 'std_dev': 0}

        # Calculate all consecutive spacings
        spacings = []
        spacing_pairs = []

        max_samples = min(num_samples, len(marking_positions) - 1)

        for i in range(max_samples):
            if i + 1 < len(marking_positions):
                pos1 = marking_positions[i]
                pos2 = marking_positions[i + 1]

                # Sub-pixel distance calculation
                distance_px = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                spacings.append(distance_px)
                spacing_pairs.append((i, i+1, distance_px))
                print(f"  Sample {i+1}: Mark {i} to {i+1} = {distance_px:.4f} pixels (1 cm)")

        if not spacings:
            return {'spacings': [], 'average_px_per_cm': 0, 'std_dev': 0}

        # RANSAC outlier rejection
        spacings_array = np.array(spacings)
        median_spacing = np.median(spacings_array)

        # Compute absolute deviations
        abs_deviations = np.abs(spacings_array - median_spacing)

        # Identify inliers (within threshold)
        inliers = abs_deviations <= ransac_threshold
        inlier_spacings = spacings_array[inliers]

        if len(inlier_spacings) == 0:
            inlier_spacings = spacings_array

        # Calculate final statistics
        final_px_per_cm = np.mean(inlier_spacings)
        std_dev = np.std(inlier_spacings)
        median_px_per_cm = np.median(inlier_spacings)

        spacing_info = {
            'spacings': spacings,
            'spacing_pairs': spacing_pairs,
            'average_px_per_cm': np.mean(spacings),
            'median_px_per_cm': median_spacing,
            'std_dev': np.std(spacings),
            'ransac_inliers': np.sum(inliers),
            'ransac_outliers': np.sum(~inliers),
            'filtered_px_per_cm': final_px_per_cm,
            'filtered_std_dev': std_dev,
            'num_samples': len(spacings)
        }

        print(f"\n  ✓ Pixel spacing analysis (with RANSAC):")
        print(f"    Average: {np.mean(spacings):.4f} pixels/cm")
        print(f"    Median: {median_spacing:.4f} pixels/cm")
        print(f"    RANSAC inliers: {np.sum(inliers)}/{len(spacings)}")
        print(f"    Final (filtered): {final_px_per_cm:.4f} pixels/cm ± {std_dev:.4f}")

        return spacing_info

    def auto_calibrate_from_scale(self, image: np.ndarray, scale_mask: np.ndarray) -> Dict:
        """MAIN METHOD: Automatically calibrate with SUB-PIXEL precision

        Args:
            image: Original preprocessed image
            scale_mask: Binary mask of the detected scale

        Returns:
            Dictionary with calibration results
        """
        print("\n" + "="*60)
        print("AUTOMATIC SCALE CALIBRATION - SUB-PIXEL PRECISION")
        print("="*60)

        # Step 1: Detect yellow scale
        scale_region, yellow_info = self.detect_yellow_scale(image, scale_mask)

        if not yellow_info['is_yellow_scale']:
            print("\n  ⚠ Warning: Scale does not appear to be yellow!")
            print("    Continuing with detection anyway...")

        # Step 2: Detect black markings with sub-pixel accuracy
        marking_info = self.detect_black_markings_subpixel(scale_region, scale_mask)

        if marking_info['num_lines'] < 2:
            print("\n  ✗ ERROR: Not enough marking lines detected!")
            return None

        # Step 3: Calculate pixel-to-cm ratio with RANSAC
        spacing_info = self.calculate_cm_spacings_ransac(
            marking_info['positions'],
            num_samples=min(4, marking_info['num_lines'] - 1)
        )

        if spacing_info['filtered_px_per_cm'] == 0:
            print("\n  ✗ ERROR: Could not calculate pixel-to-cm ratio!")
            return None

        # Store calibration results
        self.pixel_to_cm_ratio = spacing_info['filtered_px_per_cm']
        self.pixel_to_mm_ratio = self.pixel_to_cm_ratio / 10

        self.scale_info = {
            'scale_type': 'Yellow ruler with black cm markings',
            'detection_method': 'Automatic with sub-pixel precision',
            'yellow_percentage': yellow_info['yellow_percentage'],
            'num_markings_detected': marking_info['num_lines'],
            'samples_used': spacing_info['num_samples'],
            'pixel_per_cm': self.pixel_to_cm_ratio,
            'pixel_per_mm': self.pixel_to_mm_ratio,
            'std_deviation': spacing_info['filtered_std_dev'],
            'ransac_inliers': spacing_info['ransac_inliers'],
            'all_spacings': spacing_info['spacings']
        }

        print("\n" + "="*60)
        print("CALIBRATION COMPLETE - SUB-PIXEL PRECISION")
        print("="*60)
        print(f"✓ Method: Gaussian fitting + RANSAC")
        print(f"✓ Markings detected: {marking_info['num_lines']}")
        print(f"✓ RANSAC inliers: {spacing_info['ransac_inliers']}/{spacing_info['num_samples']}")
        print(f"✓ Pixel-to-cm ratio: {self.pixel_to_cm_ratio:.6f} pixels/cm")
        print(f"✓ Pixel-to-mm ratio: {self.pixel_to_mm_ratio:.6f} pixels/mm")
        print(f"✓ Precision: ±{spacing_info['filtered_std_dev']:.4f} pixels")
        print("="*60 + "\n")

        return self.scale_info

    def refine_contour_subpixel(self, contour: np.ndarray, 
                                gray_image: np.ndarray) -> np.ndarray:
        """Refine contour to sub-pixel accuracy using cornerSubPix

        Args:
            contour: Integer pixel contour
            gray_image: Grayscale image

        Returns:
            Sub-pixel refined contour
        """
        # Convert contour to corner format
        corners = contour.reshape(-1, 1, 2).astype(np.float32)

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

        # Refine using cornerSubPix
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
        """Measure concrete block with SUB-PIXEL precision

        Args:
            concrete_boundaries: Dictionary with concrete boundary info
            concrete_mask: Binary mask of concrete block
            image: Original image (for sub-pixel refinement)

        Returns:
            Dictionary with high-precision measurements
        """
        if self.pixel_to_mm_ratio is None:
            raise ValueError("Calibration not performed!")

        print("\n[Measurement] Measuring concrete block with sub-pixel precision...")

        # Convert to grayscale for sub-pixel refinement
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get contour and refine to sub-pixel accuracy
        contour = concrete_boundaries['contour']
        refined_contour = self.refine_contour_subpixel(contour, gray)

        # Calculate sub-pixel area and perimeter
        area_px = cv2.contourArea(refined_contour)
        perimeter_px = cv2.arcLength(refined_contour, closed=True)

        # Bounding box from refined contour
        x_coords = refined_contour[:, 0]
        y_coords = refined_contour[:, 1]
        width_px = np.max(x_coords) - np.min(x_coords)
        height_px = np.max(y_coords) - np.min(y_coords)

        # Convert to physical units
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
            'measurement_precision': 'Sub-pixel (Gaussian + cornerSubPix)'
        }

        print(f"✓ Concrete block measured (sub-pixel precision):")
        print(f"  Dimensions: {width_mm:.3f} x {height_mm:.3f} mm")
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
            text2 = f"Method: {m['measurement_precision']}"
            cv2.putText(vis_image, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_image, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if 'phenophthalein_analysis' in self.measurements:
            pa = self.measurements['phenophthalein_analysis']
            text = f"Affected: {pa['affected_percentage']:.2f}%"
            cv2.putText(vis_image, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"✓ Visualization saved to {output_path}")

        return vis_image

    def generate_report(self, output_path: str = "measurement_report.txt") -> str:
        """Generate detailed measurement report"""
        report_lines = [
            "="*60,
            "CONCRETE BLOCK ANALYSIS REPORT",
            "Sub-Pixel Precision Measurement",
            "="*60,
            "",
            "CALIBRATION (SUB-PIXEL PRECISION):",
        ]

        if self.scale_info:
            report_lines.extend([
                f"  Scale Type: {self.scale_info['scale_type']}",
                f"  Detection: {self.scale_info['detection_method']}",
                f"  Markings Found: {self.scale_info['num_markings_detected']}",
                f"  RANSAC Inliers: {self.scale_info['ransac_inliers']}",
                f"  Pixel/cm ratio: {self.scale_info['pixel_per_cm']:.6f} px/cm",
                f"  Pixel/mm ratio: {self.scale_info['pixel_per_mm']:.6f} px/mm",
                f"  Precision (±): {self.scale_info['std_deviation']:.4f} pixels",
                ""
            ])

        if 'concrete_block' in self.measurements:
            m = self.measurements['concrete_block']
            report_lines.extend([
                "CONCRETE BLOCK DIMENSIONS (SUB-PIXEL):",
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

        report_lines.append("="*60)
        report = "\n".join(report_lines)

        with open(output_path, 'w') as f:
            f.write(report)

        print(f"✓ Report saved to {output_path}")
        return report


if __name__ == "__main__":
    print("Sub-Pixel Calibration and Measurement Module")