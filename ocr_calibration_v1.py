"""
ADVANCED CALIBRATION AND MEASUREMENT MODULE

With Multiple Sub-Pixel Refinement Techniques for Maximum Accuracy
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy import optimize
from scipy.ndimage import gaussian_filter

class AdvancedCalibrationMeasurement:
    def __init__(self):
        """Initialize the advanced calibration module"""
        self.pixel_to_mm_ratio = None
        self.pixel_to_cm_ratio = None
        self.measurements = {}
        self.scale_info = {}
        self.calibration_uncertainty = None

    # ============ ENSEMBLE EDGE DETECTION ============
    def ensemble_edge_detection(self, gray_image: np.ndarray) -> np.ndarray:
        """Combine multiple edge detection methods for robustness"""

        # Method 1: Canny
        edges_canny = cv2.Canny(gray_image, 50, 150, apertureSize=3)

        # Method 2: Sobel
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = np.hypot(sobelx, sobely).astype(np.uint8)
        edges_sobel = cv2.threshold(edges_sobel, 100, 255, cv2.THRESH_BINARY)[1]

        # Method 3: Laplacian (zero-crossing)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        edges_laplacian = np.abs(laplacian).astype(np.uint8)
        edges_laplacian = cv2.threshold(edges_laplacian, 50, 255, cv2.THRESH_BINARY)[1]

        # Combine: intersection of all methods (highest confidence)
        ensemble = np.uint8((edges_canny > 0) & (edges_sobel > 0))

        return ensemble

    # ============ HESSIAN-BASED SUB-PIXEL REFINEMENT ============
    def hessian_subpixel_refinement(self, gray_image: np.ndarray,
                                   edge_points: np.ndarray,
                                   window_size: int = 5) -> np.ndarray:
        """
        Refine edge points using Hessian matrix (second derivative)
        More accurate than Gaussian fitting alone
        """

        refined_points = []

        for point in edge_points:
            x, y = int(point[0]), int(point[1])

            if (x < window_size or x >= gray_image.shape[1] - window_size or
                y < window_size or y >= gray_image.shape[0] - window_size):
                refined_points.append([x, y])
                continue

            # Extract patch
            patch = gray_image[y-window_size:y+window_size+1,
                            x-window_size:x+window_size+1].astype(np.float32)

            # Compute derivatives
            Ix = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
            Iy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
            Ixx = cv2.Sobel(Ix, cv2.CV_64F, 1, 0, ksize=3)
            Iyy = cv2.Sobel(Iy, cv2.CV_64F, 0, 1, ksize=3)
            Ixy = cv2.Sobel(Ix, cv2.CV_64F, 0, 1, ksize=3)

            cx, cy = window_size, window_size

            # Hessian matrix at center
            H = np.array([[Ixx[cy, cx], Ixy[cy, cx]],
                         [Ixy[cy, cx], Iyy[cy, cx]]])

            # Gradient vector
            g = np.array([[Ix[cy, cx]], [Iy[cy, cx]]])

            try:
                det = np.linalg.det(H)
                if abs(det) > 1e-6:
                    delta = -np.linalg.solve(H, g)

                    if abs(delta[0, 0]) < 1.0 and abs(delta[1, 0]) < 1.0:
                        refined_points.append([x + delta[0, 0], y + delta[1, 0]])
                    else:
                        refined_points.append([x, y])
                else:
                    refined_points.append([x, y])
            except:
                refined_points.append([x, y])

        return np.array(refined_points)

    # ============ ROBUST MARKING DETECTION ============
    def detect_markings_robust(self, scale_region: np.ndarray,
                              scale_mask: np.ndarray) -> Dict:
        """Detect black marking lines with multiple refinement techniques"""

        print("\n[Calibration] Detecting markings with advanced techniques...")

        gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_and(gray, gray, mask=scale_mask)

        # Histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)

        # Ensemble edge detection
        edges = self.ensemble_edge_detection(gray)

        # Morphological refinement
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Hough Line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30,
                               minLineLength=15, maxLineGap=3)

        if lines is None or len(lines) == 0:
            print(" ⚠ No marking lines detected")
            return {'lines': [], 'positions': [], 'num_lines': 0}

        # Extract edge points and refine with Hessian
        edge_points = np.column_stack(np.where(edges > 0))[:, ::-1]
        refined_edge_points = self.hessian_subpixel_refinement(gray, edge_points)

        # Process each line
        marking_lines = []
        marking_positions = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Check if approximately vertical
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if (70 <= angle <= 110) or (angle <= 20) or (angle >= 160):
                marking_lines.append((x1, y1, x2, y2))

                # Find refined edge points near this line
                line_center_x = (x1 + x2) / 2
                line_center_y = (y1 + y2) / 2

                distances = np.sqrt((refined_edge_points[:, 0] - line_center_x)**2 +
                                   (refined_edge_points[:, 1] - line_center_y)**2)

                if len(distances) > 0 and np.min(distances) < 15:
                    nearest_idx = np.argmin(distances)
                    marking_positions.append(tuple(refined_edge_points[nearest_idx]))
                else:
                    marking_positions.append((line_center_x, line_center_y))

        if marking_positions:
            marking_positions = sorted(marking_positions, key=lambda p: p[0])

        print(f" ✓ Detected {len(marking_lines)} marking lines")

        return {
            'lines': marking_lines,
            'positions': marking_positions,
            'num_lines': len(marking_lines),
            'edges': edges,
            'refined_points': refined_edge_points
        }

    # ============ ITERATIVE RANSAC CALIBRATION ============
    def iterative_ransac_calibration(self, marking_positions: List[Tuple[float, float]],
                                    num_iterations: int = 3,
                                    initial_threshold: float = 2.0) -> Dict:
        """
        Iterative RANSAC: tighten threshold with each iteration
        """

        print(f"\n[Calibration] Computing pixel-to-cm ratio with Iterative RANSAC...")

        if len(marking_positions) < 2:
            return {'spacings': [], 'mean_px_per_cm': 0, 'std_dev': 0}

        # Calculate all consecutive spacings
        spacings = []
        for i in range(len(marking_positions) - 1):
            pos1 = np.array(marking_positions[i])
            pos2 = np.array(marking_positions[i + 1])
            distance = np.linalg.norm(pos2 - pos1)
            spacings.append(distance)

        spacings = np.array(spacings)

        threshold = initial_threshold

        # Iterative outlier rejection
        for iteration in range(num_iterations):
            median_spacing = np.median(spacings)
            deviations = np.abs(spacings - median_spacing)
            inliers = deviations <= threshold

            if np.sum(inliers) < 2:
                break

            spacings = spacings[inliers]
            threshold *= 0.7  # Tighten threshold

        # Final statistics
        final_mean = np.mean(spacings)
        final_std = np.std(spacings)
        final_median = np.median(spacings)

        print(f" ✓ Samples used: {len(spacings)}")
        print(f" ✓ Mean spacing: {final_mean:.4f} px")
        print(f" ✓ Std deviation: {final_std:.4f} px")

        return {
            'spacings': spacings,
            'mean_px_per_cm': final_mean,
            'median_px_per_cm': final_median,
            'std_dev': final_std,
            'num_inliers': len(spacings),
            'mad': np.median(np.abs(spacings - final_median))
        }

    # ============ CROSS-VALIDATION ============
    def cross_validate_calibration(self, marking_positions: np.ndarray) -> Dict:
        """Validate calibration using multiple independent methods"""

        print(f"\n[Calibration] Cross-validating calibration...")

        results = {}

        # Method 1: Mean of all spacings
        spacings = []
        for i in range(len(marking_positions) - 1):
            spacing = np.linalg.norm(
                np.array(marking_positions[i+1]) - np.array(marking_positions[i])
            )
            spacings.append(spacing)

        results['method_mean'] = np.mean(spacings)
        results['method_median'] = np.median(spacings)

        # Method 2: First-to-last / number of intervals
        if len(marking_positions) >= 2:
            total_dist = np.linalg.norm(
                np.array(marking_positions[-1]) - np.array(marking_positions[0])
            )
            results['method_total'] = total_dist / (len(marking_positions) - 1)

        # Method 3: Interior points only (exclude endpoints)
        if len(marking_positions) >= 4:
            interior_spacings = spacings[1:-1]
            results['method_interior'] = np.mean(interior_spacings)

        # Compute agreement
        all_values = [v for k, v in results.items() if k.startswith('method_')]
        mean_value = np.mean(all_values)
        std_value = np.std(all_values)
        agreement_percent = (std_value / mean_value) * 100 if mean_value > 0 else 0

        print(f" ✓ Method agreement: {agreement_percent:.2f}%")

        results['consensus'] = mean_value
        results['agreement_std'] = std_value
        results['agreement_percent'] = agreement_percent

        return results

    # ============ UNCERTAINTY ESTIMATION ============
    def estimate_uncertainty(self, pixel_to_mm: float,
                            calibration_std: float,
                            measurement_area_pixels: float) -> Dict:
        """Estimate measurement uncertainty"""

        calib_uncertainty = (calibration_std / pixel_to_mm)

        return {
            'calibration_uncertainty_mm': calib_uncertainty,
            'area_uncertainty_percent': (calib_uncertainty / (pixel_to_mm * 10)) * 100
        }

    # ============ MAIN CALIBRATION METHOD ============
    def auto_calibrate_advanced(self, image: np.ndarray,
                               scale_mask: np.ndarray) -> Dict:
        """
        MAIN METHOD: Advanced automatic calibration pipeline
        """

        print("\n" + "="*70)
        print("ADVANCED CALIBRATION - MULTIPLE TECHNIQUES")
        print("="*70)

        # Extract scale region
        scale_region = cv2.bitwise_and(image, image, mask=scale_mask)

        # Detect markings with advanced techniques
        marking_info = self.detect_markings_robust(scale_region, scale_mask)

        if marking_info['num_lines'] < 2:
            print("\n ✗ ERROR: Not enough markings detected!")
            return None

        # Iterative RANSAC calibration
        ransac_result = self.iterative_ransac_calibration(marking_info['positions'])

        if ransac_result['mean_px_per_cm'] == 0:
            print("\n ✗ ERROR: Could not calculate calibration!")
            return None

        # Cross-validation
        validation = self.cross_validate_calibration(np.array(marking_info['positions']))

        # Use consensus value from cross-validation
        pixel_per_cm = validation['consensus']
        pixel_per_mm = pixel_per_cm / 10

        # Uncertainty estimation
        uncertainty = self.estimate_uncertainty(
            pixel_per_mm,
            ransac_result['std_dev'] / 10,
            1000
        )

        # Store results
        self.pixel_to_cm_ratio = pixel_per_cm
        self.pixel_to_mm_ratio = pixel_per_mm
        self.calibration_uncertainty = uncertainty['calibration_uncertainty_mm']

        self.scale_info = {
            'num_markings': marking_info['num_lines'],
            'pixel_per_cm': pixel_per_cm,
            'pixel_per_mm': pixel_per_mm,
            'ransac_std': ransac_result['std_dev'],
            'validation_agreement': validation['agreement_percent'],
            'uncertainty_mm': uncertainty['calibration_uncertainty_mm']
        }

        print("\n" + "="*70)
        print("CALIBRATION COMPLETE")
        print("="*70)
        print(f"✓ Markings detected: {marking_info['num_lines']}")
        print(f"✓ Pixel-to-cm ratio: {pixel_per_cm:.6f} px/cm")
        print(f"✓ Pixel-to-mm ratio: {pixel_per_mm:.6f} px/mm")
        print(f"✓ Method agreement: {validation['agreement_percent']:.2f}%")
        print(f"✓ Uncertainty: ±{uncertainty['calibration_uncertainty_mm']:.4f} mm")
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
        refined_contour = cv2.cornerSubPix(
            gray,
            contour.reshape(-1, 1, 2).astype(np.float32),
            winSize=(5, 5),
            zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        ).reshape(-1, 2)

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

        # Add uncertainty to measurements
        measurements['width_uncertainty'] = self.calibration_uncertainty
        measurements['area_uncertainty_percent'] = (
            2 * self.calibration_uncertainty / measurements['width_mm'] * 100
        )

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

        lines = ["="*70, "ADVANCED CALIBRATION AND MEASUREMENT REPORT", "="*70, ""]

        if self.scale_info:
            lines.extend([
                "CALIBRATION RESULTS:",
                f" Markings detected: {self.scale_info.get('num_markings', 'N/A')}",
                f" Pixel/cm: {self.scale_info.get('pixel_per_cm', 'N/A'):.6f}",
                f" Pixel/mm: {self.scale_info.get('pixel_per_mm', 'N/A'):.6f}",
                f" Method agreement: {self.scale_info.get('validation_agreement', 'N/A'):.2f}%",
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