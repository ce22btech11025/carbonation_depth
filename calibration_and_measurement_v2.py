"""Calibration and Measurement Module - FIXED MARKING DETECTION

Automatically detects yellow scale with ONLY primary black cm markings
Uses smarter line clustering to find correct 1cm spacing

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

    def cluster_marking_lines(self, lines_array: np.ndarray, 
                              eps: float = 15.0,
                              min_samples: int = 2) -> Dict:
        """Cluster similar marking lines using DBSCAN

        Groups lines that are close together (noise/fragments) into single markings

        Args:
            lines_array: Nx4 array of [x1,y1,x2,y2] line segments
            eps: Distance threshold for clustering
            min_samples: Minimum samples to form a cluster

        Returns:
            Dictionary with clustered line positions
        """
        if len(lines_array) == 0:
            return {'clusters': [], 'positions': [], 'num_clusters': 0}

        # Extract midpoints of lines
        midpoints = np.array([
            ((x1 + x2) / 2, (y1 + y2) / 2) 
            for x1, y1, x2, y2 in lines_array
        ])

        # Cluster using DBSCAN (groups nearby lines)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(midpoints)
        labels = clustering.labels_

        # Group lines by cluster
        clusters = {}
        for label, midpoint in zip(labels, midpoints):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(midpoint)

        # Calculate cluster centers (these are the marking positions)
        positions = []
        for label in sorted(clusters.keys()):
            if label >= 0:  # Ignore noise points (label = -1)
                cluster_points = np.array(clusters[label])
                center = np.mean(cluster_points, axis=0)
                positions.append(center)

        # Sort by x-coordinate
        positions = sorted(positions, key=lambda p: p[0])

        return {
            'clusters': clusters,
            'positions': positions,
            'num_clusters': len(positions),
            'all_labels': labels
        }

    def detect_black_markings_improved(self, scale_region: np.ndarray, 
                                       scale_mask: np.ndarray) -> Dict:
        """Detect primary BLACK marking lines with clustering

        Uses improved morphology and clustering to find only major cm markings

        Args:
            scale_region: Cropped scale region
            scale_mask: Binary mask of scale

        Returns:
            Dictionary with detected marking positions
        """
        print("\n[Auto-Calibration] Detecting cm markings with smart clustering...")

        # Convert to grayscale
        gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_and(gray, gray, mask=scale_mask)

        # Threshold to get black markings
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        # Remove noise with morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))

        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)

        # Apply edge detection
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)

        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=20,
            minLineLength=20,  # Longer lines = primary markings
            maxLineGap=5
        )

        if lines is None or len(lines) == 0:
            print("  ⚠ Warning: No marking lines detected!")
            return {'positions': [], 'num_markings': 0, 'all_lines': []}

        # Extract and filter lines
        lines_array = lines.reshape(-1, 4)

        # Filter for perpendicular lines
        filtered_lines = []
        for x1, y1, x2, y2 in lines_array:
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            is_perpendicular = (70 <= angle <= 110) or (angle <= 20) or (angle >= 160)

            if is_perpendicular:
                filtered_lines.append([x1, y1, x2, y2])

        if not filtered_lines:
            print("  ⚠ Warning: No perpendicular lines found!")
            return {'positions': [], 'num_markings': 0, 'all_lines': []}

        filtered_lines = np.array(filtered_lines)

        print(f"  [Hough] Found {len(filtered_lines)} perpendicular line segments")

        # CLUSTER SIMILAR LINES - THIS IS THE KEY!
        cluster_info = self.cluster_marking_lines(filtered_lines, eps=20.0, min_samples=1)

        print(f"  [Clustering] Grouped into {cluster_info['num_clusters']} primary markings")

        marking_info = {
            'lines': filtered_lines,
            'positions': cluster_info['positions'],
            'num_markings': cluster_info['num_clusters'],
            'edges': edges,
            'binary': binary,
            'clusters': cluster_info['clusters']
        }

        return marking_info

    def calculate_cm_spacings_robust(self, marking_positions: List[Tuple[float, float]],
                                     num_samples: int = None) -> Dict:
        """Calculate pixel-to-cm ratio using ALL consecutive markings

        Uses robust statistics (median, IQR-based filtering)

        Args:
            marking_positions: List of (x, y) positions of primary markings
            num_samples: Number of samples to use (None = use all)

        Returns:
            Dictionary with robust spacing measurements
        """
        if len(marking_positions) < 2:
            print("  ⚠ Warning: Not enough markings detected!")
            return {'spacings': [], 'average_px_per_cm': 0, 'std_dev': 0}

        print(f"\n[Auto-Calibration] Analyzing {len(marking_positions)} primary markings...")

        # Calculate ALL consecutive spacings
        spacings = []
        for i in range(len(marking_positions) - 1):
            pos1 = marking_positions[i]
            pos2 = marking_positions[i + 1]

            # Euclidean distance
            distance_px = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            spacings.append(distance_px)
            print(f"  Sample {i+1}: Mark {i} to {i+1} = {distance_px:.4f} pixels (1 cm)")

        spacings = np.array(spacings)

        # Use median and IQR for robust estimation
        median_spacing = np.median(spacings)
        q1 = np.percentile(spacings, 25)
        q3 = np.percentile(spacings, 75)
        iqr = q3 - q1

        # Identify outliers using IQR method
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        inliers = (spacings >= lower_bound) & (spacings <= upper_bound)
        inlier_spacings = spacings[inliers]
        outlier_spacings = spacings[~inliers]

        # Final estimation using median of inliers (most robust)
        final_px_per_cm = np.median(inlier_spacings) if len(inlier_spacings) > 0 else median_spacing
        std_dev = np.std(inlier_spacings) if len(inlier_spacings) > 0 else np.std(spacings)

        spacing_info = {
            'spacings': spacings.tolist(),
            'all_spacings_count': len(spacings),
            'inlier_count': np.sum(inliers),
            'outlier_count': np.sum(~inliers),
            'outlier_indices': np.where(~inliers)[0].tolist(),
            'median_px_per_cm': median_spacing,
            'mean_px_per_cm': np.mean(spacings),
            'final_px_per_cm': final_px_per_cm,
            'std_dev': std_dev,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

        print(f"\n  [Statistics]:")
        print(f"    All measurements: {spacing_info['all_spacings_count']}")
        print(f"    Inliers: {spacing_info['inlier_count']}")
        print(f"    Outliers: {spacing_info['outlier_count']}")
        print(f"    Median: {median_spacing:.4f} pixels/cm")
        print(f"    Mean: {np.mean(spacings):.4f} pixels/cm")
        print(f"    Final (IQR-filtered): {final_px_per_cm:.4f} pixels/cm")
        print(f"    Std Dev (inliers): {std_dev:.4f} pixels")

        if np.sum(~inliers) > 0:
            print(f"    Outliers detected at indices: {np.where(~inliers)[0].tolist()}")

        return spacing_info

    def auto_calibrate_from_scale(self, image: np.ndarray, scale_mask: np.ndarray) -> Dict:
        """MAIN: Automatically calibrate with improved marking detection

        Args:
            image: Original preprocessed image
            scale_mask: Binary mask of the detected scale

        Returns:
            Dictionary with calibration results
        """
        print("\n" + "="*60)
        print("AUTOMATIC SCALE CALIBRATION - IMPROVED")
        print("="*60)

        # Step 1: Detect yellow scale
        scale_region, yellow_info = self.detect_yellow_scale(image, scale_mask)

        if not yellow_info['is_yellow_scale']:
            print("\n  ⚠ Warning: Scale does not appear to be yellow!")
            print("    Continuing with detection anyway...")

        # Step 2: Detect and cluster black markings
        marking_info = self.detect_black_markings_improved(scale_region, scale_mask)

        if marking_info['num_markings'] < 2:
            print("\n  ✗ ERROR: Not enough primary markings detected!")
            return None

        # Step 3: Calculate pixel-to-cm ratio with robust statistics
        spacing_info = self.calculate_cm_spacings_robust(marking_info['positions'])

        if spacing_info['final_px_per_cm'] == 0:
            print("\n  ✗ ERROR: Could not calculate pixel-to-cm ratio!")
            return None

        # Store calibration results
        self.pixel_to_cm_ratio = spacing_info['final_px_per_cm']
        self.pixel_to_mm_ratio = self.pixel_to_cm_ratio / 10

        self.scale_info = {
            'scale_type': 'Yellow ruler with black cm markings',
            'detection_method': 'Improved clustering + IQR filtering',
            'yellow_percentage': yellow_info['yellow_percentage'],
            'num_markings_detected': marking_info['num_markings'],
            'pixel_per_cm': self.pixel_to_cm_ratio,
            'pixel_per_mm': self.pixel_to_mm_ratio,
            'std_deviation': spacing_info['std_dev'],
            'all_spacings': spacing_info['spacings'],
            'inlier_count': spacing_info['inlier_count'],
            'outlier_count': spacing_info['outlier_count']
        }

        print("\n" + "="*60)
        print("CALIBRATION COMPLETE - IMPROVED")
        print("="*60)
        print(f"✓ Method: Clustering + IQR-based filtering")
        print(f"✓ Primary markings: {marking_info['num_markings']}")
        print(f"✓ Valid measurements: {spacing_info['inlier_count']}/{spacing_info['all_spacings_count']}")
        print(f"✓ Pixel-to-cm ratio: {self.pixel_to_cm_ratio:.6f} pixels/cm")
        print(f"✓ Pixel-to-mm ratio: {self.pixel_to_mm_ratio:.6f} pixels/mm")
        print(f"✓ Precision: ±{spacing_info['std_dev']:.4f} pixels")
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
        """Measure concrete block with sub-pixel precision

        Args:
            concrete_boundaries: Dictionary with concrete boundary info
            concrete_mask: Binary mask of concrete block
            image: Original image (for sub-pixel refinement)

        Returns:
            Dictionary with measurements
        """
        if self.pixel_to_mm_ratio is None:
            raise ValueError("Calibration not performed!")

        print("\n[Measurement] Measuring concrete block...")

        # Convert to grayscale for refinement
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get refined contour
        contour = concrete_boundaries['contour']
        refined_contour = self.refine_contour_subpixel(contour, gray)

        # Calculate measurements
        area_px = cv2.contourArea(refined_contour)
        perimeter_px = cv2.arcLength(refined_contour, closed=True)

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
            'measurement_precision': 'Sub-pixel (cornerSubPix)'
        }

        print(f"✓ Concrete block measured:")
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
            "="*60,
            "CONCRETE BLOCK ANALYSIS REPORT",
            "Improved Calibration with Smart Clustering",
            "="*60,
            "",
            "CALIBRATION (IMPROVED):",
        ]

        if self.scale_info:
            report_lines.extend([
                f"  Scale Type: {self.scale_info['scale_type']}",
                f"  Detection: {self.scale_info['detection_method']}",
                f"  Primary Markings: {self.scale_info['num_markings_detected']}",
                f"  Valid Samples: {self.scale_info['inlier_count']}/{self.scale_info['inlier_count'] + self.scale_info['outlier_count']}",
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

        report_lines.append("="*60)
        report = "\n".join(report_lines)

        with open(output_path, 'w') as f:
            f.write(report)

        print(f"✓ Report saved to {output_path}")
        return report


if __name__ == "__main__":
    print("Improved Calibration and Measurement Module")