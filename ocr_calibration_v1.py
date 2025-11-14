"""OCR-BASED CALIBRATION MODULE - WITH CM-ONLY FILTERING

Reads ruler text and filters to ONLY consider CM markings (top row)
Ignores INCH markings (bottom row) and downward numbers

Key improvements:
- Detects both cm and inch rows using clustering
- Filters to use only the CM row (upper/top)
- Ignores all downward markings
- More robust for dual-sided rulers
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: EasyOCR not installed. Run: pip install easyocr")

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Run: pip install scikit-learn")


class OCRCalibration:
    """OCR-based scale calibration with CM-only filtering"""

    def __init__(self, gpu: bool = True):
        """Initialize OCR reader

        Args:
            gpu: Use GPU for OCR (faster)
        """
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR required. Install: pip install easyocr")

        self.gpu = gpu
        self.reader = easyocr.Reader(['en'], gpu=gpu)
        self.pixel_to_mm_ratio = None
        self.pixel_to_cm_ratio = None
        self.measurements = {}
        self.calibration_info = {}

    def detect_scale_text(self, scale_region: np.ndarray) -> List[Dict]:
        """Detect text on scale using OCR

        Returns:
            List of detected texts with positions
        """
        print("\n[OCR] Reading scale text...")

        # Run OCR
        results = self.reader.readtext(scale_region)

        detected_text = []
        for (bbox, text, confidence) in results:
            # bbox format: ((x1,y1), (x2,y2), (x3,y3), (x4,y4))
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]

            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)

            text_clean = text.strip()

            # Filter to numeric only
            if text_clean.replace('.', '').replace('-', '').isdigit():
                detected_text.append({
                    'text': text_clean,
                    'x': center_x,
                    'y': center_y,
                    'confidence': confidence,
                    'bbox': bbox
                })
                print(f"  Found: '{text_clean}' at ({center_x:.1f}, {center_y:.1f}) "
                      f"(confidence: {confidence:.2f})")

        if not detected_text:
            print("  ⚠ No scale text detected!")
            return []

        # Sort by x-coordinate (left to right)
        detected_text = sorted(detected_text, key=lambda x: x['x'])
        print(f"  ✓ Detected {len(detected_text)} scale numbers")
        return detected_text

    def filter_cm_markings_only(self, detected_texts: List[Dict]) -> List[Dict]:
        """Filter to keep only CM markings (ignore inch markings)

        For dual-sided rulers (cm on top, inches on bottom):
        - Cluster text by y-coordinate (vertical position)
        - Keep only the topmost cluster (CM markings)
        - Discard downward markings (inch markings)

        Args:
            detected_texts: List of all detected texts

        Returns:
            Filtered list with only CM markings
        """
        if len(detected_texts) < 2:
            print("  ⚠ Not enough markings to filter")
            return detected_texts

        print("\n[Filtering] Separating CM from INCH markings...")

        # Get y-coordinates
        ys = np.array([item['y'] for item in detected_texts])

        print(f"  Y-coordinates range: {ys.min():.1f} to {ys.max():.1f}")

        # Method 1: Use K-means clustering if sklearn available
        if SKLEARN_AVAILABLE and len(np.unique(ys)) > 2:
            try:
                km = KMeans(n_clusters=2, random_state=42, n_init=10)
                km.fit(ys.reshape(-1, 1))
                centers = km.cluster_centers_.flatten()

                # CM row is typically at smaller y value (top of image)
                cm_cluster = np.argmin(centers)
                cm_y_center = centers[cm_cluster]
                inch_y_center = centers[1 - cm_cluster]

                print(f"  Cluster 1 (y={cm_y_center:.1f}): CM markings (top)")
                print(f"  Cluster 2 (y={inch_y_center:.1f}): INCH markings (bottom)")

                filtered_texts = [
                    item for idx, item in enumerate(detected_texts)
                    if km.labels_[idx] == cm_cluster
                ]

                print(f"  ✓ Kept {len(filtered_texts)} CM markings")
                print(f"  ✗ Discarded {len(detected_texts) - len(filtered_texts)} inch markings")

                return filtered_texts

            except Exception as e:
                print(f"  ⚠ Clustering failed: {e}. Using fallback method.")

        # Method 2: Fallback - keep top half (simple y-threshold)
        print("  Using y-coordinate threshold method...")
        height = detected_texts[0]['y'] * 2  # Rough estimate of scale height
        threshold_y = np.median(ys)  # Split at median

        filtered_texts = [item for item in detected_texts if item['y'] <= threshold_y]

        print(f"  ✓ Kept {len(filtered_texts)} CM markings (y <= {threshold_y:.1f})")
        print(f"  ✗ Discarded {len(detected_texts) - len(filtered_texts)} inch markings (y > {threshold_y:.1f})")

        return filtered_texts

    def calculate_calibration_from_ocr(self, detected_texts: List[Dict]) -> Optional[Dict]:
        """Calculate pixel-to-mm ratio from OCR detected text positions

        Args:
            detected_texts: List of detected text with positions (already filtered to CM)

        Returns:
            Calibration dictionary or None
        """
        if len(detected_texts) < 2:
            print("✗ Not enough text detected!")
            return None

        print("\n[Calibration] Computing pixel-to-mm ratio from CM markings...")

        # Parse text to numbers
        try:
            numbers = []
            for item in detected_texts:
                try:
                    num = float(item['text'])
                    numbers.append((num, item['x'], item['y']))
                except ValueError:
                    pass

            if len(numbers) < 2:
                print("✗ Could not parse numbers!")
                return None

            numbers = sorted(numbers, key=lambda x: x[0])
            print(f"  Detected CM numbers: {[n[0] for n in numbers]}")

            # Calculate spacings between consecutive numbers
            spacings_1cm = []

            for i in range(len(numbers) - 1):
                num1, x1, y1 = numbers[i]
                num2, x2, y2 = numbers[i + 1]

                pixel_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                cm_dist = num2 - num1

                if abs(cm_dist - 1.0) < 0.5:  # Consecutive numbers = 1cm
                    spacing_px_per_cm = pixel_dist / cm_dist
                    spacings_1cm.append(spacing_px_per_cm)
                    print(f"    {num1:.0f}→{num2:.0f}: {pixel_dist:.2f} px = {cm_dist:.0f} cm "
                          f"({spacing_px_per_cm:.4f} px/cm)")
                elif cm_dist > 1.0:
                    spacing_px_per_cm = pixel_dist / cm_dist
                    print(f"    {num1:.0f}→{num2:.0f}: {pixel_dist:.2f} px = {cm_dist:.0f} cm "
                          f"({spacing_px_per_cm:.4f} px/cm)")
                    spacings_1cm.append(spacing_px_per_cm)

            if not spacings_1cm:
                print("✗ Could not calculate spacings!")
                return None

            # Robust statistics
            spacings_1cm = np.array(spacings_1cm)
            median_px_per_cm = np.median(spacings_1cm)
            mean_px_per_cm = np.mean(spacings_1cm)
            std_dev = np.std(spacings_1cm)
            px_per_mm = median_px_per_cm / 10.0

            calibration = {
                'method': 'OCR-based scale text detection (CM-only)',
                'detected_numbers': [n[0] for n in numbers],
                'num_measurements': len(spacings_1cm),
                'median_px_per_cm': median_px_per_cm,
                'mean_px_per_cm': mean_px_per_cm,
                'std_deviation': std_dev,
                'pixel_per_cm': median_px_per_cm,
                'pixel_per_mm': px_per_mm,
                'all_spacings': spacings_1cm.tolist()
            }

            print(f"\n  [Statistics]:")
            print(f"    Measurements: {len(spacings_1cm)}")
            print(f"    Median: {median_px_per_cm:.4f} px/cm")
            print(f"    Mean: {mean_px_per_cm:.4f} px/cm")
            print(f"    Std Dev: {std_dev:.4f} pixels")
            print(f"\n  ✓ Calibration: {median_px_per_cm:.4f} px/cm ({px_per_mm:.6f} px/mm)")

            return calibration

        except Exception as e:
            print(f"✗ Error: {e}")
            return None

    def auto_calibrate_ocr(self, image: np.ndarray, scale_mask: np.ndarray) -> Optional[Dict]:
        """Main OCR calibration method with CM-only filtering

        Args:
            image: Original image
            scale_mask: Binary mask of scale region

        Returns:
            Calibration dictionary or None
        """
        print("\n" + "="*70)
        print("OCR-BASED SCALE CALIBRATION (CM-ONLY)")
        print("="*70)

        # Extract scale region
        scale_region = cv2.bitwise_and(image, image, mask=scale_mask)

        # Detect ALL text
        detected_texts = self.detect_scale_text(scale_region)

        if not detected_texts:
            print("\n✗ Calibration failed: No text detected")
            return None

        # FILTER TO CM ONLY (removes inch markings)
        detected_texts = self.filter_cm_markings_only(detected_texts)

        if not detected_texts:
            print("\n✗ Calibration failed: No CM markings after filtering")
            return None

        # Calculate calibration
        calibration = self.calculate_calibration_from_ocr(detected_texts)

        if calibration is None:
            print("\n✗ Calibration failed: Could not calculate ratio")
            return None

        # Store
        self.pixel_to_cm_ratio = calibration['pixel_per_cm']
        self.pixel_to_mm_ratio = calibration['pixel_per_mm']
        self.calibration_info = calibration

        print("\n" + "="*70)
        print("✓ OCR CALIBRATION SUCCESSFUL (CM markings only)")
        print("="*70)

        return calibration

    def measure_concrete_block(self, concrete_boundaries: Dict,
                                concrete_mask: np.ndarray,
                                image: np.ndarray) -> Dict:
        """Measure concrete block"""
        if self.pixel_to_mm_ratio is None:
            raise ValueError("Calibration not performed!")

        print("\n[Measurement] Measuring concrete block...")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contour = concrete_boundaries['contour']

        # Sub-pixel refinement
        corners = contour.reshape(-1, 1, 2).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        refined_contour = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        refined_contour = refined_contour.reshape(-1, 2)

        # Dimensions
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
            'width_mm': width_mm,
            'height_mm': height_mm,
            'width_cm': width_mm / 10,
            'height_cm': height_mm / 10,
            'area_mm2': area_mm2,
            'area_cm2': area_mm2 / 100,
            'perimeter_mm': perimeter_mm,
            'perimeter_cm': perimeter_mm / 10,
            'calibration_used': f'{self.pixel_to_cm_ratio:.4f} px/cm'
        }

        print(f"✓ Dimensions: {width_mm:.2f} × {height_mm:.2f} mm")
        print(f"✓ Area: {area_mm2:.2f} mm² ({area_mm2/100:.2f} cm²)")

        self.measurements['concrete_block'] = measurements
        return measurements

    def get_affected_area(self, image: np.ndarray, concrete_mask: np.ndarray) -> Dict:
        """Calculate phenophthalein coverage"""
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

        print(f"✓ Affected: {affected_area_mm2:.2f} mm² ({affected_pct:.2f}%)")

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

        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"✓ Visualization saved to {output_path}")

        return vis_image

    def generate_report(self, output_path: str = "ocr_report.txt") -> str:
        """Generate report"""
        report_lines = [
            "="*70,
            "OCR-BASED CALIBRATION REPORT (CM-ONLY)",
            "="*70,
            ""
        ]

        if self.calibration_info:
            report_lines.extend([
                "CALIBRATION:",
                f"  Method: {self.calibration_info['method']}",
                f"  Detected CM numbers: {self.calibration_info['detected_numbers']}",
                f"  Measurements: {self.calibration_info['num_measurements']}",
                f"  Median: {self.calibration_info['median_px_per_cm']:.4f} px/cm",
                f"  Pixel/mm: {self.calibration_info['pixel_per_mm']:.6f}",
                ""
            ])

        if 'concrete_block' in self.measurements:
            m = self.measurements['concrete_block']
            report_lines.extend([
                "MEASUREMENTS:",
                f"  Width: {m['width_mm']:.2f} mm",
                f"  Height: {m['height_mm']:.2f} mm",
                f"  Area: {m['area_cm2']:.2f} cm²",
                ""
            ])

        report_lines.append("="*70)
        report = "\n".join(report_lines)

        with open(output_path, 'w') as f:
            f.write(report)

        return report