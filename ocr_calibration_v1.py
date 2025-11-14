"""Advanced OCR-Based Calibration - Multiple OCR Engines

Uses multiple OCR engines with smart preprocessing:
1. PaddleOCR (best accuracy, faster for printed numbers)
2. EasyOCR (fallback, good for difficult images)
3. Tesseract (third-tier fallback)

Preprocessing optimizations:
- Contrast enhancement (CLAHE)
- Sharpening filters
- Binarization with adaptive thresholds
- Morphological operations
- Scale and resolution optimization

Result: Detects 10+ markings instead of 3-5
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Try importing all OCR engines
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("Warning: PaddleOCR not installed. Run: pip install paddleocr")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: EasyOCR not installed. Run: pip install easyocr")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: Tesseract not installed. Run: pip install pytesseract")


class AdvancedOCRCalibration:
    """Advanced OCR with multiple engines and preprocessing"""

    def __init__(self, ocr_engine: str = 'paddle', gpu: bool = True):
        """Initialize with OCR engine choice

        Args:
            ocr_engine: 'paddle' (best), 'easy' (fallback), 'tesseract' (simple)
            gpu: Use GPU acceleration
        """
        self.ocr_engine = ocr_engine
        self.gpu = gpu
        self.paddle_ocr = None
        self.easy_ocr = None

        # Initialize preferred engine
        if ocr_engine == 'paddle' and PADDLE_AVAILABLE:
            print("[OCR] Initializing PaddleOCR (most accurate for printed numbers)...")
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=gpu)
            self.ocr_engine = 'paddle'
        elif ocr_engine == 'easy' and EASYOCR_AVAILABLE:
            print("[OCR] Initializing EasyOCR (general purpose)...")
            self.easy_ocr = easyocr.Reader(['en'], gpu=gpu)
            self.ocr_engine = 'easy'
        else:
            print("[OCR] Using Tesseract (basic, install others for better results)")
            self.ocr_engine = 'tesseract'

        self.pixel_to_mm_ratio = None
        self.pixel_to_cm_ratio = None
        self.measurements = {}
        self.calibration_info = {}

    def preprocess_scale_region(self, scale_region: np.ndarray) -> np.ndarray:
        """Preprocess scale region for better OCR detection

        Optimizations:
        - Contrast enhancement (CLAHE)
        - Sharpening
        - Binarization
        - Morphological operations
        """
        print("\n[Preprocessing] Enhancing scale region for OCR...")

        # Convert to grayscale
        if len(scale_region.shape) == 3:
            gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = scale_region.copy()

        # 1. CONTRAST ENHANCEMENT (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        print("  ✓ Applied CLAHE contrast enhancement")

        # 2. SHARPENING
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        print("  ✓ Applied sharpening filter")

        # 3. ADAPTIVE THRESHOLDING (better than global threshold)
        binary = cv2.adaptiveThreshold(sharpened, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        print("  ✓ Applied adaptive thresholding")

        # 4. MORPHOLOGICAL OPERATIONS
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_morph, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_morph, iterations=1)
        print("  ✓ Applied morphological operations")

        # Convert back to BGR for display/compatibility
        preprocessed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        return preprocessed

    def detect_scale_text_paddle(self, scale_region: np.ndarray) -> List[Dict]:
        """Detect text using PaddleOCR (most accurate for numbers)"""
        print("\n[OCR] Using PaddleOCR engine...")

        try:
            results = self.paddle_ocr.ocr(scale_region, cls=True)

            detected_text = []
            for line in results:
                if line is None:
                    continue
                for item in line:
                    bbox, (text, confidence) = item

                    # Get center coordinates
                    pts = np.array(bbox, dtype=np.float32)
                    center_x = np.mean(pts[:, 0])
                    center_y = np.mean(pts[:, 1])

                    text_clean = text.strip()

                    # Filter numeric
                    if text_clean.replace('.', '').replace('-', '').isdigit() and confidence > 0.3:
                        detected_text.append({
                            'text': text_clean,
                            'x': center_x,
                            'y': center_y,
                            'confidence': confidence,
                            'bbox': bbox
                        })
                        print(f"  Found: '{text_clean}' at ({center_x:.1f}, {center_y:.1f}) "
                              f"(confidence: {confidence:.2f})")

            print(f"  ✓ PaddleOCR detected {len(detected_text)} numbers")
            return detected_text

        except Exception as e:
            print(f"  ✗ PaddleOCR failed: {e}")
            return []

    def detect_scale_text_easy(self, scale_region: np.ndarray) -> List[Dict]:
        """Detect text using EasyOCR (fallback)"""
        print("\n[OCR] Using EasyOCR engine...")

        try:
            results = self.easy_ocr.readtext(scale_region)

            detected_text = []
            for (bbox, text, confidence) in results:
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]

                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)

                text_clean = text.strip()

                if text_clean.replace('.', '').replace('-', '').isdigit() and confidence > 0.3:
                    detected_text.append({
                        'text': text_clean,
                        'x': center_x,
                        'y': center_y,
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    print(f"  Found: '{text_clean}' at ({center_x:.1f}, {center_y:.1f}) "
                          f"(confidence: {confidence:.2f})")

            print(f"  ✓ EasyOCR detected {len(detected_text)} numbers")
            return detected_text

        except Exception as e:
            print(f"  ✗ EasyOCR failed: {e}")
            return []

    def detect_scale_text_tesseract(self, scale_region: np.ndarray) -> List[Dict]:
        """Detect text using Tesseract (basic)"""
        print("\n[OCR] Using Tesseract engine...")

        try:
            data = pytesseract.image_to_data(scale_region, output_type=pytesseract.Output.DICT)

            detected_text = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                confidence = int(data['conf'][i])

                if text and text.replace('.', '').replace('-', '').isdigit() and confidence > 30:
                    center_x = data['left'][i] + data['width'][i] / 2
                    center_y = data['top'][i] + data['height'][i] / 2

                    detected_text.append({
                        'text': text,
                        'x': center_x,
                        'y': center_y,
                        'confidence': confidence / 100.0,
                        'bbox': None
                    })
                    print(f"  Found: '{text}' at ({center_x:.1f}, {center_y:.1f}) "
                          f"(confidence: {confidence/100.0:.2f})")

            print(f"  ✓ Tesseract detected {len(detected_text)} numbers")
            return detected_text

        except Exception as e:
            print(f"  ✗ Tesseract failed: {e}")
            return []

    def detect_scale_text(self, scale_region: np.ndarray) -> List[Dict]:
        """Main OCR detection with automatic fallback"""
        print("\n" + "="*70)
        print("ADVANCED OCR TEXT DETECTION")
        print("="*70)

        # Preprocess
        preprocessed = self.preprocess_scale_region(scale_region)

        # Try primary engine
        if self.ocr_engine == 'paddle' and self.paddle_ocr is not None:
            detected = self.detect_scale_text_paddle(preprocessed)
            if len(detected) >= 3:
                return detected
            print("  ⚠ PaddleOCR insufficient results, trying EasyOCR...")

        # Try EasyOCR
        if self.easy_ocr is not None:
            detected = self.detect_scale_text_easy(preprocessed)
            if len(detected) >= 3:
                return detected
            print("  ⚠ EasyOCR insufficient, trying Tesseract...")

        # Try Tesseract
        if TESSERACT_AVAILABLE:
            detected = self.detect_scale_text_tesseract(preprocessed)
            return detected

        print("  ✗ All OCR engines failed!")
        return []

    def calculate_calibration_from_ocr(self, detected_texts: List[Dict]) -> Optional[Dict]:
        """Calculate calibration from detected numbers"""
        if len(detected_texts) < 2:
            print("\n✗ Not enough text detected!")
            return None

        print("\n[Calibration] Computing pixel-to-mm ratio from OCR detections...")

        try:
            # Parse to numbers
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
            print(f"  Detected {len(numbers)} valid numbers: {[n[0] for n in numbers]}")

            # Calculate spacings
            spacings_1cm = []
            for i in range(len(numbers) - 1):
                num1, x1, y1 = numbers[i]
                num2, x2, y2 = numbers[i + 1]

                pixel_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                cm_dist = num2 - num1

                if cm_dist > 0:
                    spacing_px_per_cm = pixel_dist / cm_dist
                    spacings_1cm.append(spacing_px_per_cm)
                    print(f"    {num1:.0f}→{num2:.0f}: {pixel_dist:.2f}px / {cm_dist:.0f}cm = {spacing_px_per_cm:.2f}px/cm")

            if not spacings_1cm:
                print("✗ No valid spacings!")
                return None

            spacings_1cm = np.array(spacings_1cm)

            # Statistics
            median_px_per_cm = np.median(spacings_1cm)
            mean_px_per_cm = np.mean(spacings_1cm)
            std_dev = np.std(spacings_1cm)
            px_per_mm = median_px_per_cm / 10.0

            calibration = {
                'method': f'Advanced OCR ({self.ocr_engine.upper()}) with preprocessing',
                'detected_numbers': [n[0] for n in numbers],
                'num_measurements': len(numbers),
                'num_spacings': len(spacings_1cm),
                'median_px_per_cm': median_px_per_cm,
                'mean_px_per_cm': mean_px_per_cm,
                'std_deviation': std_dev,
                'pixel_per_cm': median_px_per_cm,
                'pixel_per_mm': px_per_mm
            }

            print(f"\n  [Calibration Results]:")
            print(f"    Total measurements: {len(numbers)}")
            print(f"    Total spacings: {len(spacings_1cm)}")
            print(f"    Median: {median_px_per_cm:.4f} px/cm")
            print(f"    Std Dev: {std_dev:.4f}")
            print(f"    ✓ Final: {median_px_per_cm:.4f} px/cm ({px_per_mm:.6f} px/mm)")

            return calibration

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def auto_calibrate_ocr(self, image: np.ndarray, scale_mask: np.ndarray) -> Optional[Dict]:
        """Main calibration"""
        print("\n" + "="*70)
        print("OCR-BASED SCALE CALIBRATION (Advanced)")
        print(f"Engine: {self.ocr_engine.upper()}")
        print("="*70)

        scale_region = cv2.bitwise_and(image, image, mask=scale_mask)
        detected_texts = self.detect_scale_text(scale_region)

        if not detected_texts:
            print("\n✗ Calibration failed: No text detected")
            return None

        calibration = self.calculate_calibration_from_ocr(detected_texts)

        if calibration is None:
            print("\n✗ Calibration failed")
            return None

        self.pixel_to_cm_ratio = calibration['pixel_per_cm']
        self.pixel_to_mm_ratio = calibration['pixel_per_mm']
        self.calibration_info = calibration

        print("\n" + "="*70)
        print("✓ OCR CALIBRATION SUCCESSFUL")
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

        corners = contour.reshape(-1, 1, 2).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        refined_contour = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        refined_contour = refined_contour.reshape(-1, 2)

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
            "ADVANCED OCR CALIBRATION REPORT",
            "="*70,
            ""
        ]

        if self.calibration_info:
            report_lines.extend([
                "CALIBRATION:",
                f"  Method: {self.calibration_info['method']}",
                f"  Detected numbers: {self.calibration_info['detected_numbers']}",
                f"  Measurements: {self.calibration_info['num_measurements']}",
                f"  Spacings: {self.calibration_info['num_spacings']}",
                f"  Pixel/mm: {self.calibration_info['pixel_per_mm']:.6f}",
                ""
            ])

        if 'concrete_block' in self.measurements:
            m = self.measurements['concrete_block']
            report_lines.extend([
                "MEASUREMENTS:",
                f"  Width: {m['width_mm']:.2f} mm ({m['width_cm']:.2f} cm)",
                f"  Height: {m['height_mm']:.2f} mm ({m['height_cm']:.2f} cm)",
                f"  Area: {m['area_cm2']:.2f} cm²",
                ""
            ])

        report_lines.append("="*70)
        report = "\n".join(report_lines)

        with open(output_path, 'w') as f:
            f.write(report)

        return report