"""Main Pipeline Script - OCR-BASED CALIBRATION

Complete concrete block analysis with:
- GPU-accelerated segmentation
- OCR-based automatic calibration
- High-precision measurements

Usage:
    python main_v2.py image.jpg
    or
    python main_v2.py image.jpg --method ocr  (use OCR)
    python main_v2.py image.jpg --method spacing  (use old spacing method)
"""

import cv2
import os
import sys
import argparse
from pathlib import Path

# Import custom modules
try:
    from image_preprocessor import ImagePreprocessor
    from segmentation_module import SegmentationModule
    from ocr_calibration_v1 import OCRCalibration
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all module files are in the same directory.")
    sys.exit(1)


class ConcreteAnalysisPipeline:
    def __init__(self, output_dir: str = "output", use_ocr: bool = True):
        """Initialize pipeline

        Args:
            output_dir: Output directory
            use_ocr: Use OCR calibration (True) or old spacing method (False)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.preprocessor = ImagePreprocessor()
        self.segmenter = SegmentationModule()
        self.use_ocr = use_ocr

        if use_ocr:
            try:
                self.calibrator = OCRCalibration(gpu=True)
                print("✓ Using OCR-based calibration")
            except ImportError:
                print("⚠ EasyOCR not available. Install: pip install easyocr")
                print("  Falling back to spacing pattern method")
                self.use_ocr = False

        self.original_image = None
        self.preprocessed_image = None
        self.segmentation_results = None

    def run_pipeline(self, image_path: str):
        """Run complete analysis

        Args:
            image_path: Path to input image
        """
        print("\n" + "="*70)
        print("CONCRETE BLOCK ANALYSIS PIPELINE")
        calibration_method = "OCR-Based" if self.use_ocr else "Spacing Pattern"
        print(f"Calibration Method: {calibration_method}")
        print("="*70 + "\n")

        # Device info
        device_info = self.segmenter.get_device_info()
        print(f"🖥️  Computing Device: {device_info['device'].upper()}")
        if device_info['device'] == 'cuda':
            print(f"   GPU: {device_info['device_name']}")
        print()

        # STAGE 1: Preprocessing
        print("[STAGE 1] Image Preprocessing...")
        print("-" * 40)
        try:
            self.preprocessed_image = self.preprocessor.preprocess(
                image_path=image_path,
                save_path=str(self.output_dir / "01_preprocessed.jpg")
            )
            print("✓ Stage 1 complete\n")
        except Exception as e:
            print(f"✗ Stage 1 failed: {e}")
            return

        # STAGE 2: Segmentation
        print("[STAGE 2] GPU-Accelerated Segmentation...")
        print("-" * 40)
        try:
            self.segmentation_results = self.segmenter.segment_and_extract(
                self.preprocessed_image
            )

            if self.segmentation_results.get('masks'):
                self.segmenter.visualize_segmentation(
                    self.preprocessed_image,
                    self.segmentation_results['masks'],
                    output_path=str(self.output_dir / "02_segmentation.jpg")
                )

                for name, mask in self.segmentation_results.get('masks', {}).items():
                    cv2.imwrite(str(self.output_dir / f"mask_{name}.jpg"), mask)

            print("✓ Stage 2 complete\n")
        except Exception as e:
            print(f"✗ Stage 2 failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # STAGE 3: Calibration & Measurement
        print("[STAGE 3] Calibration & Measurement...")
        print("-" * 40)
        try:
            scale_mask = self.segmentation_results.get('masks', {}).get('scale')

            if scale_mask is None:
                print("✗ No scale detected!")
                return

            # Calibration
            if self.use_ocr:
                print("\n>>> STARTING OCR-BASED CALIBRATION <<<\n")
                calibration_info = self.calibrator.auto_calibrate_ocr(
                    self.preprocessed_image,
                    scale_mask
                )
            else:
                print("\n>>> STARTING SPACING PATTERN CALIBRATION <<<\n")
                # For legacy support, would call old method
                calibration_info = None
                print("Legacy method not available. Install EasyOCR for OCR method.")
                return

            if calibration_info is None:
                print("\n✗ Calibration failed!")
                return

            print(f"\n✓ Calibration successful!")
            print(f"  Method: {calibration_info.get('method', 'Unknown')}")
            print(f"  Numbers detected: {calibration_info.get('detected_numbers', [])}")
            print(f"  Pixel/mm: {calibration_info.get('pixel_per_mm', 0):.6f}")
            print("="*70)

            # Measurement
            concrete_mask = self.segmentation_results['masks'].get('concrete_block')

            if concrete_mask is not None and 'concrete_boundaries' in self.segmentation_results:
                measurements = self.calibrator.measure_concrete_block(
                    self.segmentation_results['concrete_boundaries'],
                    concrete_mask,
                    self.preprocessed_image
                )

                area_analysis = self.calibrator.get_affected_area(
                    self.preprocessed_image,
                    concrete_mask
                )

                self.calibrator.create_measurement_visualization(
                    self.preprocessed_image,
                    concrete_mask,
                    area_analysis['magenta_mask'],
                    output_path=str(self.output_dir / "03_final_analysis.jpg")
                )

                report = self.calibrator.generate_report(
                    output_path=str(self.output_dir / "analysis_report.txt")
                )
                print("\n" + report)
            else:
                print("\n⚠ Concrete block not detected")

            print("\n✓ Stage 3 complete\n")

        except Exception as e:
            print(f"✗ Stage 3 failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # Summary
        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("="*70)
        print(f"\nOutputs saved to: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        for f in sorted(self.output_dir.glob("*")):
            print(f"  - {f.name}")
        print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Concrete Block Analysis - OCR or Spacing Pattern Calibration"
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to input image"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=['ocr', 'spacing'],
        default='ocr',
        help="Calibration method: ocr (recommended) or spacing pattern"
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    pipeline = ConcreteAnalysisPipeline(
        output_dir=args.output_dir,
        use_ocr=(args.method == 'ocr')
    )
    pipeline.run_pipeline(image_path=args.image)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # Interactive mode
        print("Concrete Block Analysis Pipeline")
        print("="*50)
        print("\nMethods available:")
        print("  1. OCR-based (recommended)")
        print("  2. Spacing pattern")

        image_path = input("\nEnter image path: ").strip()

        if not os.path.exists(image_path):
            print(f"Error: File not found: {image_path}")
            sys.exit(1)

        method = input("Select method (ocr/spacing) [ocr]: ").strip().lower() or "ocr"

        pipeline = ConcreteAnalysisPipeline(
            output_dir="output",
            use_ocr=(method == 'ocr')
        )
        pipeline.run_pipeline(image_path=image_path)