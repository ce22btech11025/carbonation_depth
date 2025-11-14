"""Main Pipeline - SMART ADAPTIVE CALIBRATION

Uses pattern learning to work with ANY ruler format!
"""

import cv2
import os
import sys
from pathlib import Path

try:
    from image_preprocessor import ImagePreprocessor
    from segmentation_module import SegmentationModule
    from ocr_calibration_v1_SMART import SmartRulerCalibration
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class ConcreteAnalysisPipeline:
    """Analysis pipeline with smart adaptive calibration"""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.preprocessor = ImagePreprocessor()
        self.segmenter = SegmentationModule()
        self.calibrator = SmartRulerCalibration()

    def run_pipeline(self, image_path: str):
        """Run complete analysis"""
        print("\n" + "="*70)
        print("CONCRETE BLOCK ANALYSIS PIPELINE")
        print("Calibration: Smart Adaptive Pattern Learning")
        print("="*70 + "\n")

        device_info = self.segmenter.get_device_info()
        print(f"🖥️ Device: {device_info['device'].upper()}")
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
            segmentation_results = self.segmenter.segment_and_extract(
                self.preprocessed_image
            )

            if segmentation_results.get('masks'):
                self.segmenter.visualize_segmentation(
                    self.preprocessed_image,
                    segmentation_results['masks'],
                    output_path=str(self.output_dir / "02_segmentation.jpg")
                )

                for name, mask in segmentation_results.get('masks', {}).items():
                    cv2.imwrite(str(self.output_dir / f"mask_{name}.jpg"), mask)

            print("✓ Stage 2 complete\n")
        except Exception as e:
            print(f"✗ Stage 2 failed: {e}")
            return

        # STAGE 3: Calibration & Measurement
        print("[STAGE 3] Calibration & Measurement...")
        print("-" * 40)
        try:
            scale_mask = segmentation_results.get('masks', {}).get('scale')
            if scale_mask is None:
                print("✗ No scale detected!")
                return

            # Smart adaptive calibration
            print("\n>>> SMART PATTERN LEARNING <<<\n")
            calibration_info = self.calibrator.auto_calibrate(
                self.preprocessed_image,
                scale_mask
            )

            if calibration_info is None:
                print("\n✗ Calibration failed!")
                return

            print(f"\n✓ Calibration successful!")
            print(f" Method: {calibration_info.get('method', 'Unknown')}")
            print(f" Markings: {calibration_info.get('num_markings', 0)}")
            print(f" Fundamental spacing: {calibration_info.get('fundamental_spacing_px', 0):.2f} px")
            print(f" Major period: {calibration_info.get('major_period', 0)}")
            print(f" Pixel/mm: {calibration_info.get('pixel_per_mm', 0):.6f}")
            print("="*70)

            # Measurement
            concrete_mask = segmentation_results['masks'].get('concrete_block')
            if concrete_mask is not None and 'concrete_boundaries' in segmentation_results:
                measurements = self.calibrator.measure_concrete_block(
                    segmentation_results['concrete_boundaries'],
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
        print(f"\nOutputs: {self.output_dir.absolute()}")
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Concrete Block Analysis - Smart Adaptive Calibration"
    )
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    pipeline = ConcreteAnalysisPipeline(output_dir=args.output_dir)
    pipeline.run_pipeline(image_path=args.image)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print("Concrete Block Analysis - Smart Adaptive Calibration")
        print("="*50)
        print()
        image_path = input("Enter image path: ").strip()
        if not os.path.exists(image_path):
            print(f"Error: File not found")
            sys.exit(1)
        output_dir = input("Enter output directory [output]: ").strip() or "output"
        pipeline = ConcreteAnalysisPipeline(output_dir=output_dir)
        pipeline.run_pipeline(image_path=image_path)