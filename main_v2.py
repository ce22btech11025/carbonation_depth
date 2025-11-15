"""
MAIN PIPELINE - ADAPTIVE CALIBRATION v3

Uses multiple detection strategies to handle various scale/ruler configurations
Automatically falls back to next strategy if current one fails

Pipeline:
1. Image Preprocessing
2. GPU-Accelerated Segmentation (SAM2)
3. Adaptive Calibration (Contour → Projection → Manual Threshold)
4. Precision Measurement
5. Reporting
"""

import cv2
import os
import sys
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np

try:
    from image_preprocessor import ImagePreprocessor
    from segmentation_module import SegmentationModule
    from ocr_calibration_v3 import AdvancedCalibrationMeasurement
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

class ConcreteAnalysisPipeline:
    """Analysis pipeline with adaptive calibration"""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.preprocessor = ImagePreprocessor()
        self.segmenter = SegmentationModule()
        self.calibrator = AdvancedCalibrationMeasurement()

        self.results = {
            'preprocessing': None,
            'segmentation': None,
            'calibration': None,
            'measurements': None,
            'analysis': None
        }

    def _print_header(self, title: str) -> None:
        """Print formatted header"""
        print("\n" + "="*70)
        print(f" {title}")
        print("="*70)

    def _print_subheader(self, title: str) -> None:
        """Print formatted subheader"""
        print(f"\n>>> {title} <<<\n")

    def stage_1_preprocessing(self, image_path: str) -> Optional[np.ndarray]:
        """Stage 1: Image Preprocessing"""
        self._print_header("STAGE 1: IMAGE PREPROCESSING")
        print("-" * 70)

        try:
            self.preprocessed_image = self.preprocessor.preprocess(
                image_path=image_path,
                save_path=str(self.output_dir / "01_preprocessed.jpg")
            )

            self.results['preprocessing'] = {
                'status': 'success',
                'shape': self.preprocessed_image.shape,
                'saved': str(self.output_dir / "01_preprocessed.jpg")
            }

            print("✓ Stage 1 complete")
            return self.preprocessed_image

        except Exception as e:
            print(f"✗ Stage 1 failed: {e}")
            self.results['preprocessing'] = {'status': 'failed', 'error': str(e)}
            return None

    def stage_2_segmentation(self, preprocessed_image: np.ndarray) -> Optional[Dict]:
        """Stage 2: GPU-Accelerated Segmentation"""
        self._print_header("STAGE 2: GPU-ACCELERATED SEGMENTATION")
        print("-" * 70)

        device_info = self.segmenter.get_device_info()
        print(f"Device: {device_info['device'].upper()}")

        if device_info['device'] == 'cuda':
            print(f"GPU: {device_info['device_name']}")
            print(f"Memory: {device_info['memory_gb']:.2f} GB")
        print()

        try:
            segmentation_results = self.segmenter.segment_and_extract(
                preprocessed_image
            )

            if segmentation_results.get('masks'):
                self.segmenter.visualize_segmentation(
                    preprocessed_image,
                    segmentation_results['masks'],
                    output_path=str(self.output_dir / "02_segmentation.jpg")
                )

                for name, mask in segmentation_results.get('masks', {}).items():
                    cv2.imwrite(
                        str(self.output_dir / f"mask_{name}.jpg"),
                        mask
                    )

            self.results['segmentation'] = {
                'status': 'success',
                'masks_found': list(segmentation_results.get('masks', {}).keys()),
                'saved': str(self.output_dir / "02_segmentation.jpg")
            }

            print("✓ Stage 2 complete")
            return segmentation_results

        except Exception as e:
            print(f"✗ Stage 2 failed: {e}")
            self.results['segmentation'] = {'status': 'failed', 'error': str(e)}
            return None

    def stage_3_adaptive_calibration(self,
                                     preprocessed_image: np.ndarray,
                                     scale_mask: np.ndarray) -> Optional[Dict]:
        """Stage 3: Adaptive Calibration with Multiple Strategies"""
        self._print_header("STAGE 3: ADAPTIVE CALIBRATION")
        print("-" * 70)

        self._print_subheader("Multiple Detection Strategies")
        print("✓ Strategy 1: Contour-based marking detection")
        print("✓ Strategy 2: Projection-based peak detection")
        print("✓ Strategy 3: Manual threshold detection")
        print("✓ Automatic fallback between strategies")

        try:
            calibration_info = self.calibrator.auto_calibrate_advanced(
                preprocessed_image,
                scale_mask
            )

            if calibration_info is None:
                print("\n✗ Calibration failed!")
                return None

            self.results['calibration'] = calibration_info
            return calibration_info

        except Exception as e:
            print(f"\n✗ Stage 3 failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['calibration'] = {'status': 'failed', 'error': str(e)}
            return None

    def stage_4_measurement(self,
                           preprocessed_image: np.ndarray,
                           segmentation_results: Dict) -> Optional[Dict]:
        """Stage 4: Precision Measurement"""
        self._print_header("STAGE 4: PRECISION MEASUREMENT")
        print("-" * 70)

        try:
            concrete_mask = segmentation_results['masks'].get('concrete_block')
            if concrete_mask is None:
                print("⚠ Concrete block not detected")
                return None

            print("\n[Measurement] Computing block dimensions...")
            measurements = self.calibrator.measure_concrete_block_with_uncertainty(
                concrete_mask,
                preprocessed_image
            )

            if measurements is None:
                print("⚠ Measurement failed")
                return None

            print("\n[Visualization] Creating analysis visualization...")
            vis_image = self.calibrator.create_measurement_visualization(
                preprocessed_image,
                concrete_mask,
                output_path=str(self.output_dir / "03_final_analysis.jpg")
            )

            self.results['measurements'] = measurements
            print("\n✓ Stage 4 complete")
            return measurements

        except Exception as e:
            print(f"✗ Stage 4 failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['measurements'] = {'status': 'failed', 'error': str(e)}
            return None

    def stage_5_reporting(self) -> str:
        """Stage 5: Generate Report"""
        self._print_header("STAGE 5: COMPREHENSIVE REPORTING")
        print("-" * 70)

        try:
            report = self.calibrator.generate_advanced_report(
                output_path=str(self.output_dir / "analysis_report.txt")
            )

            print("\n" + report)

            json_path = str(self.output_dir / "results.json")
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)

            print(f"\n✓ JSON results saved to {json_path}")

            self._print_subheader("ANALYSIS SUMMARY")
            if self.results.get('calibration', {}).get('status') != 'failed':
                calib = self.results['calibration']
                print(f"Pixel/CM: {calib.get('pixel_per_cm', 0):.2f}")
                print(f"Markings: {calib.get('num_markings', 0)}")

            if self.results.get('measurements', {}).get('status') != 'failed':
                meas = self.results['measurements']
                print(f"\nBlock Area: {meas.get('area_cm2', 0):.2f} cm²")

            print("\n✓ Stage 5 complete")
            return report

        except Exception as e:
            print(f"✗ Stage 5 failed: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def run_pipeline(self, image_path: str) -> bool:
        """Run complete pipeline"""
        
        self._print_header("CONCRETE BLOCK ANALYSIS - ADAPTIVE CALIBRATION v3")
        print("Handles various ruler/scale configurations automatically")
        print(f"\nInput image: {image_path}")
        print(f"Output directory: {self.output_dir.absolute()}")

        # Stage 1
        preprocessed_image = self.stage_1_preprocessing(image_path)
        if preprocessed_image is None:
            return False

        # Stage 2
        segmentation_results = self.stage_2_segmentation(preprocessed_image)
        if segmentation_results is None:
            return False

        # Stage 3
        scale_mask = segmentation_results.get('masks', {}).get('scale')
        if scale_mask is None:
            print("✗ No scale detected!")
            return False

        calibration_info = self.stage_3_adaptive_calibration(
            preprocessed_image,
            scale_mask
        )

        if calibration_info is None:
            return False

        # Stage 4
        measurements = self.stage_4_measurement(preprocessed_image, segmentation_results)
        if measurements is None:
            return False

        # Stage 5
        report = self.stage_5_reporting()

        # Final
        self._print_header("PIPELINE COMPLETE")
        print(f"\nAll outputs saved to: {self.output_dir.absolute()}")
        print(f"\nGenerated files:")
        for file in sorted(self.output_dir.glob("*")):
            print(f" ✓ {file.name}")
        print("\n" + "="*70)

        return True

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Concrete Block Analysis - Adaptive Calibration"
    )

    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    pipeline = ConcreteAnalysisPipeline(output_dir=args.output_dir)
    success = pipeline.run_pipeline(image_path=args.image)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print("Concrete Block Analysis - Adaptive Calibration v3")
        print("="*60)
        print()

        image_path = input("Enter image path: ").strip()
        if not os.path.exists(image_path):
            print(f"Error: File not found")
            sys.exit(1)

        output_dir = input("Enter output directory [output]: ").strip() or "output"

        pipeline = ConcreteAnalysisPipeline(output_dir=output_dir)
        success = pipeline.run_pipeline(image_path=image_path)

        sys.exit(0 if success else 1)