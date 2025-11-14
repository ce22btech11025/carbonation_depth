"""Main Pipeline Script - WITH SUB-PIXEL PRECISION & GPU ACCELERATION

Complete concrete block analysis with:
- GPU-accelerated segmentation
- Sub-pixel calibration and measurements
- High-precision results

"""

import cv2
import os
import sys
from pathlib import Path

# Import custom modules
try:
    from image_preprocessor import ImagePreprocessor
    from segmentation_module import SegmentationModule
    from calibration_and_measurement_v2 import CalibrationMeasurement
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all module files are in the same directory.")
    sys.exit(1)


class ConcreteAnalysisPipeline:
    def __init__(self, output_dir: str = "output"):
        """Initialize the complete analysis pipeline

        Args:
            output_dir: Directory to save all output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize modules
        self.preprocessor = ImagePreprocessor()
        self.segmenter = SegmentationModule()  # Auto-detects GPU
        self.calibrator = CalibrationMeasurement()

        # Storage for pipeline data
        self.original_image = None
        self.preprocessed_image = None
        self.segmentation_results = None

    def run_pipeline(self, image_path: str):
        """Run complete analysis with sub-pixel precision and GPU acceleration

        Args:
            image_path: Path to input image
        """
        print("\n" + "="*60)
        print("CONCRETE BLOCK ANALYSIS PIPELINE")
        print("Sub-Pixel Precision + GPU Acceleration")
        print("="*60 + "\n")

        # Print device info
        device_info = self.segmenter.get_device_info()
        print(f"🖥️  Computing Device: {device_info['device'].upper()}")
        if device_info['device'] == 'cuda':
            print(f"   GPU: {device_info['device_name']}")
            print(f"   Memory: {device_info['memory_gb']:.2f} GB")
        print()

        # Stage 1: Preprocessing
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

        # Stage 2: GPU-Accelerated Segmentation
        print("[STAGE 2] GPU-Accelerated Segmentation...")
        print("-" * 40)
        try:
            self.segmentation_results = self.segmenter.segment_and_extract(
                self.preprocessed_image
            )

            # Save segmentation visualization
            if self.segmentation_results.get('masks'):
                self.segmenter.visualize_segmentation(
                    self.preprocessed_image,
                    self.segmentation_results['masks'],
                    output_path=str(self.output_dir / "02_segmentation.jpg")
                )

            # Save individual masks
            for name, mask in self.segmentation_results.get('masks', {}).items():
                cv2.imwrite(str(self.output_dir / f"mask_{name}.jpg"), mask)

            print("✓ Stage 2 complete\n")
        except Exception as e:
            print(f"✗ Stage 2 failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # Stage 3: Sub-Pixel Calibration and Measurement
        print("[STAGE 3] Sub-Pixel Calibration & Measurement...")
        print("-" * 40)
        try:
            # Check if we have scale mask
            scale_mask = self.segmentation_results.get('masks', {}).get('scale')
            if scale_mask is None:
                print("✗ No scale detected. Cannot calibrate.")
                return

            # Automatic sub-pixel calibration
            print("\n>>> STARTING SUB-PIXEL CALIBRATION <<<")
            calibration_info = self.calibrator.auto_calibrate_from_scale(
                self.preprocessed_image,
                scale_mask
            )

            if calibration_info is None:
                print("\n✗ Calibration failed!")
                return

            print(f"\n>>> CALIBRATION SUCCESSFUL <<<")
            print(f"Method: {calibration_info['detection_method']}")
            print(f"Precision: ±{calibration_info['std_deviation']:.4f} pixels")
            print("="*60)

            # Measure concrete block with sub-pixel precision
            concrete_mask = self.segmentation_results['masks'].get('concrete_block')
            if concrete_mask is not None and 'concrete_boundaries' in self.segmentation_results:
                # Pass original image for sub-pixel refinement
                measurements = self.calibrator.measure_concrete_block(
                    self.segmentation_results['concrete_boundaries'],
                    concrete_mask,
                    self.preprocessed_image  # Added for sub-pixel refinement
                )

                # Analyze phenophthalein coverage
                area_analysis = self.calibrator.get_affected_area(
                    self.preprocessed_image,
                    concrete_mask
                )

                # Create visualization
                self.calibrator.create_measurement_visualization(
                    self.preprocessed_image,
                    concrete_mask,
                    area_analysis['magenta_mask'],
                    output_path=str(self.output_dir / "03_final_analysis.jpg")
                )

                # Generate report
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
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
        print(f"\nOutputs saved to: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        for f in sorted(self.output_dir.glob("*")):
            print(f"  - {f.name}")
        print()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Concrete Analysis - Sub-Pixel Precision + GPU Acceleration"
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to input image with concrete block and scale"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )

    args = parser.parse_args()

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # Create and run pipeline
    pipeline = ConcreteAnalysisPipeline(output_dir=args.output_dir)
    pipeline.run_pipeline(image_path=args.image)


if __name__ == "__main__":
    # Example usage
    print("Concrete Block Analysis Pipeline")
    print("Sub-Pixel Precision + GPU Acceleration")
    print("=" * 50)

    # Check if running with arguments
    if len(sys.argv) > 1:
        main()
    else:
        # Interactive mode
        print("\nInteractive Mode")
        print("-" * 50)
        image_path = input("Enter path to image: ").strip()

        if not os.path.exists(image_path):
            print(f"Error: File not found: {image_path}")
            sys.exit(1)

        pipeline = ConcreteAnalysisPipeline(output_dir="output")
        pipeline.run_pipeline(image_path=image_path)

        print("\nDone! Check 'output' folder for results.")