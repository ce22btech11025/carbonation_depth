"""Main Pipeline Script - WITH AUTOMATIC SCALE CALIBRATION

Orchestrates the complete concrete block analysis pipeline with automatic
yellow scale detection and calibration (no manual input required!)
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
        self.segmenter = SegmentationModule()
        self.calibrator = CalibrationMeasurement()
        
        # Storage for pipeline data
        self.original_image = None
        self.preprocessed_image = None
        self.segmentation_results = None
    
    def run_pipeline(self, image_path: str):
        """Run the complete analysis pipeline with AUTOMATIC calibration
        
        Args:
            image_path: Path to input image
        """
        print("\n" + "="*60)
        print("CONCRETE BLOCK ANALYSIS PIPELINE")
        print("WITH AUTOMATIC SCALE CALIBRATION")
        print("="*60 + "\n")
        
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
        
        # Stage 2: Segmentation
        print("[STAGE 2] Object Segmentation with SAM 2...")
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
        
        # Stage 3: AUTOMATIC Calibration and Measurement
        print("[STAGE 3] Automatic Calibration and Measurement...")
        print("-" * 40)
        try:
            # Check if we have scale mask
            scale_mask = self.segmentation_results.get('masks', {}).get('scale')
            
            if scale_mask is None:
                print("✗ No scale detected in image. Cannot perform calibration.")
                print("  Make sure yellow scale with black markings is visible.")
                return
            
            # ===== AUTOMATIC CALIBRATION (NO MANUAL INPUT!) =====
            print("\n>>> STARTING AUTOMATIC CALIBRATION <<<")
            calibration_info = self.calibrator.auto_calibrate_from_scale(
                self.preprocessed_image,
                scale_mask
            )
            
            if calibration_info is None:
                print("\n✗ Automatic calibration failed!")
                print("  Check if scale markings are visible and clear.")
                return
            
            print(f"\n>>> CALIBRATION SUCCESSFUL <<<")
            print(f"Scale detected: {calibration_info['scale_type']}")
            print(f"Ratio used: {calibration_info['pixel_per_cm']:.2f} pixels/cm")
            print("="*60)
            
            # Measure concrete block
            concrete_mask = self.segmentation_results['masks'].get('concrete_block')
            
            if concrete_mask is not None and 'concrete_boundaries' in self.segmentation_results:
                measurements = self.calibrator.measure_concrete_block(
                    self.segmentation_results['concrete_boundaries'],
                    concrete_mask
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
                print("\n⚠ Concrete block not detected or boundaries missing.")
            
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
        print(f"\nAll outputs saved to: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        for f in sorted(self.output_dir.glob("*")):
            print(f"  - {f.name}")
        print()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Concrete Block Analysis - Automatic Yellow Scale Detection"
    )
    
    parser.add_argument(
        "image",
        type=str,
        help="Path to input image containing concrete block and yellow scale"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save output files (default: output)"
    )
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    # Create and run pipeline
    pipeline = ConcreteAnalysisPipeline(output_dir=args.output_dir)
    pipeline.run_pipeline(image_path=args.image)


if __name__ == "__main__":
    # Example usage
    print("Concrete Block Analysis Pipeline")
    print("With Automatic Yellow Scale Detection!")
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
        
        print("\nDone! Check the 'output' folder for results.")