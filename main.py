"""Main Pipeline Script
Orchestrates the complete concrete block analysis pipeline
"""

import cv2
import os
import sys
from pathlib import Path

# Import custom modules
try:
    from image_preprocessor import ImagePreprocessor
    from segmentation_module import SegmentationModule
    from calibration_and_measurement import CalibrationMeasurement
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
    
    def run_pipeline(self, image_path: str, scale_length_mm: float = None):
        """Run the complete analysis pipeline
        
        Args:
            image_path: Path to input image
            scale_length_mm: Actual physical length of scale in mm
        """
        print("\n" + "="*60)
        print("CONCRETE BLOCK ANALYSIS PIPELINE")
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
            print("4")
            self.segmentation_results = self.segmenter.segment_and_extract(
                self.preprocessed_image
            )
            print("3")
            # Save segmentation visualization
            if self.segmentation_results.get('masks'):
                self.segmenter.visualize_segmentation(
                    self.preprocessed_image,
                    self.segmentation_results['masks'],
                    output_path=str(self.output_dir / "02_segmentation.jpg")
                )
            print("2")
            # Save individual masks
            for name, mask in self.segmentation_results.get('masks', {}).items():
                cv2.imwrite(str(self.output_dir / f"mask_{name}.jpg"), mask)
            
            print("✓ Stage 2 complete\n")
        except Exception as e:
            print(f"✗ Stage 2 failed: {e}")
            return
        
        # Stage 3: Calibration and Measurement
        print("[STAGE 3] Calibration and Measurement...")
        print("-" * 40)
        try:
            # Get user input for scale length if not provided
            if scale_length_mm is None:
                print("\nPlease enter the actual physical length of the scale:")
                scale_length_mm = float(input("Scale length (in mm): "))
            
            # Check if we have scale boundaries
            if 'scale_boundaries' not in self.segmentation_results:
                print("✗ No scale detected in image. Cannot perform calibration.")
                return
            
            # Calculate calibration
            self.calibrator.calculate_pixel_ratio(
                self.segmentation_results['scale_boundaries'],
                scale_length_mm
            )
            
            # Measure concrete block
            if 'concrete_boundaries' in self.segmentation_results:
                self.calibrator.measure_concrete_block(
                    self.segmentation_results['concrete_boundaries'],
                    self.segmentation_results['masks']['concrete_block']
                )
            
            # Analyze phenophthalein coverage
            if 'concrete_block' in self.segmentation_results['masks']:
                area_analysis = self.calibrator.get_affected_area(
                    self.preprocessed_image,
                    self.segmentation_results['masks']['concrete_block']
                )
                
                # Create visualization
                self.calibrator.create_measurement_visualization(
                    self.preprocessed_image,
                    self.segmentation_results['masks']['concrete_block'],
                    area_analysis['magenta_mask'],
                    output_path=str(self.output_dir / "03_final_analysis.jpg")
                )
            
            # Generate report
            report = self.calibrator.generate_report(
                output_path=str(self.output_dir / "analysis_report.txt")
            )
            
            print("\n" + report)
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
        description="Concrete Block Analysis - Phenophthalein Detection System"
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to input image containing concrete block and scale"
    )
    parser.add_argument(
        "--scale-length",
        type=float,
        default=None,
        help="Actual physical length of scale in millimeters"
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
    print(args.image)
    # Create and run pipeline
    pipeline = ConcreteAnalysisPipeline(output_dir=args.output_dir)
    pipeline.run_pipeline(
        image_path=args.image,
        scale_length_mm=args.scale_length
    )


if __name__ == "__main__":
    # Example usage without command line arguments
    print("Concrete Block Analysis Pipeline")
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