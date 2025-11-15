"""
DEBUGGING SCRIPT - Check what's happening with marking detection

This script helps visualize what markings are being detected
"""

import cv2
import numpy as np
from pathlib import Path

def debug_calibration(image_path: str, scale_mask_path: str, output_dir: str = "debug_output"):
    """Debug calibration by visualizing marking detection"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load image and mask
    image = cv2.imread(image_path)
    scale_mask = cv2.imread(scale_mask_path, cv2.IMREAD_GRAYSCALE)
    
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {scale_mask.shape}")
    
    # Extract scale region
    scale_region = cv2.bitwise_and(image, image, mask=scale_mask)
    
    gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Simple edge detection
    gray_blur = cv2.medianBlur(gray, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    gray_morph = cv2.morphologyEx(gray_blur, cv2.MORPH_CLOSE, kernel)
    
    sobelx = cv2.Sobel(gray_morph, cv2.CV_64F, 1, 0, ksize=3)
    edges = np.abs(sobelx).astype(np.uint8)
    _, edges = cv2.threshold(edges, 80, 255, cv2.THRESH_BINARY)
    
    # Save edges
    cv2.imwrite(str(output_dir / "01_edges.jpg"), edges)
    print(f"✓ Saved edges to debug_output/01_edges.jpg")
    
    # Line detection
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=40,
        minLineLength=50,
        maxLineGap=2
    )
    
    print(f"\n[Lines Detection]")
    print(f"Total lines: {len(lines) if lines is not None else 0}")
    
    if lines is not None:
        # Filter vertical lines
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dy = abs(y2 - y1)
            dx = abs(x2 - x1)
            
            if dy > dx * 3:
                vertical_lines.append((x1, y1, x2, y2))
        
        print(f"Vertical lines: {len(vertical_lines)}")
        
        # Visualization
        vis_all = scale_region.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(vis_all, (x1, y1), (x2, y2), (0, 0, 255), 1)
        
        cv2.imwrite(str(output_dir / "02_all_lines.jpg"), vis_all)
        print(f"✓ Saved all lines visualization to debug_output/02_all_lines.jpg")
        
        # Visualization - vertical only
        vis_vertical = scale_region.copy()
        for x1, y1, x2, y2 in vertical_lines:
            cv2.line(vis_vertical, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imwrite(str(output_dir / "03_vertical_lines.jpg"), vis_vertical)
        print(f"✓ Saved vertical lines visualization to debug_output/03_vertical_lines.jpg")
        
        # Extract positions
        positions = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2 in vertical_lines]
        positions = sorted(positions, key=lambda p: p[0])
        
        # Calculate spacings
        if len(positions) >= 2:
            spacings = []
            for i in range(len(positions) - 1):
                spacing = positions[i+1][0] - positions[i][0]
                spacings.append(spacing)
            
            spacings = np.array(spacings)
            
            print(f"\n[Spacings Analysis]")
            print(f"Min spacing: {spacings.min():.2f} px")
            print(f"Max spacing: {spacings.max():.2f} px")
            print(f"Mean spacing: {spacings.mean():.2f} px")
            print(f"Std dev: {spacings.std():.2f} px")
            
            # Calculate calibration
            # Standard ruler: 2mm between markings
            pixel_per_cm = spacings.mean() / 0.2
            pixel_per_mm = pixel_per_cm / 10
            
            print(f"\n[Calibration]")
            print(f"Mean spacing: {spacings.mean():.2f} px")
            print(f"Pixel/CM: {pixel_per_cm:.2f}")
            print(f"Pixel/MM: {pixel_per_mm:.6f}")
            
            # Check against expected
            if pixel_per_mm > 0:
                print(f"\n[Expected vs Actual]")
                print(f"Expected pixel/CM: ~420.17")
                print(f"Expected pixel/MM: ~42.017")
                print(f"Your pixel/MM: {pixel_per_mm:.2f}")
                print(f"Ratio (actual/expected): {pixel_per_mm / 42.017:.2f}x")
    
    print("\n" + "="*70)
    print("Debug completed. Check debug_output/ folder for visualizations")
    print("="*70)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python debug_calibration.py <image.jpg> <scale_mask.jpg>")
        print("\nExample:")
        print("  python debug_calibration.py output/01_preprocessed.jpg output/mask_scale.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    mask_path = sys.argv[2]
    
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    if not Path(mask_path).exists():
        print(f"Error: Mask not found: {mask_path}")
        sys.exit(1)
    
    debug_calibration(image_path, mask_path)