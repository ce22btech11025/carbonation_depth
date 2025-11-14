"""Image Preprocessor Module

Handles image loading, shadow detection and removal while preserving phenophthalein coloration

"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ImagePreprocessor:
    def __init__(self, blur_kernel: Tuple[int, int] = (5, 5)):
        """Initialize the preprocessor with configuration

        Args:
            blur_kernel: Gaussian blur kernel size for noise reduction
        """
        self.blur_kernel = blur_kernel
        self.original_image = None
        self.preprocessed_image = None

    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path

        Args:
            image_path: Path to the input image

        Returns:
            Loaded image as numpy array
        """
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Failed to load image from {image_path}")

        print(f"✓ Image loaded: {self.original_image.shape}")
        return self.original_image.copy()

    def detect_shadows(self, image: np.ndarray) -> np.ndarray:
        """Detect shadow regions using HSV color space

        Args:
            image: Input BGR image

        Returns:
            Binary mask where shadows are marked as 255
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Shadow detection: low value and low saturation
        lower_shadow = np.array([0, 0, 0])
        upper_shadow = np.array([180, 80, 100])
        shadow_mask = cv2.inRange(hsv, lower_shadow, upper_shadow)

        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)

        print(f"✓ Shadows detected: {np.sum(shadow_mask > 0)} pixels")
        return shadow_mask

    def remove_shadows(self, image: np.ndarray, shadow_mask: np.ndarray) -> np.ndarray:
        """Remove shadows while preserving original colors

        Args:
            image: Input BGR image
            shadow_mask: Binary mask of shadow regions

        Returns:
            Image with shadows removed
        """
        result = image.copy()

        # Convert to LAB color space for illumination correction
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Increase L channel in shadow areas
        shadow_regions = shadow_mask > 0
        l_channel[shadow_regions] = np.clip(l_channel[shadow_regions] * 1.5, 0, 255).astype(np.uint8)

        # Merge channels back
        corrected_lab = cv2.merge([l_channel, a_channel, b_channel])
        result = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

        # Preserve magenta phenophthalein regions
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_magenta = np.array([140, 50, 50])
        upper_magenta = np.array([170, 255, 255])
        magenta_mask = cv2.inRange(hsv, lower_magenta, upper_magenta)

        # Restore original magenta regions
        result[magenta_mask > 0] = image[magenta_mask > 0]

        print("✓ Shadows removed, magenta coloration preserved")
        return result

    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur for noise reduction

        Args:
            image: Input image

        Returns:
            Blurred image
        """
        return cv2.GaussianBlur(image, self.blur_kernel, 0)

    def preprocess(self, image_path: str, save_path: Optional[str] = None) -> np.ndarray:
        """Complete preprocessing pipeline

        Args:
            image_path: Path to input image
            save_path: Optional path to save preprocessed image

        Returns:
            Preprocessed image
        """
        # Load image
        image = self.load_image(image_path)

        # Apply Gaussian blur
        blurred = self.apply_gaussian_blur(image)

        # Detect shadows
        shadow_mask = self.detect_shadows(blurred)

        # Remove shadows
        self.preprocessed_image = self.remove_shadows(blurred, shadow_mask)

        # Save if requested
        if save_path:
            cv2.imwrite(save_path, self.preprocessed_image)
            print(f"✓ Preprocessed image saved to {save_path}")

        return self.preprocessed_image

    def save_preprocessed_image(self, output_path: str) -> None:
        """Save the preprocessed image

        Args:
            output_path: Path where to save the image
        """
        if self.preprocessed_image is None:
            raise ValueError("No preprocessed image available. Run preprocess() first.")

        cv2.imwrite(output_path, self.preprocessed_image)
        print(f"✓ Image saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    preprocessor = ImagePreprocessor()

    # Process image
    preprocessed = preprocessor.preprocess(
        image_path="input_image.jpg",
        save_path="preprocessed_output.jpg"
    )

    print("\n✓ Preprocessing complete!")