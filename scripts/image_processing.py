import cv2
import numpy as np

def load_image(image_path):
    """
    Load an image from the specified file path.

    Args:
    image_path (str): Path to the image file.

    Returns:
    np.ndarray: Loaded image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the image for model input.

    Args:
    image (np.ndarray): Input image.
    target_size (tuple): Target size for the image (width, height).

    Returns:
    np.ndarray: Preprocessed image.
    """
    # Resize the image to the target size
    image = cv2.resize(image, target_size)
    # Normalize the image to the range [0, 1]
    image = image / 255.0
    return image

def display_image(image, window_name="Image"):
    """
    Display the image in a window.

    Args:
    image (np.ndarray): Image to display.
    window_name (str): Name of the display window.
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/image.jpg"
    image = load_image(image_path)
    preprocessed_image = preprocess_image(image)
    display_image(preprocessed_image)
