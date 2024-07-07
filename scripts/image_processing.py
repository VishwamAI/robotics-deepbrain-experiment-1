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

def generate_hologram(image, position, intensity):
    """
    Generate a 2D imagination hologram based on the model output.

    Args:
    image (np.ndarray): Input image.
    position (np.ndarray): Position array from the neural network model.
    intensity (np.ndarray): Intensity array from the neural network model.

    Returns:
    np.ndarray: Generated hologram image.
    """
    # Create a blank canvas for the hologram
    hologram = np.zeros_like(image)

    # Define the center position for the hologram
    center_x = int(position[0] * image.shape[1])
    center_y = int(position[1] * image.shape[0])

    # Define the radius and intensity for the hologram
    radius = int(0.1 * min(image.shape[:2]))  # Assuming a fixed radius for simplicity
    intensity = np.clip(intensity, 0, 1)

    # Draw the hologram on the canvas
    cv2.circle(hologram, (center_x, center_y), radius, (intensity, intensity, intensity), -1)

    # Combine the hologram with the original image
    combined_image = cv2.addWeighted(image, 0.5, hologram, 0.5, 0)

    return combined_image

if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/image.jpg"
    image = load_image(image_path)
    preprocessed_image = preprocess_image(image)
    display_image(preprocessed_image)

    # Example model output (random values for demonstration)
    position = np.random.rand(2)
    intensity = np.random.rand()

    # Generate and display the hologram
    hologram = generate_hologram(preprocessed_image, position, intensity)
    display_image(hologram, window_name="Hologram")
