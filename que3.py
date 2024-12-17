from PIL import Image, ImageFilter
import numpy as np

def identify_flag(image_path):
    """Identifies if an image is the flag of Indonesia or Poland."""
    try:
        # Load the image
        img = Image.open(image_path).convert('RGB')
        img = img.filter(ImageFilter.GaussianBlur(radius=5))  # Apply smoothing
        img_array = np.array(img)

        # Debug: Check if the image loaded correctly
        print(f"Image loaded successfully. Shape: {img_array.shape}")

        # Get image dimensions
        height, width, _ = img_array.shape

        # Focus on the central region (50% of the width)
        central_start = width // 4
        central_end = 3 * width // 4

        # Divide the image into two central halves
        top_half = img_array[:height // 2, central_start:central_end, :]
        bottom_half = img_array[height // 2:, central_start:central_end, :]

        # Calculate the average color for each half
        top_avg_color = np.mean(top_half, axis=(0, 1))
        bottom_avg_color = np.mean(bottom_half, axis=(0, 1))

        # Debug: Print average colors
        print(f"Top half average color: {top_avg_color}")
        print(f"Bottom half average color: {bottom_avg_color}")

        # Check the flag pattern
        if is_red(top_avg_color) and is_white(bottom_avg_color):
            return "This is the flag of Indonesia."
        elif is_white(top_avg_color) and is_red(bottom_avg_color):
            return "This is the flag of Poland."
        else:
            return "This is neither the flag of Indonesia nor Poland."
    except Exception as e:
        return f"Error processing the image: {e}"

def is_red(color):
    """Check if the color is predominantly red."""
    r, g, b = color
    return r > 150 and g < 100 and b < 100  # Red threshold adjusted for flags with waves

def is_white(color):
    """Check if the color is predominantly white."""
    r, g, b = color
    return r > 200 and g > 200 and b > 200  # White threshold

# Test the function
image_path = "image3.jpg"  # Replace with the path to your image
result = identify_flag(image_path)
print(result)
