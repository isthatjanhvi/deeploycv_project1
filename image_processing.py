import cv2
import numpy as np
import matplotlib.pyplot as plt

# Capture an image
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Unable to capture image.")
    return frame

# Grayscale conversion
def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Thresholding (black and white)
def threshold_image(image, threshold=128):
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary

# Reduce to 16 gray levels
def reduce_gray_levels(image):
    return (image // 16) * 16

# Sobel filter
def sobel_filter(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.magnitude(sobel_x, sobel_y)

# Canny edge detection
def canny_edge(image):
    return cv2.Canny(image, 100, 200)

# Gaussian filter
def apply_gaussian(image):
    kernel = (1/16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    return cv2.filter2D(image, -1, kernel)

# Sharpen image
def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Convert RGB to BGR
def convert_rgb_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Main code
if __name__ == "__main__":
    original = capture_image()
    grayscale = to_grayscale(original)
    thresholded = threshold_image(grayscale)
    reduced_gray = reduce_gray_levels(grayscale)
    sobel = sobel_filter(grayscale)
    canny = canny_edge(grayscale)
    gaussian = apply_gaussian(grayscale)
    sharpened = sharpen_image(gaussian)
    bgr = convert_rgb_to_bgr(original)

    # Display in a grid
    images = [grayscale, thresholded, reduced_gray, sobel, canny, gaussian, sharpened, bgr]
    titles = ["Grayscale", "Thresholded", "16 Grays", "Sobel", "Canny", "Gaussian", "Sharpened", "BGR"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))

    for ax, img, title in zip(axes.ravel(), images, titles):
        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("combined_image.jpg")
    plt.show()
