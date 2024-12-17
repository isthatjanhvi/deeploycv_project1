import cv2
import numpy as np
import matplotlib.pyplot as plt

def low_pass_filter(image, kernel_size=15):
    """Applies a Gaussian Blur to retain low frequencies."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def high_pass_filter(image, kernel_size=15):
    """Subtracts the low-frequency components to retain high frequencies."""
    low_pass = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return cv2.subtract(image, low_pass)

def hybrid_image(low_image, high_image, alpha=0.5, beta=0.5):
    """Combines the low-pass and high-pass filtered images."""
    return cv2.addWeighted(low_image, alpha, high_image, beta, 0)

# Load your own images
image1 = cv2.imread("image1.jpeg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("image2.jpeg", cv2.IMREAD_GRAYSCALE)

# Resize both images to the same size
image1 = cv2.resize(image1, (500, 500))
image2 = cv2.resize(image2, (500, 500))

# Apply filters
low_pass_img = low_pass_filter(image1)
high_pass_img = high_pass_filter(image2)

# Create hybrid image
hybrid_img = hybrid_image(low_pass_img, high_pass_img)

# Save results
cv2.imwrite("low_pass_image.jpg", low_pass_img)
cv2.imwrite("high_pass_image.jpg", high_pass_img)
cv2.imwrite("hybrid_image.jpg", hybrid_img)

# Plot and display the images
images = [image1, image2, low_pass_img, high_pass_img, hybrid_img]
titles = ["Original 1", "Original 2", "Low-Pass Image", "High-Pass Image", "Hybrid Image"]

fig, axes = plt.subplots(1, 5, figsize=(20, 8))
for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.savefig("combined_hybrid_image.jpg")  # Save the combined output
plt.show()
