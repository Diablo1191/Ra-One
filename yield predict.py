import cv2
import numpy as np

def detect_green_percentage(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return None

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for the green color in HSV (extended range)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])

    # Create a mask for green color
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    # Count the number of green pixels
    green_pixels = cv2.countNonZero(green_mask)

    # Total number of pixels in the image
    total_pixels = image.shape[0] * image.shape[1]

    # Calculate the percentage of green color
    green_percentage = (green_pixels / total_pixels) * 100

    return green_percentage

def predict_yield(green_percentage):
    # Define yield prediction model parameters
    a = 35  # Adjust this value based on historical data (sensitivity)
    b = 500  # Base yield when green percentage is 0

    # Calculate predicted yield
    predicted_yield = a * green_percentage + b
    return predicted_yield

# Example usage
image_path = 'Screenshot 2024-10-19 121812.png'
green_percentage = detect_green_percentage(image_path)

if green_percentage is not None:
    predicted_yield = predict_yield(green_percentage)
    print(f"Green color percentage: {green_percentage:.2f}%")
    print(f"Predicted yield: {predicted_yield:.2f} kg/ha")  # Adjust unit as needed
