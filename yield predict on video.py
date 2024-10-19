import cv2
import numpy as np

def detect_green_percentage(frame):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

    # Total number of pixels in the frame
    total_pixels = frame.shape[0] * frame.shape[1]

    # Calculate the percentage of green color
    green_percentage = (green_pixels / total_pixels) * 100

    return green_percentage

def predict_yield(green_percentage):
    # Define yield prediction model parameters
    a = 35  # kg/ha increase for each percentage of green cover
    b = 500  # kg/ha base yield

    # Calculate predicted yield
    predicted_yield = a * green_percentage + b
    return predicted_yield

def process_video(input_video_path, output_video_path):
    # Capture the video
    cap = cv2.VideoCapture(input_video_path)
    
    # Get the video frame width, height, and frames per second (FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for output video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect green percentage and predict yield
        green_percentage = detect_green_percentage(frame)
        predicted_yield = predict_yield(green_percentage)

        # Overlay the results on the frame
        cv2.putText(frame, f"Green Percentage: {green_percentage:.2f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Predicted Yield: {predicted_yield:.2f} kg/ha", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame)

        # Display the frame (optional)
        cv2.imshow('Video Processing', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
input_video_path = 'farmvid.mp4'
output_video_path = 'yeild.avi'
process_video(input_video_path, output_video_path)
