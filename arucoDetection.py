import numpy as np
import time
import cv2

# Print the list of available functions in the aruco module and the OpenCV version
#print(dir(cv2.aruco))
#print(cv2.__version__)

# Dictionary mapping ArUco marker types to their corresponding OpenCV constants
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

'''# Function to check available cameras
def check_cameras(max_index=10):
    index = 0
    while index < max_index:  # Set a maximum index limit to avoid an infinite loop
        cap = cv2.VideoCapture(index)
        if cap.isOpened():  # Check if the camera at this index can be opened
            print(f"Camera with index {index} is available.")
            cap.release()
        else:
            print(f"Camera with index {index} is not available.")
        index += 1

# Run the camera check for a range of possible indices
check_cameras(20)  # Adjust the number to a higher value if needed
'''

# This ends here

# Function to display detected ArUco markers on the image
def aruco_display(corners, ids, rejected, image):
    # Check if there are any detected markers
    if len(corners) > 0:
        
        ids = ids.flatten()  # Flatten the list of detected marker IDs
        
        # Loop through each detected marker's corners and its ID
        for (markerCorner, markerID) in zip(corners, ids):
            
            corners = markerCorner.reshape((4, 2))  # Reshape the corners into (4, 2) array
            (topLeft, topRight, bottomRight, bottomLeft) = corners  # Unpack the corner points
            
            # Convert the corner points to integers for drawing purposes
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # Draw the bounding box around the marker
            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            
            # Calculate the center of the marker and draw a circle there
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            
            # Display the marker ID on the image
            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            print("[Inference] ArUco marker ID: {}".format(markerID))
            
    return image

# Specify the type of ArUco dictionary being used
aruco_type = "DICT_5X5_250"

# Load the predefined dictionary for ArUco markers
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

# Create parameters for the ArUco detector
arucoParams = cv2.aruco.DetectorParameters()

# Initialize video capture with the first available camera (index 0)
cap = cv2.VideoCapture(1)

# Set the desired frame width and height for the video feed
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Main loop for capturing and processing video frames
while cap.isOpened():
    
    ret, img = cap.read()  # Capture a frame from the video feed

    if not ret:
        print("Error: Failed to capture image from the camera.")
        break

    #h, w, _ = img.shape  # Get the height and width of the captured frame

    # Resize the image to a fixed width while maintaining the aspect ratio
    #width = 1000
    #height = int(width * (h / w))
    #img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
 
    # Detect ArUco markers in the frame
    corners, ids, rejected = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)

    # Display the detected markers on the image
    detected_markers = aruco_display(corners, ids, rejected, img)

    # Show the image with the detected markers
    cv2.imshow("Image", detected_markers)

    # Break the loop if the 'q' key is pressed
    cv2.waitKey(1) 

# Release resources and close any open windows
cv2.destroyAllWindows()
cap.release()


'''
# Detecting Aruco Markers with saved local images
import numpy as np
import cv2 as cv
import os

# Dictionary mapping ArUco marker types to their corresponding OpenCV constants
ARUCO_DICT = {
    "DICT_4X4_50": cv.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11
}

# Function to display detected ArUco markers on the image
def aruco_display(corners, ids, rejected, image):
    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # Turning corners to points
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # Bounding box gang
            cv.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            # Find center of ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            # Draw the ArUco marker ID on the image
            cv.putText(image, f"Marker ID: {markerID}", (topLeft[0], topLeft[1] + 20), cv.FONT_HERSHEY_SIMPLEX,
            0.4, (0, 150, 255), 2)  # Adjusted position, size, color, and thickness
            print("[Inference] ArUco marker ID: {}".format(markerID))


    return image


aruco_type = "DICT_5X5_250"
arucoDict = cv.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
arucoParams = cv.aruco.DetectorParameters()

image_path = "arucomarker1.png"  # image name in my directory

if not os.path.exists(image_path):
    print("Error: The specified image path does not exist.")
    exit()

image = cv.imread(image_path)

if image is None:
    print("Error: Could not load image from file. Check the path.")
    exit()

corners, ids, rejected = cv.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

detected_markers = aruco_display(corners, ids, rejected, image)

cv.imshow("Image", detected_markers)
cv.waitKey(0)

cv.destroyAllWindows()
'''
