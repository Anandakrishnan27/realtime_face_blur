import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for better face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through the detected faces and blur them
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) where the face is detected
        face_region = frame[y:y+h, x:x+w]

        # Apply a Gaussian blur to the face region
        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)

        # Replace the face region with the blurred face in the frame
        frame[y:y+h, x:x+w] = blurred_face

    # Display the resulting frame
    cv2.imshow('Face Blur', frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
