#Always save the file before running the code otherwise the code will not load the changes made.
import cv2  # OpenCV library

# Load the pre-trained face detection model https://github.com/opencv/opencv/tree/3.4/data
# store the model in the variable face_cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from the default webcam (0)
capture = cv2.VideoCapture(0)

while True:
    # Read frame from the webcam
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1) # Flip the frame horizontally
    if not ret:
     break
    
    # Convert frame to grayscale for better detection
    # cvtColor() function is used to convert an image from one color space to another.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces and eyes in the frame
    # detectMultiScale() function detects objects in an image.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))  #Lower minNeighbors value will result in more detections but with lower quality
 
    # Draw bounding boxes around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 250), 3)
    
    #Display is face detected
    if len(faces) > 0:
        cv2.putText(frame, 'Face Detected', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250), 2)
        #Display face count
        cv2.putText(frame, 'Number of faces: ' + str(len(faces)), (30, 60), cv2.FONT_ITALIC, 1, (0, 0, 250), 2)
    else:
        cv2.putText(frame, 'No Face Detected', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250), 2)

    # Display the output (should be at the end after bounding box and text)
    
    cv2.imshow('Face Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'): # 0xFF is a hexadecimal constant which is 11111111 in binary.
        break

# Release resources
capture.release()   #release the webcam
cv2.destroyAllWindows() #close the window
