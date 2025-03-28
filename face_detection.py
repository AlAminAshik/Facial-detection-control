#Always save the file before running the code otherwise the code will not load the changes made.
import cv2  # OpenCV library

# Load the pre-trained face detection model https://github.com/opencv/opencv/tree/3.4/data
# store the model in the variable face_cascade
# haarcascade_frontalface_default.xml is a pre-trained model for detecting faces in an image.
# haarcascade_eye.xml is a pre-trained model for detecting eyes in an image.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start video capture from the default webcam (0)
capture = cv2.VideoCapture(0)

while True:
    # Read frame from the webcam
    #If the webcam is working fine: ret = True, and frame contains the image data.
    #If the webcam is disconnected: ret = False, and the loop exits gracefully.
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1) # Flip the frame horizontally
    if not ret:
     break
    
    # Convert frame to grayscale for better detection
    # cvtColor() function is used to convert an image from one color space to another.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces and eyes in the frame
    # detectMultiScale() function detects objects in an image.
    # scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
    # minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    # minSize: Minimum possible object size. Objects smaller than this are ignored.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))  #Lower minNeighbors value will result in more detections but with lower quality
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7, minSize=(30, 30))    #higher minNeighbors value will result in less detections but with higher quality

    # Draw bounding boxes around detected faces
    # rectangle() function is used to draw a rectangle on an image.
    # The first argument is the image, the second argument is the top-left corner of the rectangle, the third argument is the bottom-right corner of the rectangle, the fourth argument is the color of the rectangle, and the fifth argument is the thickness of the rectangle.
    # The for loop iterates over the faces array and draws a rectangle around each face.
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 250), 3)
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 250), 3)

    #Display is face detected
    if len(faces) > 0:
        cv2.putText(frame, 'Face Detected', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250), 2)
        #Display face count
        cv2.putText(frame, 'Number of faces: ' + str(len(faces)), (30, 60), cv2.FONT_ITALIC, 1, (0, 0, 250), 2)
    else:
        cv2.putText(frame, 'No Face Detected', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250), 2)

    #Display is eye detected
    if len(eyes) > 0:
        #Display eye count
        cv2.putText(frame, 'Number of eyes: ' + str(len(eyes)), (30, 90), cv2.FONT_ITALIC, 1, (0, 0, 250), 2)    

    # Display the output (should be at the end after bounding box and text)
    
    cv2.imshow('Face Detection', frame)
    
    # Break the loop if 'q' is pressed
    # ord() function returns an integer representing the Unicode code point of the character.
    # waitKey(1) returns -1 when no key is pressed.
    # if q is pressed waitkey(1) returns 113 which is equal to ord('q').
    if cv2.waitKey(1) & 0xFF == ord('q'): # 0xFF is a hexadecimal constant which is 11111111 in binary.
        break

# Release resources
capture.release()   #release the webcam
cv2.destroyAllWindows() #close the window
