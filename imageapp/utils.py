# imageapp/utils.py
import cv2

def validate_face_alignment(image_path1, image_path2, tolerance_x = 60, tolerance_y = 60,tolerance_wh = 100):
    print('Inside validate_face_alignment')
    # Load images using OpenCV
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    if img1 is None or img2 is None:
        print("One of the images could not be loaded.")
        return False

    # Use Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces1 = face_cascade.detectMultiScale(img1, scaleFactor=1.1, minNeighbors=4)
    faces2 = face_cascade.detectMultiScale(img2, scaleFactor=1.1, minNeighbors=4)
    
    if len(faces1) == 0 or len(faces2) == 0:
        print("No faces detected. faces1:", faces1, "faces2:", faces2)
        return False
    
    # Use the first detected face in each image
    (x1, y1, w1, h1) = faces1[0]
    (x2, y2, w2, h2) = faces2[0]
    print(f"Image1 face: x={x1}, y={y1}, w={w1}, h={h1}")
    print(f"Image2 face: x={x2}, y={y2}, w={w2}, h={h2}")
    
    # Calculate differences for each coordinate/dimension
    diff_x = abs(x1 - x2)
    diff_y = abs(y1 - y2)
    diff_w = abs(w1 - w2)
    diff_h = abs(h1 - h2)
    print(f"diff_x: {diff_x}, diff_y: {diff_y}, diff_w: {diff_w}, diff_h: {diff_h}")
    
    # Check if each difference is within the tolerance (2 pixels)
    if diff_x > tolerance_x or diff_y > tolerance_y or diff_w > tolerance_wh or diff_h > tolerance_wh:
        print("Differences exceed tolerance")
        return False

    print("Face alignment within tolerance")
    return True
