import cv2
import numpy as np

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

webcam = cv2.VideoCapture(0)

color_ranges = {
    'Red': [([0, 120, 70], [10, 255, 255]), ([170, 120, 70], [180, 255, 255])],
    'Green': [([36, 50, 70], [89, 255, 255])],
    'Blue': [([90, 60, 0], [130, 255, 255])],
    'Yellow': [([20, 100, 100], [30, 255, 255])],
    'Orange': [([10, 100, 20], [20, 255, 255])],
    'Purple': [([129, 50, 70], [158, 255, 255])],
    'Pink': [([160, 100, 100], [169, 255, 255])],
    'Brown': [([10, 100, 20], [20, 200, 200])],
    'White': [([0, 0, 200], [180, 25, 255])],
    'Black': [([0, 0, 0], [180, 255, 50])]
}

label_colors = {
    'Red': (0, 0, 255),
    'Green': (0, 255, 0),
    'Blue': (255, 0, 0),
    'Yellow': (0, 255, 255),
    'Orange': (0, 165, 255),
    'Purple': (255, 0, 255),
    'Pink': (255, 105, 180),
    'Brown': (19, 69, 139),
    'White': (255, 255, 255),
    'Black': (50, 50, 50)
}

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # Blur to remove noise
    frame_blur = cv2.GaussianBlur(frame, (7, 7), 0)

    # Detect and mask face
    gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    mask_face = np.zeros(frame.shape[:2], dtype=np.uint8)
    for (x, y, w, h) in faces:
        cv2.rectangle(mask_face, (x, y), (x + w, y + h), 255, -1)

    hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

    for color_name, ranges in color_ranges.items():
        total_mask = None
        for lower, upper in ranges:
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(mask_face))  # Remove face area
            total_mask = mask if total_mask is None else cv2.bitwise_or(total_mask, mask)

        kernel = np.ones((5, 5), np.uint8)
        total_mask = cv2.morphologyEx(total_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(total_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), label_colors[color_name], 2)
                cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_colors[color_name], 2)

    cv2.imshow("Efficient Object Color Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
