import cv2
import dlib

# Inicialización de detector
detector = dlib.get_frontal_face_detector()

# Capturar un video
cap = cv2.VideoCapture("\media\video_of_people_walking (1080p).mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = detector(gray)

    for face in faces_detected:
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
