import cv2
import dlib
import numpy as np

# Lista de embeddings y nombres (debes definir estos basado en tu base de datos)
face_embeddings = [
    # Ejemplo: np.array([...]),
]
names = [
    # Ejemplo: "Persona 1",
]

# Inicialización de detector y modelo para face embedding
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Capturar un video
cap = cv2.VideoCapture("\media\video_of_people_walking (1080p).mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = detector(gray)

    for face in faces_detected:
        shape = predictor(gray, face)
        face_embedding = face_rec_model.compute_face_descriptor(frame, shape)
        face_embedding_np = np.array(face_embedding)

        distances = np.linalg.norm(face_embeddings - face_embedding_np, axis=1)
        best_match_index = np.argmin(distances)

        if distances[best_match_index] < 0.6:  # 0.6 es un umbral, puedes ajustarlo
            cv2.putText(frame, names[best_match_index], (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
