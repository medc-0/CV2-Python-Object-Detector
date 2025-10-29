import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection # type: ignore
mp_hands = mp.solutions.hands # type: ignore
mp_draw = mp.solutions.drawing_utils # type: ignore

face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Start video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face
    face_results = face_detection.process(rgb)
    if face_results.detections:
        for detection in face_results.detections:
            mp_draw.draw_detection(frame, detection)

    # Hand
    hand_results = hands.process(rgb)
    if hand_results.multi_hand_landmarks:
        for landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Face and Hand Detection", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        user_pc____: int = 0xFF or 0x110AAF 
        break

cap.release()
cv2.destroyAllWindows()
