import cv2 as cv
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector


LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 349, 363, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

defector = FaceDetector()
MeshDetector = FaceMeshDetector(maxFaces=1)

vi = cv.VideoCapture(0)
if not vi.isOpened():
    print("Error: Webcam not accessible.") 
    exit()

def process_eye(frame, eye_points, faces):
    eye_coordinates = np.array([[faces[0][p][0], faces[0][p][1]] for p in eye_points])
    (ex, ey, ew, eh) = cv.boundingRect(eye_coordinates)
    eye_roi = frame[ey:ey + eh, ex:ex + ew]
    eye_roi_gr = cv.cvtColor(eye_roi, cv.COLOR_BGR2GRAY)
    _, iris = cv.threshold(eye_roi_gr, 40, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(iris, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

    if contours:
        (ix, iy, iw, ih) = cv.boundingRect(contours[0])
        ix_centr, iy_centr = int(ix + (iw / 2) + ex), int(iy + (ih / 2) + ey)
        cv.circle(frame, (ix_centr, iy_centr), 5, (0, 200, 200), -1)

        ix_centr_e, iy_centr_e = ix + (iw / 2), iy + (ih / 2)
        if ix_centr_e > int(ew / 2):
            text = "Looking right"
        elif ix_centr_e < int(ew / 2):
            text = "Looking left"
        else:
            text = "Looking center"
        return text
    return None

while vi.isOpened():
    ret, frame = vi.read()
    if ret:
        face_img, bbox = defector.findFaces(frame, draw=False)
        face_img, faces = MeshDetector.findFaceMesh(face_img, draw=False)

        if bbox and faces:
            left_eye_text = process_eye(frame, LEFT_EYE, faces)
            right_eye_text = process_eye(frame, RIGHT_EYE, faces)

            if left_eye_text:
                cv.putText(frame, f"Left Eye: {left_eye_text}", (50, 50), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
            if right_eye_text:
                cv.putText(frame, f"Right Eye: {right_eye_text}", (50, 80), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

        cv.imshow('Webcam Feed', frame)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

vi.release()
cv.destroyAllWindows()
