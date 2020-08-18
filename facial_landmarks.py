import os
import argparse
import numpy as np
import cv2
from mtcnn import MTCNN
import face_alignment

from videostream import QueuedStream

from line_segments import PARTS, COLORS


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_path", type=str, default=r"C:\Users\Thanh\Downloads\Bill.mp4")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    #detector = MTCNN()
    face_al = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=args.device, face_detector="sfd")
    stream = QueuedStream(uri=args.video_path)
    stream.start()

    run = True

    while run:
        ret, frame, frame_id = stream.read()

        if not ret:
            break
        rgb_frame = frame[:, :, ::-1]
        #faces = detector.detect_faces(rgb_frame)
        #for face in faces:
        #    x, y, w, h = face["box"]
        #    confidence = np.round(face["confidence"], 2)
        #    if confidence < 0.98:
        #        continue
        #    cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 255, 0))
        #    cv2.putText(frame, text="w: {}, h: {}, confidence: {}".format(w, h, confidence), org=(x, y - 30),
        #                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #                fontScale=0.5, color=(0, 0, 255))
        landmarks = face_al.get_landmarks_from_image(rgb_frame)
        for landmark in landmarks:
            #for i, (x, y) in enumerate(landmark):
            #    #cv2.circle(frame, center=(int(x), int(y)), radius=2, color=(0, 0, 255), thickness=-1)
            #    cv2.putText(frame, text=str(i), org=(int(x), int(y)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #                fontScale=0.3, color=(0, 0, 255))
            for part in PARTS:
                for line in part:
                    cv2.line(frame, pt1=(int(landmark[line[0]][0]), int(landmark[line[0]][1])), pt2=(int(landmark[line[1]][0]), int(landmark[line[1]][1])), color=COLORS[part],
                             thickness=1)
        cv2.imshow("Anh", frame)
        cv2.waitKey(1)
