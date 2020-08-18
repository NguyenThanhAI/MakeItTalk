import os
import argparse
from tqdm import tqdm
import numpy as np
import cv2

import face_alignment

from line_segments import PARTS, COLORS


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--videos_dir", type=str, default=r"K:\VoxCeleb2\vox2_test_mp4\mp4")
    parser.add_argument("--saved_dir", type=str, default=r"K:\VoxCeleb2\data")
    parser.add_argument("--out_image_size", type=str, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--show_image", type=str2bool, default=True)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    face_al = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=args.device, face_detector="sfd")

    if not os.path.exists(args.saved_dir):
        os.makedirs(args.saved_dir, exist_ok=True)

    videos_list = []
    for dirs, _, files in os.walk(args.videos_dir):
        for file in files:
            if file.endswith(".mp4"):
                videos_list.append(os.path.join(dirs, file))

    for video in tqdm(videos_list):
        cap = cv2.VideoCapture(video)

        length = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        while True:

            while True:
                source_index = np.random.choice(np.arange(length), size=1, replace=False)[0]
                target_index = np.random.choice(np.arange(length), size=1, replace=False)[0]
                if abs(target_index - source_index) > 0.8 * length:
                    break

            cap.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            ret, source_image = cap.read()

            if not ret:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
            ret, target_image = cap.read()

            if not ret:
                continue

            source_image = cv2.resize(source_image, dsize=(256, 256))
            target_image = cv2.resize(target_image, dsize=(256, 256))

            source_landmarks = 255 * np.ones_like(source_image)
            landmarks = face_al.get_landmarks_from_image(source_image)
            if len(landmarks) != 1:
                continue
            for landmark in landmarks:
                for part in PARTS:
                    for line in part:
                        cv2.line(source_landmarks, pt1=(int(landmark[line[0]][0]), int(landmark[line[0]][1])),
                                 pt2=(int(landmark[line[1]][0]), int(landmark[line[1]][1])), color=COLORS[part],
                                 thickness=1)

            target_landmarks = 255 * np.ones_like(target_image)
            landmarks = face_al.get_landmarks_from_image(target_image)
            if len(landmarks) != 1:
                continue
            for landmark in landmarks:
                for part in PARTS:
                    for line in part:
                        cv2.line(target_landmarks, pt1=(int(landmark[line[0]][0]), int(landmark[line[0]][1])),
                                 pt2=(int(landmark[line[1]][0]), int(landmark[line[1]][1])), color=COLORS[part],
                                 thickness=1)

            if args.show_image:
                cv2.imshow("Source image", source_image)
                cv2.imshow("Target image", target_image)
                cv2.imshow("Source landmarks", source_landmarks)
                cv2.imshow("Target landmarks", target_landmarks)
                cv2.waitKey(200)

            rel_dir = os.path.relpath(video, args.videos_dir)
            save_path = os.path.join(args.saved_dir, rel_dir)
            save_dir = save_path.split(".")[0]

            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            cv2.imwrite(os.path.join(save_dir, "source_image.jpg"), source_image)
            cv2.imwrite(os.path.join(save_dir, "source_landmarks.jpg"), source_landmarks)
            cv2.imwrite(os.path.join(save_dir, "target_image.jpg"), target_image)
            cv2.imwrite(os.path.join(save_dir, "target_landmarks.jpg"), target_landmarks)

            break
