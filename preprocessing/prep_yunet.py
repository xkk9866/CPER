"""Face tensors + metadata for raw-video datasets without landmarks (eNTERFACE'05, MELD) with the OpenCV YuNet detector
(5 facial landmarks; the same similarity alignment and 112x112 template as prep_data.py / prep_mtcnn.py).  YuNet
replaces facenet-pytorch's MTCNN on the second workstation, whose torchvision build has no CUDA NMS (MTCNN ran at a few
seconds per clip there).  Largest face of each frame = the speaker's; a frame without a face repeats the previous one.

Outputs ($EMO_ROOT/datasets/proc):  {enterface,meld}_faces.npy (N,16,112,112,3) uint8 RGB, {enterface,meld}_meta.csv
  enterface: subjects, 6 expressions, 5 subject-independent folds over sorted subject ids
  meld: the six main speakers (train+dev+test), 7 emotions, one fold per speaker; 'order' = season / episode / start time

usage:  python prep_yunet.py --which enterface      python prep_yunet.py --which meld
"""
import argparse, os, time
import cv2, numpy as np, pandas as pd
from prep_data import TEMPLATE, S, PROC, ROOT, EMO_ROOT
from prep_mtcnn import frames, align, items_enterface, items_meld

YUNET = f"{EMO_ROOT}/models/yunet/face_detection_yunet_2023mar.onnx"
cv2.setNumThreads(2)


class Detector:
    def __init__(self):
        self.det = cv2.FaceDetectorYN.create(YUNET, "", (320, 320), score_threshold=0.6, nms_threshold=0.3, top_k=50)
        self.size = None

    def __call__(self, rgb):
        """largest face of an RGB frame -> 5 landmarks (image-left eye, image-right eye, nose, image-left mouth, image-right mouth)"""
        h, w = rgb.shape[:2]
        if self.size != (w, h):
            self.det.setInputSize((w, h)); self.size = (w, h)
        _, faces = self.det.detect(np.ascontiguousarray(rgb[..., ::-1]))  # YuNet expects BGR
        if faces is None or len(faces) == 0:
            return None
        k = int(np.argmax(faces[:, 2] * faces[:, 3]))
        return faces[k, 4:14].reshape(5, 2)


def clip_faces(det, path):
    fs = frames(path)
    if not fs:
        return None
    out, last = [], None
    for f in fs:
        pts = det(f)
        a = align(f, pts) if pts is not None else None
        a = a if a is not None else last
        out.append(a); last = a
    out = [o for o in out if o is not None]
    if not out:
        return None
    while len(out) < 16:
        out.append(out[-1])
    return np.stack(out[:16])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--which", default="enterface")
    a = ap.parse_args()
    det = Detector()
    for name in a.which.split(","):
        clips, rows, t = [], [], time.time()
        for k, (p, row) in enumerate(items_enterface() if name == "enterface" else items_meld()):
            arr = clip_faces(det, p)
            if arr is None:
                continue
            row["idx"] = len(clips); rows.append(row); clips.append(arr)
            if k % 100 == 0:
                print(name, k, len(clips), f"{(time.time() - t) / 60:.1f} min", flush=True)
        meta = pd.DataFrame(rows)
        subs = sorted(meta.actor.unique())
        nf = len(subs) if name == "meld" else 5
        meta["fold"] = meta.actor.map({s: i * nf // len(subs) for i, s in enumerate(subs)})
        np.save(f"{PROC}/{name}_faces.npy", np.stack(clips)); meta.to_csv(f"{PROC}/{name}_meta.csv", index=False)
        print(name, len(meta), meta.label.value_counts().sort_index().to_dict(), meta.groupby("fold").actor.nunique().to_dict(), flush=True)


if __name__ == "__main__":
    main()
