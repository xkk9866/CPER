"""Build aligned face tensors + metadata for FERPlus, RAVDESS and CREMA-D.

Outputs (all on E:):
  proc/ferplus_img48.npy     (N,48,48) uint8 gray
  proc/ferplus_meta.csv      split,label,votes(8),idx
  proc/{ravdess,cremad}_faces.npy  (N,16,S,S,3) uint8 RGB, 5-point aligned
  proc/{ravdess,cremad}_meta.csv   clip,label,actor,sentence,intensity,fold
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import argparse, ast, glob, io, os, sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image

EMO_ROOT = _data_root  # data root (E: on the first workstation, Y: NAS on the second)
ROOT = f"{EMO_ROOT}/datasets"
PROC = f"{ROOT}/proc"
S = 112  # stored face size (sources are ~80-110 px faces, so this is near lossless)

FERPLUS_EMO = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
RAVDESS_EMO = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
CREMAD_EMO = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
MEAD_EMO = ["Neutral", "Angry", "Contempt", "Disgust", "Fear", "Happy", "Sad", "Surprised"]

# ArcFace 5-point template (112x112), shrunk toward the face centre so brows and chin stay in view
_ARC = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                 [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)
TEMPLATE = (_ARC - [56.0, 66.0]) * 0.80 + [56.0, 60.0]
TEMPLATE = TEMPLATE * (S / 112.0)


def ferplus_label(raw):
    """Microsoft FERPlus 'majority' rule: drop single votes, keep if peak > half, else unknown."""
    raw = np.array(raw, np.float64).copy()
    raw[raw < 1.0 + 1e-9] = 0.0
    tot = raw.sum()
    if tot <= 0 or raw.max() <= 0.5 * tot:
        return -1
    k = int(raw.argmax())
    return k if k < 8 else -1  # 8 = unknown, 9 = not-a-face


def build_ferplus():
    lab = pd.read_csv(f"{ROOT}/ferplus/fer2013new.csv")
    imgs, rows = [], []
    for sp, usage in [("train", "Training"), ("val", "PublicTest"), ("test", "PrivateTest")]:
        df = pd.read_parquet(glob.glob(f"{ROOT}/fer2013/data/{ {'train':'train','val':'publicTest','test':'privateTest'}[sp] }*.parquet")[0])
        L = lab[lab.Usage == usage].reset_index(drop=True)
        assert len(L) == len(df)
        for i in range(len(df)):
            if not isinstance(L.loc[i, "Image name"], str):
                continue
            raw = L.loc[i, FERPLUS_EMO + ["unknown", "NF"]].values.astype(float)
            y = ferplus_label(raw)
            if y < 0:
                continue
            im = np.array(Image.open(io.BytesIO(df.image[i]["bytes"])).convert("L"))
            rows.append(dict(split=sp, label=y, **{f"v{k}": raw[k] for k in range(8)}, idx=len(imgs)))
            imgs.append(im)
    np.save(f"{PROC}/ferplus_img48.npy", np.stack(imgs))
    meta = pd.DataFrame(rows)
    meta.to_csv(f"{PROC}/ferplus_meta.csv", index=False)
    print("FERPlus", meta.groupby("split").size().to_dict(), meta.label.value_counts().sort_index().to_dict())


def align_frame(frame, pts, aspect):
    """Undo the 224x224 squash (aspect = original w/h), then similarity-align on 5 points."""
    h, w = frame.shape[:2]
    W = int(round(h * aspect))
    fr = cv2.resize(frame, (W, h), interpolation=cv2.INTER_CUBIC)
    p = np.array(pts, np.float32) * [W / w, 1.0]
    M, _ = cv2.estimateAffinePartial2D(p, TEMPLATE, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(fr, M, (S, S), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def read_clip(mp4, pt_file, aspect):
    pts = [ast.literal_eval(l) for l in open(pt_file).read().strip().splitlines()]
    cap = cv2.VideoCapture(mp4)
    out, last = [], None
    t = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        a = align_frame(f, pts[min(t, len(pts) - 1)], aspect) if t < len(pts) else None
        a = a if a is not None else last
        out.append(a)
        last = a
        t += 1
    out = [o for o in out if o is not None]
    if not out:
        return None
    while len(out) < 16:
        out.append(out[-1])
    return np.stack(out[:16])[..., ::-1]  # BGR -> RGB


def build_video(name):
    if name == "ravdess":
        base, aspect = f"{ROOT}/ravdess/RAVDESS_16F", 16 / 9
    elif name == "mead":
        base, aspect = f"{ROOT}/mead/MEAD_16F", 16 / 9
    else:
        base, aspect = f"{ROOT}/cremad/CREMA-D_16F", 4 / 3
    vids = sorted(glob.glob(f"{base}/Video/*/*/*.mp4"))
    clips, rows = [], []
    for k, v in enumerate(vids):
        v = v.replace("\\", "/")
        sp, cls, fn = v.split("/")[-3:]
        stem = fn[:-4]
        pt = f"{base}/Annotations/{sp}/point/{cls}/{stem}.txt"
        if not os.path.exists(pt):
            print("missing points", stem)
            continue
        arr = read_clip(v, pt, aspect)
        if arr is None:
            print("bad clip", stem)
            continue
        if name == "ravdess":
            f = stem.split("-")
            label, intensity, sentence, actor = int(f[2]) - 1, int(f[3]), int(f[4]), int(f[6])
            fold = (actor - 1) // 4  # 6 subject-independent folds of 4 actors
        elif name == "mead":  # e.g. M003_angry_level_1_..._008 ; label from the class folder
            f = stem.split("_")
            label, intensity, sentence, actor, fold = MEAD_EMO.index(cls), int(f[3]), int(f[-1]), f[0], -1
        else:
            f = stem.split("_")
            label, sentence, intensity, actor = CREMAD_EMO.index(f[2]), f[1], f[3], int(f[0])
            fold = -1
        rows.append(dict(clip=stem, label=label, actor=actor, sentence=sentence, intensity=intensity,
                         fold=fold, idx=len(clips)))
        clips.append(arr)
        if k % 500 == 0:
            print(name, k, len(vids), flush=True)
    meta = pd.DataFrame(rows)
    if name in ("cremad", "mead"):  # 5 subject-independent folds over sorted actor ids
        actors = sorted(meta.actor.unique())
        amap = {a: i * 5 // len(actors) for i, a in enumerate(actors)}
        meta["fold"] = meta.actor.map(amap)
    np.save(f"{PROC}/{name}_faces.npy", np.stack(clips))
    meta.to_csv(f"{PROC}/{name}_meta.csv", index=False)
    print(name, len(meta), meta.label.value_counts().sort_index().to_dict(), meta.groupby("fold").actor.nunique().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="ferplus,ravdess,cremad")
    a = ap.parse_args()
    os.makedirs(PROC, exist_ok=True)
    for w in a.which.split(","):
        {"ferplus": build_ferplus, "ravdess": lambda: build_video("ravdess"), "cremad": lambda: build_video("cremad"), "mead": lambda: build_video("mead")}[w]()
