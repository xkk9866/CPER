"""Face tensors + metadata for raw-video datasets without landmarks (eNTERFACE'05, MELD), with MTCNN 5-point
detection and the same similarity alignment as prep_data.py.

Outputs (E:):  proc/{enterface,meld}_faces.npy (N,16,112,112,3) uint8 RGB,  proc/{enterface,meld}_meta.csv
  enterface: 44 subjects, 6 expressions, 5 subject-independent folds over sorted subject ids
  meld:      the six main speakers of MELD (train+dev+test), 7 emotions, one fold per speaker (leave-one-speaker-out);
             'order' gives each utterance's position in the series (season, episode, dialogue, utterance) for the
             past-only protocol.  The largest face of each frame is taken as the speaker's.
"""
import argparse, glob, os, re
import cv2, numpy as np, pandas as pd, torch
from facenet_pytorch import MTCNN
from prep_data import TEMPLATE, S, PROC, ROOT

ENT_EMO = ["an", "di", "fe", "ha", "sa", "su"]
MELD_EMO = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
MAIN6 = ["Chandler", "Joey", "Monica", "Phoebe", "Rachel", "Ross"]


def frames(path, n=16, maxw=640):
    cap = cv2.VideoCapture(path); tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if tot <= 0:
        return []
    want = set(np.linspace(0, tot - 1, n).round().astype(int).tolist()); out = []
    for t in range(tot):
        ok, f = cap.read()
        if not ok:
            break
        if t in want:
            if f.shape[1] > maxw:
                f = cv2.resize(f, (maxw, int(f.shape[0] * maxw / f.shape[1])))
            out.append(f[..., ::-1].copy())  # RGB
    return out


def align(fr, pts):
    M, _ = cv2.estimateAffinePartial2D(np.asarray(pts, np.float32), TEMPLATE, method=cv2.LMEDS)
    return None if M is None else cv2.warpAffine(fr, M, (S, S), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def clip_faces(det, path):
    fs = frames(path)
    if not fs:
        return None
    out, last = [], None
    for f in fs:
        boxes, probs, pts = det.detect(f, landmarks=True)
        a = None
        if boxes is not None:
            k = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))  # largest face
            a = align(f, pts[k])
        a = a if a is not None else last
        out.append(a); last = a
    out = [o for o in out if o is not None]
    if not out:
        return None
    while len(out) < 16:
        out.append(out[-1])
    return np.stack(out[:16])


def items_enterface():
    for p in sorted(glob.glob(f"{ROOT}/enterface/2006_eNTERFACE05/**/*.avi", recursive=True)):
        p = p.replace("\\", "/"); fn = os.path.basename(p)
        m = re.match(r"s_?(\d+)_([a-z]{2})_(\d+)", fn)
        if not m or m.group(2) not in ENT_EMO:
            continue
        yield p, dict(clip=fn[:-4], label=ENT_EMO.index(m.group(2)), actor=int(m.group(1)), sentence=int(m.group(3)), intensity=0)


def tsec(t):  # "hh:mm:ss,mmm" -> seconds (broadcast time within the episode)
    h, m, x = t.split(":"); return int(h) * 3600 + int(m) * 60 + float(x.replace(",", "."))


def items_meld():
    vdir = f"{ROOT}/meld/MELD_raw/videos"
    for split in ["train", "dev", "test"]:
        df = pd.read_csv(f"{ROOT}/meld/{split}_sent_emo.csv", encoding="utf-8")
        for r in df.itertuples():
            if r.Speaker not in MAIN6:
                continue
            p = f"{vdir}/{split}_dia{r.Dialogue_ID}_utt{r.Utterance_ID}.mp4"
            if os.path.exists(p):
                yield p, dict(clip=f"{split}_dia{r.Dialogue_ID}_utt{r.Utterance_ID}", label=MELD_EMO.index(r.Emotion), actor=r.Speaker,
                              sentence=f"{split}_{r.Dialogue_ID}", intensity=0,
                              order=int(r.Season) * 10**8 + int(r.Episode) * 10**5 + tsec(r.StartTime))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--which", default="enterface")
    a = ap.parse_args()
    det = MTCNN(keep_all=True, device="cuda", post_process=False)
    for name in a.which.split(","):
        clips, rows = [], []
        for k, (p, row) in enumerate(items_enterface() if name == "enterface" else items_meld()):
            arr = clip_faces(det, p)
            if arr is None:
                continue
            row["idx"] = len(clips); rows.append(row); clips.append(arr)
            if k % 500 == 0:
                print(name, k, len(clips), flush=True)
        meta = pd.DataFrame(rows)
        subs = sorted(meta.actor.unique())
        nf = len(subs) if name == "meld" else 5
        meta["fold"] = meta.actor.map({s: i * nf // len(subs) for i, s in enumerate(subs)})
        np.save(f"{PROC}/{name}_faces.npy", np.stack(clips)); meta.to_csv(f"{PROC}/{name}_meta.csv", index=False)
        print(name, len(meta), meta.label.value_counts().sort_index().to_dict(), meta.groupby("fold").actor.nunique().to_dict(), flush=True)


if __name__ == "__main__":
    main()
