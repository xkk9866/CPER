"""Unpack the HF 16-frame archives (RAVDESS / CREMA-D / MEAD) and the eNTERFACE'05 archive under $EMO_ROOT/datasets and
build the aligned face tensors with prep_data.py / prep_mtcnn.py (second workstation; everything stays on the NAS root).

usage:  python prep_lab.py --which ravdess,cremad,mead,enterface
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import argparse, os, subprocess, sys, time, zipfile

E = _data_root
DS = f"{E}/datasets"
ARCH = {"ravdess": ("ravdess/RAVDESS_16F.zip", "ravdess/RAVDESS_16F"), "cremad": ("cremad/CREMA-D_16F.zip", "cremad/CREMA-D_16F"),
        "mead": ("mead/MEAD_16F.zip", "mead/MEAD_16F"), "enterface": ("enterface/2006_eNTERFACE05.zip", "enterface/2006_eNTERFACE05")}
HERE = os.path.dirname(os.path.abspath(__file__))


def unpack(name):
    zf, out = ARCH[name]
    zf, out = f"{DS}/{zf}", f"{DS}/{out}"
    if os.path.isdir(out) and os.path.exists(out + ".unpacked"):
        print("already unpacked", out, flush=True); return
    t = time.time()
    with zipfile.ZipFile(zf) as z:
        z.extractall(os.path.dirname(out))
    open(out + ".unpacked", "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"unpacked {name} in {(time.time() - t) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--which", default="ravdess,cremad,mead,enterface")
    a = ap.parse_args()
    for w in a.which.split(","):
        unpack(w)
        if os.path.exists(f"{DS}/proc/{w}_faces.npy"):
            print("faces exist", w, flush=True); continue
        t = time.time()
        if w == "enterface":
            subprocess.run([sys.executable, "-u", f"{HERE}/prep_mtcnn.py", "--which", "enterface"], check=True, cwd=HERE)
        else:
            subprocess.run([sys.executable, "-u", f"{HERE}/prep_data.py", "--which", w], check=True, cwd=HERE)
        print(f"prepared {w} in {(time.time() - t) / 60:.1f} min", flush=True)
    print("PREP DONE", flush=True)
