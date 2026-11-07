"""Natural-deployment data for the round-7/8 study (run once, before run_cpn.py).  Everything is written to E:.

  meld   natural TV dialogue (6 main characters, chronological order -> natural 'past' references).  Representation
         'avt' = [face: frozen CLIP-L/14 (existing meld_l14.npy) | speech: frozen WavLM-Large, mean over time, average
         of the 24 transformer layers | transcript: frozen all-mpnet-base-v2, mean-pooled].  Labels are annotated from
         the full audio-visual-text context, so the speech and transcript streams remove the label / visible-face
         mismatch of the face-only representation; the speaker's voice also fixes the identity (no face tracking).
  shemo  ShEMO (Mohamad Nezami et al., LRE 2019): Persian radio plays, 87 speakers, semi-natural speech whose emotion
         composition per speaker is natural (strongly skewed).  NEVER used during method development: the independent
         test set.  Representation 'wavlm' (as above).  Classes fixed a priori: anger, happiness, neutral, sadness,
         surprise (fear has 38 utterances in the whole corpus and is dropped); single-utterance (speaker, class) cells
         and speakers with fewer than 8 utterances or fewer than 2 classes are dropped; 5 speaker-independent folds
         balanced by utterance count.
  emoz   Emozionalmente (Catania et al., IEEE/ACM TASLP 2025; Zenodo 12616095, CC BY 4.0): Italian, 431 crowd-sourced
         amateur actors recording on their OWN devices (natural channel / identity variation), seven classes; each actor
         chose how many clips of which emotions to record, so per-person compositions are naturally skewed.  Downloaded
         in round 8 AFTER the method was frozen and never evaluated during development: the untouched external test.
         Representation 'wavlm'.  Inclusion rule = the ShEMO rule, fixed before any result: single-utterance (speaker,
         class) cells dropped, then speakers with fewer than 8 utterances or fewer than 2 classes; 5 speaker-independent
         folds balanced by utterance count.
  thai   THAI-SER (AIResearch / VISTEC, CC BY-SA 4.0), the ZOOM subset: 40 actors recorded remotely over video calls
         (natural channel / room variation), 60% improvised dialogue; labels = the MAJORITY PERCEIVED emotion of 8
         annotators (neutral, angry, happy, sad, frustrated; 'None' / 'other' dropped), so each speaker's composition
         is natural; utterances keep their order inside every improvisation / script -> natural 'past' references.
         Downloaded in round 9 after the method was frozen and never evaluated during development: the second
         untouched external test.  Same inclusion rule as ShEMO / Emozionalmente.  Representation 'wavlm'.
  thaix  THAI-SER cross-condition test (round 10): the source network is trained ONLY on the 40 Zoom actors (fold -1,
         the training pool) and frozen; it is evaluated on the 160 different actors of the 80 STUDIO sessions (studio
         microphone, never seen by the network; 5 session-isolated folds).  Labels / inclusion rule as 'thai'.
         Downloaded in round 10 and never evaluated before the formal run.  Run --thai first.
Both encoders are self-supervised / generic sentence encoders (no emotion labels in their training).

usage:  python prep_natural.py --models          download WavLM-Large and all-mpnet-base-v2 to $EMO_ROOT/models
        python prep_natural.py --meld            -> features/clip/meld_avt.npy
        python prep_natural.py --shemo           -> datasets/proc/shemo_meta.csv, features/clip/shemo_wavlm.npy
        python prep_natural.py --emoz            -> datasets/proc/emoz_meta.csv, features/clip/emoz_wavlm.npy
        python prep_natural.py --thai            -> datasets/proc/thai_meta.csv, features/clip/thai_wavlm.npy
        python prep_natural.py --thaix           -> datasets/proc/thaix_meta.csv, features/clip/thaix_wavlm.npy
        (ShEMO archives: downloaded from the authors' GitHub to $EMO_ROOT/datasets/shemo; if the download is
        blocked, put female.zip and male.zip there by hand.)
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import argparse, os, re, sys, urllib.request, zipfile
E = _data_root  # data root (E: on the first workstation, the NAS root on the second)
os.environ.setdefault("HF_HOME", f"{E}/cache/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{E}/cache/hf")
import numpy as np, pandas as pd, torch

PROC, FEATD, MODELS = f"{E}/datasets/proc", f"{E}/features/clip", f"{E}/models"
WAVLM, MPNET = f"{MODELS}/wavlm-large", f"{MODELS}/all-mpnet-base-v2"
MELD = f"{E}/datasets/meld"
SHEMO = f"{E}/datasets/shemo"
SHEMO_URLS = ["https://github.com/mansourehk/ShEMO/raw/master/{f}",
              "https://raw.githubusercontent.com/mansourehk/ShEMO/master/{f}"]
SHEMO_CLASSES = {"A": 0, "H": 1, "N": 2, "S": 3, "W": 4}  # anger, happiness, neutral, sadness, surprise (fear dropped)
EMOZ = f"{E}/datasets/emozionalmente"
EMOZ_URL = "https://zenodo.org/records/12616095/files/emozionalmente.zip?download=1"
EMOZ_MD5 = "398c6502dd2b0e1e810f927c12897547"
EMOZ_CLASSES = {"anger": 0, "disgust": 1, "fear": 2, "joy": 3, "neutrality": 4, "sadness": 5, "surprise": 6}
THAI = f"{E}/datasets/thaiser"
THAI_URL = "https://github.com/vistec-AI/dataset-releases/releases/download/v1/{f}"
THAI_FILES = ["emotion_label.json", "actor_demography.json", "zoom1-10.zip", "zoom11-20.zip"]
THAI_CLASSES = {"Angry": 0, "Frustrated": 1, "Happy": 2, "Neutral": 3, "Sad": 4}
dev = "cuda" if torch.cuda.is_available() else "cpu"
LIMIT = None  # --limit N: smoke test on the first N clips, outputs go to $EMO_ROOT/tmp only


def models():
    from huggingface_hub import snapshot_download
    for repo, dst in [("microsoft/wavlm-large", WAVLM), ("sentence-transformers/all-mpnet-base-v2", MPNET)]:
        if not os.path.exists(os.path.join(dst, "config.json")):
            snapshot_download(repo, local_dir=dst, allow_patterns=["*.json", "*.txt", "pytorch_model.bin", "model.safetensors"],
                              ignore_patterns=["openvino/*", "onnx/*", "1_Pooling/*"])
        print("model ready:", dst)


def load_wav(path, sr=16000, max_s=20.0):
    import soundfile as sf
    w, r = sf.read(path, dtype="float32", always_2d=True)
    w = w.mean(1)
    if r != sr:  # polyphase resampling (fast, no JIT)
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(r), sr)
        w = resample_poly(w, sr // g, int(r) // g).astype(np.float32)
    w = w[: int(max_s * sr)]
    if len(w) < sr // 10:  # pad very short clips to 0.1 s
        w = np.pad(w, (0, sr // 10 - len(w)))
    return w


@torch.no_grad()
def wavlm_feats(paths, bs=8, partial=None):
    """time-mean of every transformer layer of frozen WavLM-Large, averaged over the 24 layers -> (N, 1024).
    Robust on a shared workstation: fp16 weights, a batch is retried after a CUDA out-of-memory (other processes may
    hold the GPU), and progress is checkpointed to `partial` (resumed when the clip count matches)."""
    import time
    from transformers import AutoFeatureExtractor, WavLMModel
    fe = AutoFeatureExtractor.from_pretrained(WAVLM)
    m = WavLMModel.from_pretrained(WAVLM, torch_dtype=torch.float16).to(dev).eval()
    out = np.zeros((len(paths), m.config.hidden_size), np.float32)
    start, missing = 0, 0
    if partial and os.path.exists(partial + ".npy") and os.path.exists(partial + ".progress"):
        done = int(open(partial + ".progress").read())
        prev = np.load(partial + ".npy")
        if prev.shape == out.shape:
            out[:done] = prev[:done]; start = done
            print(f"wavlm: resuming at {start}/{len(paths)}", flush=True)
    for i in range(start, len(paths), bs):
        ws = []
        for p in paths[i:i + bs]:
            if p is None or not os.path.exists(p):
                ws.append(np.zeros(16000, np.float32)); missing += 1
            else:
                ws.append(load_wav(p))
        for attempt in range(40):
            try:
                inp = fe(ws, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
                hs = m(inp.input_values.to(dev, torch.float16), attention_mask=inp.attention_mask.to(dev), output_hidden_states=True).hidden_states
                L = hs[-1].shape[1]  # frame mask at the output rate
                lens = m._get_feat_extract_output_lengths(inp.attention_mask.sum(1)).to(dev)
                mask = (torch.arange(L, device=dev)[None] < lens[:, None]).float()[..., None]
                layers = torch.stack([(h.float() * mask).sum(1) / mask.sum(1).clamp_min(1) for h in hs[1:]], 0)  # (24, b, H)
                out[i:i + len(ws)] = layers.mean(0).float().cpu().numpy()
                break
            except (torch.OutOfMemoryError, MemoryError, RuntimeError) as e:  # noqa: PERF203
                if "out of memory" not in str(e).lower() and not isinstance(e, MemoryError):
                    raise
                torch.cuda.empty_cache(); print(f"  OOM at {i}, retry {attempt}", flush=True); time.sleep(30)
        else:
            raise RuntimeError("persistent out-of-memory")
        if partial and (i // bs) % 50 == 0:
            np.save(partial + ".npy", out); open(partial + ".progress", "w").write(str(i + len(ws)))
        if i % (bs * 100) == 0:
            print(f"wavlm {i}/{len(paths)}", flush=True)
    print("wavlm done, missing audio:", missing, flush=True)
    assert np.isfinite(out).all(), "non-finite WavLM features"
    return out


@torch.no_grad()
def mpnet_feats(texts, bs=64):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MPNET)
    m = AutoModel.from_pretrained(MPNET).to(dev).eval()
    out = []
    for i in range(0, len(texts), bs):
        t = tok(texts[i:i + bs], padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
        h = m(**t).last_hidden_state
        a = t.attention_mask[..., None].float()
        e = (h * a).sum(1) / a.sum(1).clamp_min(1)
        out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def meld():
    meta = pd.read_csv(f"{PROC}/meld_meta.csv")
    assert (meta.idx.values == np.arange(len(meta))).all()
    face = np.load(f"{FEATD}/meld_l14.npy")
    assert len(face) == len(meta)
    md = pd.read_csv(f"{MELD}/MELD_raw/metadata.csv")
    apath = {os.path.splitext(f)[0]: os.path.join(MELD, "MELD_raw", p) for f, p in zip(md.Audio_Filename, md.Audio_Path)}
    if LIMIT:
        meta, face = meta.iloc[:LIMIT], face[:LIMIT]
    paths = [apath.get(c) for c in meta["clip"]]
    txt = {}
    for sp in ["train", "dev", "test"]:
        t = pd.read_csv(f"{MELD}/{sp}_sent_emo.csv", encoding_errors="replace")
        for d, u, s in zip(t.Dialogue_ID, t.Utterance_ID, t.Utterance):
            txt[f"{sp}_dia{d}_utt{u}"] = str(s).replace("\x92", "'").replace("\x91", "'").replace("\x93", '"').replace("\x94", '"')
    texts = [txt.get(c, "") for c in meta["clip"]]
    print("MELD clips", len(meta), "audio found", sum(p is not None and os.path.exists(p) for p in paths), "text found", sum(bool(t) for t in texts))
    aud = wavlm_feats(paths, partial=f"{E}/tmp/meld_wavlm_partial")
    tx = mpnet_feats(texts)
    X = np.concatenate([face, aud, tx], 1).astype(np.float32)
    fn = f"{E}/tmp/meld_avt_smoke.npy" if LIMIT else f"{FEATD}/meld_avt.npy"
    np.save(fn, X)
    print("saved", fn, X.shape)


def fetch_shemo():
    os.makedirs(SHEMO, exist_ok=True)
    for f in ["female.zip", "male.zip"]:
        dst = f"{SHEMO}/{f}"
        if os.path.exists(dst) and os.path.getsize(dst) > 1 << 20:
            if not os.path.exists(dst + ".extracted"):  # pre-downloaded archive: extract it once
                with zipfile.ZipFile(dst) as z:
                    z.extractall(SHEMO)
                open(dst + ".extracted", "w").write("ok")
            continue
        for u in SHEMO_URLS:
            try:
                print("downloading", u.format(f=f), flush=True)
                urllib.request.urlretrieve(u.format(f=f), dst + ".part")
                if zipfile.is_zipfile(dst + ".part"):
                    os.replace(dst + ".part", dst); break
            except Exception as e:  # noqa: BLE001
                print("  failed:", e)
        if not os.path.exists(dst):
            raise SystemExit(f"could not download {f}: put female.zip and male.zip (ShEMO, github.com/mansourehk/ShEMO) "
                             f"into {SHEMO} and rerun")
        with zipfile.ZipFile(dst) as z:
            z.extractall(SHEMO)


def shemo():
    fetch_shemo()
    wavs = []
    for root, _, files in os.walk(SHEMO):
        for f in files:
            m = re.fullmatch(r"([FM]\d\d)([AHNSWF])(\d\d)\.wav", f, flags=re.I)
            if m:
                wavs.append((m.group(1).upper(), m.group(2).upper(), int(m.group(3)), os.path.join(root, f)))
    df = pd.DataFrame(wavs, columns=["actor", "emo", "num", "path"]).drop_duplicates("path")
    print("ShEMO utterances", len(df), df.emo.value_counts().to_dict())
    df = select_people(df[df.emo.isin(list(SHEMO_CLASSES))])  # fixed a priori, before any result
    df["label"] = df.emo.map(SHEMO_CLASSES)
    df["fold"] = balanced_folds(df.actor)
    df = df.sort_values(["actor", "emo", "num"]).reset_index(drop=True)
    meta = pd.DataFrame(dict(clip=[os.path.splitext(os.path.basename(p))[0] for p in df.path], label=df.label.values,
                             actor=df.actor.values, sentence=df.num.values, intensity=0, fold=df.fold.values,
                             idx=np.arange(len(df))))
    print("kept", len(meta), "utterances,", meta.actor.nunique(), "speakers; classes", meta.label.value_counts().sort_index().to_dict(),
          "; folds", meta.fold.value_counts().sort_index().to_dict())
    if LIMIT:
        X = wavlm_feats(df.path.tolist()[:LIMIT])
        print("smoke features", X.shape, "(nothing written to proc/ or features/)"); return
    X = wavlm_feats(df.path.tolist(), partial=f"{E}/tmp/shemo_wavlm_partial")
    np.save(f"{FEATD}/shemo_wavlm.npy", X)
    meta.to_csv(f"{PROC}/shemo_meta.csv", index=False)
    print("saved", f"{PROC}/shemo_meta.csv", f"{FEATD}/shemo_wavlm.npy", X.shape)


def select_people(df, min_utt=8, min_cls=2):
    """the a-priori inclusion rule shared by ShEMO and Emozionalmente: a (speaker, class) cell with one utterance admits
    no same-class reference -> dropped; then speakers with fewer than min_utt utterances or fewer than min_cls classes"""
    df = df[df.groupby(["actor", "emo"]).emo.transform("size") >= 2]
    g = df.groupby("actor")
    keep = g.size().ge(min_utt) & g.emo.nunique().ge(min_cls)
    return df[df.actor.isin(keep[keep].index)].copy()


def balanced_folds(actor, k=5):
    """speaker-independent folds balanced by utterance count (greedy, deterministic)"""
    spk = actor.value_counts().sort_index()
    load = np.zeros(k); fold = {}
    for s_ in spk.sort_values(ascending=False, kind="stable").index:
        j = int(np.argmin(load)); fold[s_] = j; load[j] += spk[s_]
    return actor.map(fold)


def emoz():
    import hashlib
    os.makedirs(EMOZ, exist_ok=True)
    zf = f"{EMOZ}/emozionalmente.zip"
    if not os.path.exists(zf):
        print("downloading", EMOZ_URL, flush=True)
        urllib.request.urlretrieve(EMOZ_URL, zf + ".part"); os.replace(zf + ".part", zf)
    md5 = hashlib.md5(open(zf, "rb").read()).hexdigest()
    assert md5 == EMOZ_MD5, f"checksum mismatch {md5}"
    root = f"{EMOZ}/emozionalmente"
    if not os.path.exists(f"{root}/metadata/samples.csv"):
        with zipfile.ZipFile(zf) as z:
            z.extractall(EMOZ, [n for n in z.namelist() if n.startswith("emozionalmente/") and ".." not in n])
    s = pd.read_csv(f"{root}/metadata/samples.csv")
    df = pd.DataFrame(dict(actor=s.actor.astype(str), emo=s.emotion_expressed.astype(str), sentence=s.sentence.astype(str),
                           path=[f"{root}/audio/{f}" for f in s.file_name]))
    print("Emozionalmente clips", len(df), "speakers", df.actor.nunique(), df.emo.value_counts().to_dict())
    df = select_people(df[df.emo.isin(list(EMOZ_CLASSES))])
    df["label"] = df.emo.map(EMOZ_CLASSES)
    df["fold"] = balanced_folds(df.actor)
    df = df.sort_values(["actor", "emo", "path"]).reset_index(drop=True)
    spk = {a_: f"S{i:03d}" for i, a_ in enumerate(sorted(df.actor.unique()))}  # short anonymous ids
    meta = pd.DataFrame(dict(clip=[os.path.splitext(os.path.basename(p_))[0] for p_ in df.path], label=df.label.values,
                             actor=df.actor.map(spk).values, sentence=df.sentence.values, intensity=0,
                             fold=df.fold.values, idx=np.arange(len(df))))
    print("kept", len(meta), "utterances,", meta.actor.nunique(), "speakers; classes", meta.label.value_counts().sort_index().to_dict(),
          "; folds", meta.fold.value_counts().sort_index().to_dict())
    if LIMIT:
        X = wavlm_feats(df.path.tolist()[:LIMIT])
        print("smoke features", X.shape, "(nothing written to proc/ or features/)"); return
    X = wavlm_feats(df.path.tolist(), partial=f"{E}/tmp/emoz_wavlm_partial")
    np.save(f"{FEATD}/emoz_wavlm.npy", X)
    meta.to_csv(f"{PROC}/emoz_meta.csv", index=False)
    print("saved", f"{PROC}/emoz_meta.csv", f"{FEATD}/emoz_wavlm.npy", X.shape)


def thai():
    import json
    os.makedirs(THAI, exist_ok=True)
    for f in THAI_FILES:
        if not os.path.exists(f"{THAI}/{f}"):
            print("downloading", f, flush=True)
            urllib.request.urlretrieve(THAI_URL.format(f=f), f"{THAI}/{f}.part"); os.replace(f"{THAI}/{f}.part", f"{THAI}/{f}")
    lab = json.load(open(f"{THAI}/emotion_label.json", encoding="utf-8"))
    rows = []
    for zf in ("zoom1-10.zip", "zoom11-20.zip"):
        with zipfile.ZipFile(f"{THAI}/{zf}") as z:
            names = [n for n in z.namelist() if n.endswith(".flac") and ".." not in n]
            if not os.path.exists(f"{THAI}/{names[-1]}"):
                z.extractall(THAI, names)
        for n in names:
            b = os.path.basename(n)
            m = re.fullmatch(r"z(\d+)_mic_actor(\d+)_(impro|script)(\d+)_(\d+)(?:_(\d+)([ab]))?\.flac", b)
            if not m or b not in lab:
                continue
            rec, act, kind, k, i, j, ab = m.groups()
            emo = lab[b][0]["majority_emo"]
            order = ((int(rec) * 2 + (kind == "script")) * 100 + int(k)) * 10000 + int(i) * 10 + (int(j) if j else 0) + (0.5 if ab == "b" else 0)
            rows.append((f"A{int(act):03d}", emo, f"{kind}{k}", order, f"{THAI}/{n}", f"Z{int(rec):03d}"))
    df = pd.DataFrame(rows, columns=["actor", "emo", "sentence", "order", "path", "session"])
    print("THAI-SER zoom utterances", len(df), "speakers", df.actor.nunique(), df.emo.value_counts().to_dict())
    df = select_people(df[df.emo.isin(list(THAI_CLASSES))])  # fixed a priori: the ShEMO / Emozionalmente rule
    df["label"] = df.emo.map(THAI_CLASSES)
    # SESSION-isolated folds (round 10): the two actors of one Zoom recording share a dialogue, so a recording is never
    # split between training and test people
    df["fold"] = balanced_folds(df.session)
    df = df.sort_values(["actor", "order"]).reset_index(drop=True)
    meta = pd.DataFrame(dict(clip=[os.path.splitext(os.path.basename(p_))[0] for p_ in df.path], label=df.label.values,
                             actor=df.actor.values, sentence=df.sentence.values, intensity=0, order=df.order.values,
                             fold=df.fold.values, session=df.session.values, idx=np.arange(len(df))))
    print("kept", len(meta), "utterances,", meta.actor.nunique(), "speakers; classes", meta.label.value_counts().sort_index().to_dict(),
          "; folds", meta.fold.value_counts().sort_index().to_dict())
    if LIMIT:
        X = wavlm_feats(df.path.tolist()[:LIMIT])
        print("smoke features", X.shape, "(nothing written to proc/ or features/)"); return
    old = f"{PROC}/thai_meta.csv"
    if os.path.exists(f"{FEATD}/thai_wavlm.npy") and os.path.exists(old) and             pd.read_csv(old)["clip"].tolist() == meta["clip"].tolist():  # same clips in the same order: features stay valid
        X = np.load(f"{FEATD}/thai_wavlm.npy")
        print("features reused (identical clip order); only the folds are rewritten")
    else:
        X = wavlm_feats(df.path.tolist(), partial=f"{E}/tmp/thai_wavlm_partial")
        np.save(f"{FEATD}/thai_wavlm.npy", X)
    meta.to_csv(f"{PROC}/thai_meta.csv", index=False)
    print("saved", f"{PROC}/thai_meta.csv", f"{FEATD}/thai_wavlm.npy", X.shape)


THAI_STUDIO = [f"studio{a}-{a + 9}.zip" for a in range(1, 81, 10)]


def thaix():
    import json
    zmeta, zX = pd.read_csv(f"{PROC}/thai_meta.csv"), np.load(f"{FEATD}/thai_wavlm.npy")
    assert len(zmeta) == len(zX), "run prep_natural.py --thai first"
    lab = json.load(open(f"{THAI}/emotion_label.json", encoding="utf-8"))
    rows = []
    for zf in THAI_STUDIO:
        if os.path.exists(f"{THAI}/{zf}") and not zipfile.is_zipfile(f"{THAI}/{zf}"):
            os.remove(f"{THAI}/{zf}")  # an interrupted download: fetch again
        if not os.path.exists(f"{THAI}/{zf}"):
            print("downloading", zf, flush=True)
            urllib.request.urlretrieve(THAI_URL.format(f=zf), f"{THAI}/{zf}.part"); os.replace(f"{THAI}/{zf}.part", f"{THAI}/{zf}")
        with zipfile.ZipFile(f"{THAI}/{zf}") as z:
            names = [n for n in z.namelist() if n.endswith(".flac") and ".." not in n and "_con_" in os.path.basename(n)]
            if names and not os.path.exists(f"{THAI}/{names[-1]}"):
                z.extractall(THAI, names)
        for n in names:
            b = os.path.basename(n)
            m = re.fullmatch(r"s(\d+)_con_actor(\d+)_(impro|script)(\d+)_(\d+)(?:_(\d+)([ab]))?\.flac", b)
            if not m or b not in lab:
                continue
            rec, act, kind, k, i, j, ab = m.groups()
            order = ((int(rec) * 2 + (kind == "script")) * 100 + int(k)) * 10000 + int(i) * 10 + (int(j) if j else 0) + (0.5 if ab == "b" else 0)
            rows.append((f"A{int(act):03d}", lab[b][0]["majority_emo"], f"{kind}{k}", order, f"{THAI}/{n}", f"S{int(rec):03d}"))
    df = pd.DataFrame(rows, columns=["actor", "emo", "sentence", "order", "path", "session"])
    df = df[~df.actor.isin(set(zmeta.actor))]  # people of the training pool are never test people
    print("THAI-SER studio utterances", len(df), "speakers", df.actor.nunique(), df.emo.value_counts().to_dict())
    df = select_people(df[df.emo.isin(list(THAI_CLASSES))])
    df["label"] = df.emo.map(THAI_CLASSES)
    df["fold"] = balanced_folds(df.session)
    df = df.sort_values(["actor", "order"]).reset_index(drop=True)
    smeta = pd.DataFrame(dict(clip=[os.path.splitext(os.path.basename(p_))[0] for p_ in df.path], label=df.label.values,
                              actor=df.actor.values, sentence=df.sentence.values, intensity=0, order=df.order.values,
                              fold=df.fold.values, session=df.session.values))
    pool = zmeta.drop(columns=["idx"]).assign(fold=-1)
    meta = pd.concat([pool, smeta], ignore_index=True)
    meta["idx"] = np.arange(len(meta))
    print("training pool (Zoom, fold -1):", len(pool), "clips,", pool.actor.nunique(), "people; test (Studio):", len(smeta),
          "clips,", smeta.actor.nunique(), "people; folds", smeta.fold.value_counts().sort_index().to_dict())
    if LIMIT:
        X = wavlm_feats(df.path.tolist()[:LIMIT])
        print("smoke features", X.shape, "(nothing written to proc/ or features/)"); return
    miss = [p_ for p_ in df.path if not os.path.exists(p_)]
    assert not miss, f"{len(miss)} extracted Studio clips missing (e.g. {miss[:3]}): delete the incomplete zip and rerun"
    old_m, old_x = f"{PROC}/thaix_meta.csv", f"{FEATD}/thaix_wavlm.npy"
    if os.path.exists(old_m) and os.path.exists(old_x) and pd.read_csv(old_m)["clip"].tolist() == meta["clip"].tolist()             and np.load(old_x, mmap_mode="r").shape[0] == len(meta):  # idempotent: identical clips in identical order
        X = np.load(old_x)
        print("features reused (identical clip order)")
    else:
        X = np.concatenate([zX, wavlm_feats(df.path.tolist(), partial=f"{E}/tmp/thaix_wavlm_partial")]).astype(np.float32)
    assert len(X) == len(meta) and np.isfinite(X).all() and (np.abs(X).sum(1) > 0).all(), "invalid Studio features"
    np.save(f"{FEATD}/thaix_wavlm.npy", X)
    meta.to_csv(f"{PROC}/thaix_meta.csv", index=False)
    print("saved", f"{PROC}/thaix_meta.csv", f"{FEATD}/thaix_wavlm.npy", X.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true"); ap.add_argument("--meld", action="store_true")
    ap.add_argument("--shemo", action="store_true"); ap.add_argument("--emoz", action="store_true")
    ap.add_argument("--thai", action="store_true"); ap.add_argument("--thaix", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="smoke test on the first N clips (no output files)")
    a = ap.parse_args()
    LIMIT = a.limit
    os.makedirs(f"{E}/tmp", exist_ok=True)
    if not (a.models or a.meld or a.shemo or a.emoz or a.thai or a.thaix):
        ap.print_help(); sys.exit(0)
    if a.models or a.meld or a.shemo or a.emoz or a.thai or a.thaix:
        models()
    if a.meld:
        meld()
    if a.shemo:
        shemo()
    if a.emoz:
        emoz()
    if a.thai:
        thai()
    if a.thaix:
        thaix()
