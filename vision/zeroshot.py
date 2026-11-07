"""Zero-shot baselines: CLIP (prompt ensemble) and Qwen2-VL-2B-Instruct (4-bit, option-word scoring)."""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import FaceSet, LABELS, MEAN, STD
from train import metrics

RUNS = f"{_data_root}/runs/zeroshot"
TEMPLATES = ["a photo of a {} face.", "a face showing {}.", "a close-up photo of a person looking {}.",
             "a photo of a person with a {} facial expression.", "an expression of {}."]
WORDS = {"neutral": "neutral", "happy": "happy", "surprise": "surprised", "sad": "sad", "angry": "angry",
         "disgust": "disgusted", "fear": "fearful", "contempt": "contemptuous", "calm": "calm"}


def test_split(name, fold):
    ds = FaceSet(name)
    m = ds.meta
    return ds.subset((m.split == "test").values if name == "ferplus" else (m.fold == fold).values)


def frames_of(ds, rows, T=8):
    fr = np.tile(np.arange(T) * (16 // T), (len(rows), 1)) if ds.video else np.zeros((len(rows), 1), int)
    return ds.fetch(rows, fr)  # (B,T,H,W,3) uint8


@torch.no_grad()
def run_clip(ds, path=f"{_data_root}/models/clip-vit-base-patch16"):
    from transformers import CLIPModel, CLIPTokenizer
    model = CLIPModel.from_pretrained(path, torch_dtype=torch.float16).cuda().eval()
    tok = CLIPTokenizer.from_pretrained(f"{_data_root}/models/clip-vit-base-patch16") if os.path.exists(path + "/vocab.json") else None
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    W = []
    for lab in ds.labels:
        t = tok([tp.format(WORDS[lab]) for tp in TEMPLATES], padding=True, return_tensors="pt").to("cuda")
        e = F.normalize(model.get_text_features(**t).float(), dim=-1).mean(0)
        W.append(F.normalize(e, dim=0))
    W = torch.stack(W)
    P = []
    for i in range(0, len(ds), 64):
        rows = np.arange(i, min(i + 64, len(ds)))
        x = frames_of(ds, rows).cuda()
        B, T = x.shape[:2]
        x = x.flatten(0, 1).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
        x = ((x - MEAN.cuda()) / STD.cuda()).half()
        f = F.normalize(model.get_image_features(pixel_values=x).float(), dim=-1).view(B, T, -1).mean(1)
        P.append((100 * F.normalize(f, dim=-1) @ W.t()).softmax(1).cpu())
    return torch.cat(P).numpy()


@torch.no_grad()
def run_qwen2vl(ds, path=f"{_data_root}/models/Qwen2-VL-2B-Instruct"):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen2VLForConditionalGeneration.from_pretrained(path, quantization_config=q, device_map={"": 0}).eval()
    proc = AutoProcessor.from_pretrained(path, min_pixels=224 * 224, max_pixels=224 * 224)
    first = [proc.tokenizer(" " + w if False else w, add_special_tokens=False).input_ids[0] for w in ds.labels]
    assert len(set(first)) == len(first)
    question = ("Which emotion does this person's face express? Options: " + ", ".join(ds.labels) +
                ". Answer with exactly one word from the options.")
    P = []
    for i in range(len(ds)):
        x = frames_of(ds, np.array([i]))[0].numpy()
        imgs = [Image.fromarray(f).resize((224, 224), Image.BICUBIC) for f in x]
        if ds.video:
            content = [{"type": "video"}, {"type": "text", "text": question}]
            msgs = [{"role": "user", "content": content}]
            text = proc.apply_chat_template(msgs, add_generation_prompt=True)
            inp = proc(text=[text], videos=[imgs], return_tensors="pt").to("cuda")
        else:
            content = [{"type": "image"}, {"type": "text", "text": question}]
            text = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
            inp = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
        logits = model(**inp).logits[0, -1].float()
        P.append(logits[first].softmax(0).cpu())
        if i % 500 == 0:
            print(i, len(ds), flush=True)
    return torch.stack(P).numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="clip")
    ap.add_argument("--which", default="ferplus,ravdess,cremad")
    a = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    for name in a.which.split(","):
        folds = [0] if name == "ferplus" else sorted(FaceSet(name).meta.fold.unique())
        res = []
        for f in folds:
            ds = test_split(name, f)
            p = run_clip(ds) if a.model == "clip" else run_qwen2vl(ds)
            r = metrics(p, ds.y, len(ds.labels)); res.append(r)
            print(a.model, name, f, round(r["acc"], 2), round(r["uar"], 2), flush=True)
        out = dict(acc=float(np.mean([r["acc"] for r in res])), uar=float(np.mean([r["uar"] for r in res])), folds=res)
        json.dump(out, open(f"{RUNS}/{a.model}_{name}.json", "w"))
        print("ZS", a.model, name, round(out["acc"], 2), round(out["uar"], 2))
