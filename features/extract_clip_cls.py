"""Stream frozen CLIP clip-level CLS means, without storing unused patch tokens."""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import argparse, os, time
E = _data_root
os.environ.setdefault('HF_HOME', f'{E}/cache/hf')
os.environ.setdefault('TEMP', f'{E}/tmp')
os.environ.setdefault('TMP', f'{E}/tmp')
import numpy as np
import torch
from transformers import CLIPVisionModel
from extract_feats import to_input

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='meld')
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--model', default='l14', choices=['l14', 'b16'])
    ap.add_argument('--mem_frac', type=float, default=0.30)
    a = ap.parse_args()
    mdir = {'l14': 'clip-vit-large-patch14', 'b16': 'clip-vit-base-patch16'}[a.model]
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    model = CLIPVisionModel.from_pretrained(f'{E}/models/{mdir}', torch_dtype=torch.float16).cuda().eval()
    src = np.load(f'{E}/datasets/proc/{a.data}_faces.npy', mmap_mode='r')
    os.makedirs(f'{E}/features/clip', exist_ok=True)
    outpath = f'{E}/features/clip/{a.data}_{a.model}.npy'
    partial = outpath + '.partial.npy'
    progress = outpath + '.progress'
    start = int(open(progress).read()) if os.path.exists(progress) and os.path.exists(partial) else 0
    out = np.lib.format.open_memmap(partial, mode='r+' if start else 'w+', dtype=np.float32, shape=(len(src), model.config.hidden_size))
    clips = max(1, a.bs // src.shape[1]); t = time.time()
    with torch.inference_mode():
        for i in range(start, len(src), clips):
            faces = src[i:i+clips]
            x = to_input(faces.reshape(-1, *src.shape[2:]), model.config.image_size).half()
            v = model(pixel_values=x).pooler_output.float().reshape(len(faces), src.shape[1], -1).mean(1)
            out[i:i+clips] = v.cpu().numpy()
            if (i-start) % (clips*50) == 0 or i+clips >= len(src):
                out.flush()
                with open(progress, 'w') as f: f.write(str(min(i+clips, len(src))))
                print(f'{a.data}: {min(i+clips,len(src))}/{len(src)} clips, {(time.time()-t)/60:.1f} min', flush=True)
    out.flush(); del out
    os.replace(partial, outpath)
    print('DONE', outpath, flush=True)

if __name__ == '__main__': main()
