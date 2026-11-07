"""Semantic label prototypes from a 4-bit quantised LLM (no gradient, computed once).

For each emotion word w two vectors in the LLM's output space are stored:
  unemb : the output-layer (lm_head) row of the first token of " w"   -> what the LLM scores to *say* w
  state : the final-layer, final-norm hidden state after a short prompt that ends with w, i.e. the vector
          the output layer reads when the LLM has just described that expression
-> $EMO_ROOT/features/llm/{model}.npz  (words, unemb (W,H), state (W,H))"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import sys, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
WORDS = ["neutral", "calm", "happy", "sad", "angry", "fear", "disgust", "surprise"]
for name in sys.argv[1:]:
    p = f"{_data_root}/models/{name}"
    tk = AutoTokenizer.from_pretrained(p)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    m = AutoModelForCausalLM.from_pretrained(p, quantization_config=q, device_map={"": 0}).eval()
    U = m.get_output_embeddings().weight
    un, st = [], []
    for w in WORDS:
        un.append(U[tk(" " + w, add_special_tokens=False).input_ids[0]].detach().float().cpu().numpy())
        msg = f"<|im_start|>user\nDescribe how a person's face looks when the facial expression is {w}.<|im_end|>\n<|im_start|>assistant\nA {w} facial expression"
        with torch.no_grad():
            h = m(**tk(msg, return_tensors="pt").to(0), output_hidden_states=True).hidden_states[-1][0, -1]
        st.append(h.detach().float().cpu().numpy())
    np.savez(f"{_data_root}/features/llm/{name}.npz", words=np.array(WORDS), unemb=np.stack(un), state=np.stack(st))
    print(name, np.stack(st).shape, flush=True)
    del m; torch.cuda.empty_cache()
