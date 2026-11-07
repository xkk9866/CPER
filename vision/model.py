"""Model components.

FaceEncoder        CLIP ViT-B/16 -> 17 region tokens (CLS + 4x4 grid) per frame
NeutralField       denoising score field learned ONLY on neutral faces; its Tweedie estimate
                   D(z, s) = E[neutral face | z observed at noise level s] is the counterfactual neutral
DeltaTokenizer     fuses appearance Z and multi-scale displacements E_k = Z - D(Z, s_k) into M tokens/frame
LLMReasoner        4-bit Qwen2.5 + LoRA; scores label words given [visual tokens; question with options]
TinyReasoner       small transformer + label-embedding head (non-LLM control)
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from transformers import CLIPVisionModel

CLIP_PATH = f"{_data_root}/models/clip-vit-base-patch16"


class FaceEncoder(nn.Module):
    def __init__(self, n_train_blocks=4, grid=4, path=CLIP_PATH):
        super().__init__()
        self.vm = CLIPVisionModel.from_pretrained(path).vision_model
        self.grid = grid
        self.k = len(self.vm.encoder.layers) - n_train_blocks
        for p in self.vm.parameters():
            p.requires_grad = False
        for blk in self.vm.encoder.layers[self.k:]:
            for p in blk.parameters():
                p.requires_grad = True
        if n_train_blocks > 0:
            for p in self.vm.post_layernorm.parameters():
                p.requires_grad = True
        self.dim = self.vm.config.hidden_size

    def forward(self, x):
        vm = self.vm
        with torch.no_grad():
            h = vm.pre_layrnorm(vm.embeddings(x))
            for blk in vm.encoder.layers[:self.k]:
                h = blk(h, None, None)[0]
        for blk in vm.encoder.layers[self.k:]:
            h = blk(h, None, None)[0]
        h = vm.post_layernorm(h)
        cls, p = h[:, :1], h[:, 1:]
        g = int(p.shape[1] ** 0.5)
        p = F.adaptive_avg_pool2d(p.transpose(1, 2).reshape(len(h), -1, g, g), self.grid).flatten(2).transpose(1, 2)
        return torch.cat([cls, p], 1)  # (N, 1+grid^2, D)


class SigmaEmbed(nn.Module):
    def __init__(self, d, nf=16):
        super().__init__()
        self.register_buffer("freq", torch.randn(nf) * 2.0)
        self.mlp = nn.Sequential(nn.Linear(2 * nf, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, s):
        a = torch.log(s)[:, None] * self.freq[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], 1))


class NeutralField(nn.Module):
    """EDM-preconditioned token denoiser trained by denoising score matching on neutral faces.

    For any face Z, D(Z, s) is (by Tweedie) the posterior-mean neutral face that would explain Z at
    noise scale s; Z - D(Z, s) = -s^2 * grad log p_neutral,s(Z) is the multi-scale *neutral score*.
    """

    def __init__(self, d=256, R=17, layers=3, heads=4):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, R, d) * 0.02)
        self.semb = SigmaEmbed(d)
        self.inp = nn.Linear(d, d)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, 4 * d, 0.0, activation="gelu", batch_first=True, norm_first=True) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.register_buffer("sigma_data", torch.tensor(0.5))

    def forward(self, x, s):
        sd = self.sigma_data
        s4 = s.view(-1, 1, 1)
        c_in = 1 / (s4 ** 2 + sd ** 2).sqrt()
        c_skip = sd ** 2 / (s4 ** 2 + sd ** 2)
        c_out = s4 * sd / (s4 ** 2 + sd ** 2).sqrt()
        h = self.inp(c_in * x) + self.pos + self.semb(s)[:, None]
        for b in self.blocks:
            h = b(h)
        return c_skip * x + c_out * self.out(self.norm(h))

    def dsm_loss(self, z, p_mean=-1.2, p_std=1.2):
        """z: (N,R,d) detached neutral faces. EDM loss weighting."""
        with torch.no_grad():  # running estimate of the neutral data scale (per-dim std)
            self.sigma_data.mul_(0.99).add_(0.01 * z.float().std(0).mean())
        s = torch.exp(torch.randn(len(z), device=z.device) * p_std + p_mean) * self.sigma_data
        n = torch.randn_like(z) * s.view(-1, 1, 1)
        sd = self.sigma_data
        w = (s ** 2 + sd ** 2) / (s * sd) ** 2
        return (w.view(-1, 1, 1) * (self(z + n, s) - z) ** 2).mean()

    def counterfactual(self, z, rel_sigmas):
        """Counterfactual neutrals at several noise scales, without letting task gradients reach the field."""
        params = {k: v.detach() for k, v in self.named_parameters()}
        bufs = dict(self.named_buffers())
        outs = []
        for r in rel_sigmas:
            s = torch.full((len(z),), float(r), device=z.device) * self.sigma_data
            outs.append(functional_call(self, {**params, **bufs}, (z, s)))
        return torch.stack(outs, 1)  # (N,K,R,d)


class DeltaTokenizer(nn.Module):
    """Region tokens [Z ; E_1..E_K] -> one self-attention layer -> M query tokens per frame."""

    def __init__(self, d=256, R=17, K=3, M=4, heads=4, use_z=True, use_e=True):
        super().__init__()
        self.use_z, self.use_e, self.K = use_z, use_e, K
        n_in = (1 if use_z else 0) + (K if use_e else 0)
        self.fuse = nn.Linear(n_in * d, d)
        self.pos = nn.Parameter(torch.randn(1, R, d) * 0.02)
        self.sa = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, activation="gelu", batch_first=True, norm_first=True)
        self.q = nn.Parameter(torch.randn(1, M, d) * 0.02)
        self.ca = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln = nn.LayerNorm(d)

    def forward(self, Z, E):
        parts = ([Z] if self.use_z else []) + ([E[:, k] for k in range(self.K)] if self.use_e else [])
        h = self.fuse(torch.cat(parts, -1)) + self.pos
        h = self.sa(h)
        q = self.q.expand(len(h), -1, -1)
        return self.ln(q + self.ca(q, h, h, need_weights=False)[0])  # (N,M,d)


class TinyReasoner(nn.Module):
    """Temporal transformer over frame tokens, classifies by similarity to learned label-word embeddings."""

    def __init__(self, d=256, T_max=16, layers=2, heads=4, vocab=None, head="tf"):
        super().__init__()
        self.head = head
        if head == "mil":  # M3DFEL-style: frame instances -> gated attention pooling (no temporal transformer)
            self.att = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
            self.gate = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
        if head == "cliptext":  # DFER-CLIP-style: classifier from frozen CLIP text embeddings + learnable projection
            self.text = nn.Linear(512, d)
            self.text_bank = {}
        self.tpos = nn.Parameter(torch.randn(1, T_max, 1, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, activation="gelu", batch_first=True, norm_first=True), layers)
        self.ln = nn.LayerNorm(d)
        self.words = nn.ParameterDict({w: nn.Parameter(torch.randn(d) * 0.02) for w in vocab})
        self.scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def text_emb(self, labels, device):
        return clip_text_embeddings(labels, device)

    def forward(self, tok, labels):
        B, T, M, d = tok.shape
        if self.head == "mil":
            inst = tok.mean(2)  # (B,T,d) frame instances
            a = (self.att(inst) * 1.0).softmax(1)
            f = F.normalize(self.ln((a * inst * self.gate(inst)).sum(1)), dim=-1)
        else:
            h = (tok + self.tpos[:, :T]).reshape(B, T * M, d)
            h = self.tf(torch.cat([self.cls.expand(B, -1, -1), h], 1))
            f = F.normalize(self.ln(h[:, 0]), dim=-1)
        if self.head == "cliptext":
            W = F.normalize(self.text(self.text_emb(labels, tok.device)), dim=-1)
        else:
            W = F.normalize(torch.stack([self.words[w] for w in labels]), dim=-1)
        return self.scale.exp() * f @ W.t()


_CLIP_TEXT = {}


def clip_text_embeddings(labels, device):
    """Prompt-ensembled CLIP text embeddings for label words (cached, frozen)."""
    key = tuple(labels)
    if key not in _CLIP_TEXT:
        from transformers import CLIPModel, CLIPTokenizer
        m = CLIPModel.from_pretrained(CLIP_PATH).eval()
        tk = CLIPTokenizer.from_pretrained(CLIP_PATH)
        temps = ["a photo of a {} face.", "a face showing {}.", "a person with a {} facial expression."]
        with torch.no_grad():
            E = [F.normalize(m.get_text_features(**tk([t.format(w) for t in temps], padding=True, return_tensors="pt")), dim=-1).mean(0) for w in labels]
        _CLIP_TEXT[key] = torch.stack(E)
    return _CLIP_TEXT[key].to(device)


class LLMReasoner(nn.Module):
    """Quantized LLM reads projected visual tokens and scores the first token of each option word."""

    def __init__(self, path, d=256, T_max=16, lora_r=16, bits=4):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        self.tok = AutoTokenizer.from_pretrained(path)
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True) if bits == 4 else BitsAndBytesConfig(load_in_8bit=True)
        llm = AutoModelForCausalLM.from_pretrained(path, quantization_config=q, torch_dtype=torch.bfloat16, device_map={"": 0})
        llm = prepare_model_for_kbit_training(llm, use_gradient_checkpointing=False)
        cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
        self.llm = get_peft_model(llm, cfg)
        H = llm.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(d, H), nn.GELU(), nn.Linear(H, H))
        self.tpos = nn.Parameter(torch.randn(1, T_max, 1, d) * 0.02)
        self.emb = self.llm.get_input_embeddings()
        self._cache = {}

    def _ids(self, text):
        return self.tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0]

    def prompt(self, labels, video):
        key = (tuple(labels), video)
        if key not in self._cache:
            pre = "<|im_start|>system\nYou are an expert in facial expression analysis.<|im_end|>\n<|im_start|>user\n" + ("Face video frames: " if video else "Face image: ")
            post = ("\nWhich emotion does this face express? Options: " + ", ".join(labels) + ".<|im_end|>\n<|im_start|>assistant\nThe emotion is")
            first = [self._ids(" " + w)[0].item() for w in labels]
            assert len(set(first)) == len(first), labels
            self._cache[key] = (self._ids(pre), self._ids(post), torch.tensor(first))
        return self._cache[key]

    def forward(self, tok, labels, video):
        B, T, M, d = tok.shape
        dev = tok.device
        vis = self.proj((tok + self.tpos[:, :T]).reshape(B, T * M, d))
        pre, post, first = self.prompt(labels, video)
        e_pre, e_post = self.emb(pre.to(dev))[None], self.emb(post.to(dev))[None]
        x = torch.cat([e_pre.expand(B, -1, -1), vis.to(e_pre.dtype), e_post.expand(B, -1, -1)], 1)
        out = self.llm(inputs_embeds=x, use_cache=False, logits_to_keep=1)
        return out.logits[:, -1, first.to(dev)].float()


class RandomEffects(nn.Module):
    """Differentiable random-effects inference over a subject's observations (unrolled EM).

    h = mu_s + f_y + eps,  mu_s ~ N(mu_0, tau^2),  eps ~ N(0, sigma^2).
    tau^2 / sigma^2 are empirical-Bayes moment estimates from labelled training subjects (running stats).
    mode: 'eb' (ours) | 'learned' (precisions by backprop) | 'mean' (mu = mean of references) | 'none'
    """

    def __init__(self, D, C, iters=3, mode="eb", estep=True):
        super().__init__()
        self.f = nn.Parameter(torch.randn(C, D) * 0.02)
        self.mu0 = nn.Parameter(torch.zeros(D))
        self.log_t = nn.Parameter(torch.zeros(D))
        self.b = nn.Parameter(torch.zeros(C))
        self.log_l0 = nn.Parameter(torch.zeros(D)); self.log_le = nn.Parameter(torch.zeros(D))
        self.register_buffer("tau2", torch.ones(D)); self.register_buffer("sig2", torch.ones(D))
        self.iters, self.mode, self.estep = iters, mode, estep

    def logits(self, h, mu):
        r = h[:, :, None] - mu[:, None, None] - self.f[None, None]
        return -0.5 * (self.log_t.exp() * r ** 2).mean(-1) * 64 + self.b

    @torch.no_grad()
    def update_moments(self, h, y):
        if h.shape[1] < 2:
            return
        res = h.float() - self.f[y].float()
        m = res.mean(1)
        sig2 = ((res - m[:, None]) ** 2).sum((0, 1)) / (res.shape[0] * (res.shape[1] - 1))
        tau2 = ((m - self.mu0.float()) ** 2).mean(0) - sig2 / res.shape[1]
        self.sig2.lerp_(sig2, 0.05); self.tau2.lerp_(tau2.clamp_min(1e-4), 0.05)

    def forward(self, hq, hc):
        """hq (B,D) query, hc (B,K,D) references -> subject effect mu (B,D), logits (B,1+K,C)."""
        h = torch.cat([hq[:, None], hc], 1).float()
        n, K = h.shape[1], hc.shape[1]
        if self.mode in ("none", "icl"):
            mu = torch.zeros_like(hq).float()
        elif self.mode == "mean":
            mu = hc.float().mean(1) if K > 0 else self.mu0[None].expand(len(h), -1)
        else:
            lam0, lame = (1 / self.tau2, 1 / self.sig2) if self.mode == "eb" else (self.log_l0.exp(), self.log_le.exp())
            e = torch.zeros_like(h)
            for _ in range(self.iters if self.estep else 1):
                mu = (lam0 * self.mu0 + lame * (h - e).sum(1)) / (lam0 + n * lame)
                if self.estep:
                    e = self.logits(h, mu).softmax(-1) @ self.f
        return mu, self.logits(h, mu)


class ExpressionModel(nn.Module):
    """Encoder -> random-effects subject inference -> subject-normalised tokens -> reasoner."""

    def __init__(self, cfg, vocab, n_classes):
        super().__init__()
        self.cfg = cfg
        d, R = cfg.d, (cfg.R or 1 + cfg.grid ** 2)
        self.R = R
        self.enc = None if cfg.cached else FaceEncoder(cfg.train_blocks, cfg.grid)
        self.proj = nn.Linear(768 if cfg.cached else self.enc.dim, d)
        self.re = RandomEffects(R * d, n_classes, mode=cfg.re, estep=bool(cfg.estep))
        self.tokzr = DeltaTokenizer(d, R, 1, cfg.M, use_z=True, use_e=False)
        if cfg.reasoner == "llm":
            self.reasoner = LLMReasoner(cfg.llm_path, d, lora_r=cfg.lora_r, bits=cfg.bits)
        else:
            self.reasoner = TinyReasoner(d, vocab=vocab, head=cfg.head)
        if cfg.re == "icl":
            self.ctx_role = nn.Parameter(torch.randn(1, 1, 1, d) * 0.02)

    def features(self, x):
        h = x if self.enc is None else self.enc(x)
        return F.layer_norm(self.proj(h), (self.cfg.d,))

    def forward(self, x, B, T, labels, video, hc=None):
        Z = self.features(x).view(B, T, self.R, -1)
        hq = Z.mean(1).flatten(1)
        if hc is None:
            hc = hq.new_zeros(B, 0, hq.shape[1])
        mu, ll = self.re(hq, hc)
        Zn = Z - mu.view(B, 1, self.R, -1).to(Z.dtype)
        tok = self.tokzr(Zn.flatten(0, 1), None).view(B, T, self.cfg.M, -1)
        if self.cfg.re == "icl" and hc.shape[1] > 0:  # absolute context tokens appended as extra "frames"
            K = hc.shape[1]
            ct = self.tokzr(hc.reshape(B * K, self.R, -1).to(Z.dtype), None).view(B, K, self.cfg.M, -1)
            tok = torch.cat([tok, ct + self.ctx_role.to(ct.dtype)], 1)
        logits = self.reasoner(tok, labels, video) if self.cfg.reasoner == "llm" else self.reasoner(tok, labels)
        return logits, ll, hq
