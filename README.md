# CPER

CPER provides composition-posterior normalization (CPN) for a classifier operating on feature vectors. It predicts the class of a query using unlabeled reference vectors from the same person. It accepts embeddings produced by your own encoder.

## Project layout

| Directory | Contents |
| --- | --- |
| Repository root | Embedding API, source training, calibration and checkpoint prediction for new data |
| [preprocessing](preprocessing/) | Dataset preparation, clip metadata, face detection and alignment, speech and text preparation |
| [features](features/) | CLIP and face identity feature extraction, temporal pooling and feature conversion |
| [vision](vision/) | Frame loaders, visual networks, training, zero-shot prediction and optional job queues |
| [experiments](experiments/) | Reference CPN implementation, baseline and streaming methods, synthetic generator, suite scheduler, multi-seed runs and paired result aggregation |
| [analysis](analysis/) | Alternative normalization methods, diagnostic probes, job definitions and numerical summaries |

## Running the reference pipeline

Use Python 3.12 with a CUDA-enabled PyTorch installation for GPU workers. Run commands from the repository root. Install the core analysis dependencies with `python -m pip install -r experiments/requirements.txt`. For feature extraction, install the dependencies in `preprocessing/requirements.txt`. The MTCNN and identity-feature paths additionally require a version of `facenet-pytorch` compatible with your PyTorch installation. The optional language-model paths in `vision/` and `features/` use `accelerate`, `peft` and `bitsandbytes`.

Set `EMO_ROOT` to an absolute directory for datasets, model files, embeddings, caches, runs and reports. It defaults to `~/.cper`. For example, in PowerShell use `$env:EMO_ROOT = 'D:/cper-data'`. Source archives and pretrained models are external inputs. The lab archive names and expected subdirectories are listed in `preprocessing/prep_lab.py`. Store CLIP model directories under `models/clip-vit-large-patch14` and `models/clip-vit-base-patch16`. For the YuNet backend place the ONNX detector under `models/yunet/`.

1. Prepare laboratory clips with `python preprocessing/prep_lab.py --which ravdess,cremad,mead,enterface`. Extract both CLIP representations for each dataset, e.g. `python features/extract_clip_cls.py --data ravdess --model l14` and again with `--model b16`.
2. For MELD, unpack the source archive with `python preprocessing/prep_meld_unpack.py`, align faces with `python preprocessing/prep_yunet.py --which meld`, and extract the CLIP representations. For natural speech and text, run `python preprocessing/prep_natural.py --models` followed by the individual options `--meld`, `--shemo`, `--emoz`, `--thai`, and `--thaix` in that order. The Studio transfer preparation requires the Zoom inputs.
3. Check inference using `python experiments/cmn.py --selftest`. Inspect the suite manifest with `python experiments/run_cpn.py --list`. To exercise the file pipeline without downloading recordings, run `python experiments/synth_data.py --name synth_demo` followed by `python experiments/cmn.py --data synth_demo --feat raw --method source --folds 0 --steps 32 --suite lite`.
4. With the full inputs ready, run the commands below in order. Run profiling with exclusive access to the GPU. Set worker counts and per-worker memory fractions to suit the device. Leave output paths at their defaults for the multi-seed extension, which locates the seed-0 manifest and summaries under the shared data root.

```sh
python experiments/run_cpn.py --suites main,ablation,synth,stream,repr --workers 5 --mem-frac 0.12
python experiments/run_cpn.py --suites profile --workers 1
python experiments/aggregate_cpn.py
python experiments/run_cpn_multiseed.py --workers 5 --mem-frac 0.12
```

The multi-seed runner reuses completed seed-0 jobs, trains the two additional seeds and calls the multi-seed aggregator. Numerical summaries are written under `EMO_ROOT/reports`. The standalone embedding API below is for fitting a new classifier. Use the reference pipeline above for the original folds, comparison methods and scoring protocols.

## Installation

Clone the repository and install NumPy and PyTorch in your Python environment.

```sh
git clone https://github.com/xkk9866/CPER.git
cd CPER
python -m pip install -r requirements.txt
```

Run Python from the repository directory to import `cper`. CPU and CUDA tensors are supported. Use float32 and keep the classifier, population statistics and input on the same device.

## Quick start

The following example demonstrates the API with generated inputs and an untrained head.

```python
import numpy as np
import torch
from cper import Source, population_stats, predict

model = Source(D=8, C=3, h=12).eval()
features = torch.randn(60, 8)
labels = np.tile(np.arange(3), 20)
people = np.repeat(np.arange(10), 6)
stats = population_stats(model, features, labels, people)

# Within each row the query comes first, followed by its references.
episodes = torch.randn(2, 3, 8)
probabilities = predict(model, stats, episodes, theta=[0.0] * 6)
print(probabilities.shape)  # torch.Size([2, 3])
```

## Train and apply CPN

Start with embeddings from your chosen encoder. Save a NumPy archive containing `features` (samples by embedding dimension), `labels` (integers starting at zero) and `subjects` (one identity per sample, integer or Unicode). Use training identities only. Include at least four people and enough repeated observations for every class to occur in both the fitting and calibration groups.

```sh
python train.py train.npz model.pt --references 6 --device cuda
python predict.py model.pt episodes.npy probabilities.npy --device cuda
```

Use `--device cpu` on a CPU machine. The training program standardizes features, trains an auxiliary source classifier on a subset of people, estimates temperatures, scale and posterior weights on held-out people, fits the six stacking coefficients, and trains the final source head on all training people. The checkpoint includes feature scaling, population statistics, source parameters and calibration. Use raw encoder embeddings for both commands. For prediction provide one query followed by references from the same person in each row of a three-dimensional array. The saved probabilities have one row per query and one column per class.

Files: `train.py` fits and saves the model, `calibration.py` fits the held-out parameters, `cper.py` implements population statistics and posterior inference, and `predict.py` restores a checkpoint and applies it. This release begins at the embedding interface. Supply embeddings from the same encoder at training and prediction time.

## Using your own model

1. Train the supplied linear, BatchNorm, GELU and linear classifier on labeled embeddings, or load compatible weights using `model.load_state_dict(...)`. Switch the model to evaluation mode.
2. Call `population_stats` with source features of shape (samples, embedding dimension), integer labels from zero to C-1, and one person identifier per sample. Include every class, multiple people and repeated observations within person and class. Use a hidden width of at least C-1.
3. Pass `predict` a tensor of shape (batch, 1 + references, embedding dimension). Each row belongs to one person. The output has shape (batch, classes) in the same class order as the source labels. For a query with no references the classifier supplies the prediction directly.

## Calibration and scale

Supply six stacking coefficients in `theta`: the intercept and slope for each of the shrinkage, batch and source branches, in that order. Each slope multiplies the proposed composition's concentration. The conditional branch has logit zero. Six zeros give equal branch weights. For deployment, obtain these parameters on labeled people held out from model training.

`post_weights` weights the classifier, density and set-anchor evidence. `kappa` scales the population centre covariance. `tau`, `tau_bn` and `tau_src` are positive classifier temperatures. `log_scale` supplies the mean and standard deviation of log personal scale. The training command estimates the calibration parameters and saves them with the trained classifier.

Exact marginalization enumerates class-count vectors. Use small reference sets and class counts: with n total clips and C classes there are binomial(n+C-1, C-1) candidate vectors. The in-memory lattice cache retains up to four configurations.
