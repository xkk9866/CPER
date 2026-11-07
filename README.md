# CPER

CPER provides composition-posterior normalization (CPN) for a classifier operating on feature vectors. It predicts the class of a query using unlabeled reference vectors from the same person. It accepts embeddings produced by your own encoder.

## Installation

Clone the repository and install NumPy and PyTorch in your Python environment.

```sh
git clone https://github.com/xkk9866/CPER.git
cd CPER
python -m pip install numpy torch
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

## Using your own model

1. Train the supplied linear, BatchNorm, GELU and linear classifier on labeled embeddings, or load compatible weights using `model.load_state_dict(...)`. Switch the model to evaluation mode.
2. Call `population_stats` with source features of shape (samples, embedding dimension), integer labels from zero to C-1, and one person identifier per sample. Include every class, multiple people and repeated observations within person and class. Use a hidden width of at least C-1.
3. Pass `predict` a tensor of shape (batch, 1 + references, embedding dimension). Each row belongs to one person. The output has shape (batch, classes) in the same class order as the source labels. For a query with no references the classifier supplies the prediction directly.

## Calibration and scale

Supply six stacking coefficients in `theta`: the intercept and slope for each of the shrinkage, batch and source branches, in that order. Each slope multiplies the proposed composition's concentration. The conditional branch has logit zero. Six zeros give equal branch weights. For deployment, obtain these parameters on labeled people held out from model training.

`post_weights` weights the classifier, density and set-anchor evidence. `kappa` scales the population centre covariance. `tau`, `tau_bn` and `tau_src` are positive classifier temperatures. `log_scale` supplies the mean and standard deviation of log personal scale. The module includes inference and population-statistic estimation; trained weights and calibration fitting are supplied by the caller.

Exact marginalization enumerates class-count vectors. Use small reference sets and class counts: with n total clips and C classes there are binomial(n+C-1, C-1) candidate vectors. The in-memory lattice cache retains up to four configurations.
