# Imscore

Easily load reward models in the wild.

## Installation

```bash
pip install imscore
```

## Usage

```python
from imscore.aesthetic.model import ShadowAesthetic, CLIPAestheticScorer, SiglipAestheticScorer, Dinov2AestheticScorer
from imscore.hps.model import HPSv2
from imscore.mps.model import MPS
from imscore.preference.model import SiglipPreferenceScorer, CLIPPreferenceScorer
from imscore.pickscore.model import PickScore

import torch
import numpy as np
from PIL import Image
from einops import rearrange


model = ShadowAesthetic() # https://huggingface.co/shadowlilac/aesthetic-shadow-v2
model = PickScore() # https://github.com/yuvalkirstain/PickScore
model = MPS.from_pretrained("RE-N-Y/mpsv1") # https://github.com/Kwai-Kolors/MPS
model = HPSv2.from_pretrained("RE-N-Y/hpsv21") # https://github.com/tgxs002/HPSv2
model = LAIONAestheticScorer.from_pretrained("RE-N-Y/laion-aesthetic") # https://github.com/christophschuhmann/improved-aesthetic-predictor

# pixel only aesthetic scorers trained on imreward dataset
model = Dinov2AestheticScorer.from_pretrained("RE-N-Y/imreward-overall_rating-dinov2") # 
model = Dinov2AestheticScorer.from_pretrained("RE-N-Y/imreward-fidelity_rating-dinov2")
model = CLIPAestheticScorer.from_pretrained("RE-N-Y/ava-rating-clip-sampled-True")

# multimodal (pixels + text) preference scorers trained on PickaPicv2 dataset 
model = SiglipPreferenceScorer.from_pretrained("RE-N-Y/pickscore-siglip") # ~ 84.4% alignment with human preference
model = CLIPPreferenceScorer.from_pretrained("RE-N-Y/pickscore-clip") # ~ 83.5% alignment with human preference


prompts = "a photo of a cat"
pixels = Image.open("cat.jpg")
pixels = np.array(pixels)
pixels = rearrange(torch.tensor(pixels), "h w c -> 1 c h w") / 255.0

# prompts and pixels should have the same batch dimension
# pixels should be in the range [0, 1]
# score == logits
score = model.score(pixels, prompts) # full differentiable reward
```