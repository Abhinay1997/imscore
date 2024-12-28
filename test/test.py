from imscore.aesthetic.model import ShadowAesthetic, CLIPAestheticScorer, SiglipAestheticScorer, Dinov2AestheticScorer, LAIONAestheticScorer
from imscore.hps.model import HPSv2
from imscore.mps.model import MPS
from imscore.preference.model import SiglipPreferenceScorer, CLIPPreferenceScorer, CLIPScore
from imscore.pickscore.model import PickScorer

import torch
import numpy as np
from PIL import Image
from einops import rearrange
from loguru import logger


def factory(name:str):
    match name:
        case "ShadowAesthetic":
            return ShadowAesthetic()
        case "CLIPAestheticScorer":
            return CLIPAestheticScorer.from_pretrained("RE-N-Y/ava-rating-clip-sampled-True")
        case "SiglipAestheticScorer":
            return SiglipAestheticScorer.from_pretrained("RE-N-Y/imreward-overall_rating-siglip")
        case "Dinov2AestheticScorer":
            return Dinov2AestheticScorer.from_pretrained("RE-N-Y/imreward-overall_rating-dinov2")
        case "LAIONAestheticScorer":
            return LAIONAestheticScorer.from_pretrained("RE-N-Y/laion-aesthetic")
        case "HPSv2":
            return HPSv2.from_pretrained("RE-N-Y/hpsv21")
        case "MPS":
            return MPS.from_pretrained("RE-N-Y/mpsv1")
        case "SiglipPreferenceScorer":
            return SiglipPreferenceScorer.from_pretrained("RE-N-Y/pickscore-siglip")
        case "CLIPPreferenceScorer":
            return CLIPPreferenceScorer.from_pretrained("RE-N-Y/pickscore-clip")
        case "PickScorer":
            return PickScorer()
        case "CLIPScore":
            return CLIPScore("openai/clip-vit-large-patch14")
        case _:
            raise ValueError(f"model {name} not found")
        
    
def testrun(name:str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = factory(name)
    model.eval()
    model.to(device=device)

    prompts = "a photo of a cat"

    good, bad = Image.open("cat.png"), Image.open("badcat.png")
    good, bad = good.resize((512,512)), bad.resize((512,512))
    good, bad = np.array(good), np.array(bad)
    good = rearrange(torch.tensor(good), "h w c -> 1 c h w") / 255.0
    bad = rearrange(torch.tensor(bad), "h w c -> 1 c h w") / 255.0

    # prompts and pixels should have the same batch dimension
    # pixels should be in the range [0, 1]
    # score == logits

    pixels = torch.cat([good, bad])
    pixels = pixels.to(device=device)
    prompts = [prompts] * 2

    score = model.score(pixels, prompts) # full differentiable reward
    assert score.grad_fn is not None

    return score


if __name__ == "__main__":
    names = [
        "ShadowAesthetic", 
        "CLIPAestheticScorer", 
        "SiglipAestheticScorer", 
        "Dinov2AestheticScorer", 
        "LAIONAestheticScorer", 
        "HPSv2", "MPS",
        "SiglipPreferenceScorer",
        "CLIPPreferenceScorer", 
        "PickScorer",
        "CLIPScore"
    ]

    for name in names:
        logger.info(f"Testing {name}")
        score = testrun(name)
        logger.info(f"Score: {score}")