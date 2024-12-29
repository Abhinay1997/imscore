## Full Benchmark

| Model                  |  HPD v2 Accuracy  | ImageReward Accuracy |   Pickapicv2 Accuracy |   Average |
|:--------------------------------|----------------------:|---------------------------:|-----------------------------:|------------------:|
| pickscore                       |              0.792157 |                   0.667448 |                     0.801887 |          0.753831 |
| hpsv21                          |              0.833464 |                   0.674793 |                     0.693396 |          0.733884 |
| mpsv1                           |              0.838562 |                   0.677762 |                     0.650943 |          0.722423 |
| pickscore-siglip                |              0.771438 |                   0.646195 |                     0.693396 |          0.703676 |
| imreward-overall_rating-siglip*  |              0.770523 |                   0.654008 |                     0.589623 |          0.671385 |
| imreward                        |              0.740131 |                   0.657915 |                     0.608491 |          0.668846 |
| imreward-fidelity_rating-siglip* |              0.739869 |                   0.636037 |                     0.570755 |          0.648887 |
| imreward-overall_rating-dinov2*  |              0.701765 |                   0.623222 |                     0.568396 |          0.631128 |
| ava-rating-siglip-sampled-True*  |              0.742092 |                   0.573996 |                     0.566038 |          0.627375 |
| imreward-overall_rating-clip*   |              0.720784 |                   0.601031 |                     0.558962 |          0.626926 |
| laion-aesthetic*  |              0.736013 |                   0.566807 |                     0.551887 |          0.618236 |
| pickscore-clip                  |              0.594641 |                   0.586185 |                     0.665094 |          0.615307 |
| imreward-fidelity_rating-dinov2* |              0.66098  |                   0.60572  |                     0.549528 |          0.605409 |
| imreward-fidelity_rating-clip*  |              0.662876 |                   0.614158 |                     0.535377 |          0.604137 |
| clipscore                       |              0.626078 |                   0.571652 |                     0.606132 |          0.601287 |
| ava-rating-clip-sampled-True*  |              0.708301 |                   0.540241 |                     0.554245 |          0.600929 |
| ava-rating-dinov2-sampled-True* |              0.661765 |                   0.55493  |                     0.558962 |          0.591886 |

(*): pixel only scorers

### Benchmarking methodology

In a preference dataset, each sample consists of **(1) a prompt, (2) a list of images, and (3) ground truth ranking of the images**.
A preference scorer predicts a "score" for each image in the list and the images are ranked based on the predicted scores.
Given predicted rankings and ground truth rankings, we take unique pairwise comparisons and calculate the accuracy of the model.

**Example**
```
- Ground truth ranking A > B > C
- Model predicted ranking A > C > B
- Unique Pairwise comparisons: (A, B), (A, C), (B, C)
- Number of correct pairwise comparisons: 2 (A > B, A > C)
- Number of incorrect pairwise comparisons: 1 (C > B)
- Accuracy: 2 / 3
```

In case there are ties in the ground truth ranking, we discard the comparison.

## Aesthetic Scorer Benchmarks

Below are the benchmarks for various aesthetic scorers on AVA and ImageReward datasets. 

| Dataset | Model | Performance (MAE) |
| --- | --- | --- |
| ImageReward (overall rating) | imreward-overall_rating-siglip | 1.147 |
| ImageReward (overall rating) | imreward-overall_rating-clip | 1.213 |
| ImageReward (overall rating) | imreward-overall_rating-dinov2 | 1.111 |
| ImageReward (aesthetic rating) | imreward-fidelity_rating-siglip | 0.857 |
| ImageReward (aesthetic rating) | imreward-fidelity_rating-clip | 0.859 |
| ImageReward (aesthetic rating) | imreward-fidelity_rating-dinov2 | 0.845 |
| AVA | ava-rating-clip-sampled-True | 0.228 |
| AVA | ava-rating-clip-sampled-False | 0.236 |
| AVA | ava-rating-siglip-sampled-True | 0.292 |
| AVA | ava-rating-siglip-sampled-False | 0.276 |
| AVA | ava-rating-dinov2-sampled-True | 0.264 |
| AVA | ava-rating-dinov2-sampled-False | 0.265 |
| LAION | laion-aesthetic | 0.321 |
| ??? | ShadowAesthetic v2 | ??? |

Note : Imreward uses 1-7 likert scale, AVA uses 1-10 likert scale, Shadow Aesthetic v2 is an anime aesthetic scorer with ViT backbone, training recipe and objective is not disclosed. For AVA, there's no official test set, so I've created my own split. The performance comparison between LAION aesthetic scorer and other scorers is not an apple-to-apple comparison.