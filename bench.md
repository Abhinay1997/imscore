## Full Benchmark


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