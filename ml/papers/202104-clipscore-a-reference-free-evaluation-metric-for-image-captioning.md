# CLIPScore: A Reference-free Evaluation Metric for Image Captioning

- https://arxiv.org/abs/2104.08718
- ACL 2021

## 1 Introduction

- CLIPScore or CLIP-S
  - reference-free evaluation method
  - evaluates generated captions from images
  - limitations
    - bad at some domains e.g. news
- reference-based evaluation methods
  - expensive
  - often insufficient
    - there can be adversarial examples

## 2 Related work

(skipped)

## 3 CLIPScore

### More details

CLIP

- a cross-modal retrieval model
- trained on 400M (image, caption) pairs
  - gathered from 500K search queries
  - for each query, up to 20K (image caption) pairs were collected
- we use the model based on `ViT-B/32`
  - 12 transformer layers
  - 86M parameters
- a vocab of 49K BPE tokens
- outputs 512 dimension vector
- trained on InfoNCE loss
  - NCE stands for Noise-Contrastive Estimation

### Evaluating caption generations with CLIP

$$
\operatorname{CLIP-S}(\mathbf{c}, \mathbf{v}) = w * \max(\cos(\mathbf{c}, \mathbf{v}), 0)
$$

- $\mathbf{c}$
  - a CLIP embedding from the generated candidate caption 
- $\mathbf{v}$
  - a CLIP embedding from the image
- $\cos(\cdot, \cdot)$
  - cosine similarity
- $w=2.5$

### RefCLIPScore

$$
\operatorname{RefCLIP-S}(\mathbf{c}, \mathbf{R}, \mathbf{v}) = \operatorname{H-mean}\left(\operatorname{CLIP-S}(\mathbf{c}, \mathbf{v}),  \max\left(\max\limits_{\mathbf{r} \in \mathbf{R}}\cos(\mathbf{c}, \mathbf{r}), 0\right)\right)
$$

- $\mathbf{R}$
  - a set of CLIP embeddings corresponding to the reference captions
- taking account of
  - how much is the candidate caption close to the source image
  - how much is the candidate caption close to the reference captions

## 4 Benchmark captioning evaluations

(skipped)

## 5 Case studies using CLIPScore

(skipped)

## 6 Conclusion

