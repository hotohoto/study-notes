# Rethinking and Improving Relative Position Encoding for Vision Transformer

- https://arxiv.org/abs/2107.14222
- ICCV 2021
- image relative position encoding (iRPE)

## 1 Introduction

- relative position encoding has been verified to be effective in NLP

## 2 Background

### 2.1 Self-attention

### 2.2 Position encoding

##### Absolute position encoding

##### Relative position encoding

## 3 Method

### 3.1 Previous relative position encoding methods

##### Shaw's RPE

- 1D
- the original RPE
- clip the distance beyond $k$
  - to reduce the number of parameters

##### RPE in Transformer-XL

- 1D

##### Huang's RPE

- 1D

##### RPE in SASA

- 2D

##### PRE in Axial-Deeplab

- 2D

### 3.2 Proposed relative position encoding methods

##### Bias mode and contextual mode

<img src="./assets/image-20230512194835632.png" alt="image-20230512194835632" style="zoom:67%;" />

##### A piecewise index function

<img src="./assets/image-20230512195037924.png" alt="image-20230512195037924" style="zoom:67%;" />

##### 2D relative position calculation

###### Euclidean method

<img src="./assets/image-20230512195801074.png" alt="image-20230512195801074" style="zoom:67%;" />

###### Quantization method

<img src="./assets/image-20230512195837715.png" alt="image-20230512195837715" style="zoom:67%;" />

###### Cross method

<img src="./assets/image-20230512195904963.png" alt="image-20230512195904963" style="zoom:67%;" />

###### Product method

<img src="./assets/image-20230512195940699.png" alt="image-20230512195940699" style="zoom:67%;" />

##### An efficient implementation

## 4 Experiments

### 4.1 Implementation details

### 4.2 Analysis on relative position encoding

In terms of classification accuracy improvement

- `directed` is better than `undirected`
- `contextual` is better than `bias`
- `unshared` is better than `shared`
  - but the shared version is memory efficient
- `piecewise` is almost the same as `clip`
  - but it seems better at detection tasks
- number of buckets of 50 looks enough.
  - when it comes to `DeiT-S` that has 14x14 feature map
- the product method looks like the best
  - also, in the experiment, when equipped with relative position encoding, the absolute position encoding does not bring any gains
- the efficient implementation suggested in this paper is way better than a naive implementation

### 4.3 Comparison on image classification

### 4.4 comparison on object detection

### 4.5 Visualization

<img src="./assets/image-20230512195130039.png" alt="image-20230512195130039" style="zoom:67%;" />

- in block 0
  - the current patch focus more on its neighborhoods patches
- in block 10 (which represents for higher blocks)
  - the current patch does not.
- in theory, without RPEs, transformer does not explicitly capture locality.
- but RPEs can inject Conv-like inductive bias into transformer in terms of locality

## 5 Related work

##### Transformer

##### Relative position encoding

## 5 Conclusion and remarks

## References

- [18] Self-Attention with Relative Position Representations
  - https://arxiv.org/abs/1803.02155v2

## A Proof of formula 1

## B Proof of formula 2

## Notes

### Code analysis

- piecewise_index(relative_position, alpha, beta, gamma, ...)
  - the shape of relative_position: L, L

- quantize_values()
  - (not in use)

- METHOD
- _METHOD_FUNC
  - _rp_2d_euclidean()
  - _rp_2d_quant()
  - _rp_2d_product()
  - _rp_2d_cross_rows()
  - _rp_2d_cross_cols()
- get_num_buckets()
- iRPE
  - \__init__(..., rpe_config: single rpe config)
    - self.reset_parameters()
      - if self.transposed:
        - if self.mode == "contextual":
          - self.lookup_table_weight = nn.Parameter(torch.zeros(self.num_heads, self.head_dim, self.num_buckets))
      - else:
        - if self.mode == "contextual":
          - self.lookup_table_weight = nn.Parameter(torch.zeros(self.num_heads, self.num_buckets, self.head_dim))
  - forward()
    - self._get_rp_bucket()
      - (doesn't depend on batch or heads but on the width and height)
      - get_bucket_ids_2d()
        - get_bucket_ids_2d_without_skip()
          - pos = get_absolute_positions()
            - prepare ingredient to build coordinates
          - pos1 = pos.view((max_L, 1, 2))
            - all the coordinates
          - pos2 = pos.view((1, max_L, 2))
            - all the coordinates
          - diff = pos1 - pos2
            - build all the combinations of the per-axis distance between coordinates by broadcasting
          - bucket_ids = func(diff, alpha, beta, gamma=gamma)
          - cache bucket ids
    - (we can mask combinations here ⭐)
      - only if it's okay to mask the same combinations over all the batches
      - implementing it here might be a bit redundant if you also want to support masking without iRPE
    - self.forward_rpe_transpose()
      - usually for queries and keys
    - self.forward_rpe_no_transpose()
      - usually for values
- iRPE_Cross
  - \__init__()
    - self.rp_rows = iRPE(\**kwargs, method=METHOD.CROSS_ROWS)
    - self.rp_cols = iRPE(\**kwargs, method=METHOD.CROSS_COLS)

  - forward()
    - sekf.rp_rows(...) + self.rp_cols(...)

- get_single_rpe_config()
- get_rpe_config()
- build_rpe()
