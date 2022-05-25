# RNN

## online learning


- Truncated BPTT(Truncated Backpropagation Through Time)
  - 2002
  -  unfolds the network only for a fixed number of timesteps
  -  Learning is biased towards short-time dependencies.
- NBT(NoBackTrack)
  - 2015
  - bypasses the need for model sparsity
  - unbiased
  - cannot be applied in a blackbox fashion
  - makes implementation easy
- UORO(Unbiased Online Recurrent Optimization)
  - 2018
  - https://openreview.net/pdf?id=rJQDjk-0b
  - modified NBT


## questions

RNN is equivuilent to UTM(Universial Turing Machine)?
Halting problem?

## LSTM

- inputs
  - the size doesn't need to be the same as of hidden states
- hidden states
  - outputs in case the last layer
- cell states
  - the size is the same as of hidden states
- bidirectional layer
  - hidden states from both directions are concatenated

an example of layer composition:

- input embedding size: 10
- output size = 50
- bidirectional
- n_layers=3
- hidden state size=256

```
layers[0][forward_layer][kernels][i].shape=(10, 256)
layers[0][forward_layer][kernels][f].shape=(10, 256)
layers[0][forward_layer][kernels][c].shape=(10, 256)
layers[0][forward_layer][kernels][o].shape=(10, 256)
layers[0][forward_layer][recurrentKernels][i].shape=(256, 256)
layers[0][forward_layer][recurrentKernels][f].shape=(256, 256)
layers[0][forward_layer][recurrentKernels][c].shape=(256, 256)
layers[0][forward_layer][recurrentKernels][o].shape=(256, 256)
layers[0][forward_layer][bias][i].shape=(256,)
layers[0][forward_layer][bias][f].shape=(256,)
layers[0][forward_layer][bias][c].shape=(256,)
layers[0][forward_layer][bias][o].shape=(256,)
layers[1][backward_layer][kernels][i].shape=(10, 256)
layers[1][backward_layer][kernels][f].shape=(10, 256)
layers[1][backward_layer][kernels][c].shape=(10, 256)
layers[1][backward_layer][kernels][o].shape=(10, 256)
layers[1][backward_layer][recurrentKernels][i].shape=(256, 256)
layers[1][backward_layer][recurrentKernels][f].shape=(256, 256)
layers[1][backward_layer][recurrentKernels][c].shape=(256, 256)
layers[1][backward_layer][recurrentKernels][o].shape=(256, 256)
layers[1][backward_layer][bias][i].shape=(256,)
layers[1][backward_layer][bias][f].shape=(256,)
layers[1][backward_layer][bias][c].shape=(256,)
layers[1][backward_layer][bias][o].shape=(256,)
layers[2][forward_layer][kernels][i].shape=(512, 256)
layers[2][forward_layer][kernels][f].shape=(512, 256)
layers[2][forward_layer][kernels][c].shape=(512, 256)
layers[2][forward_layer][kernels][o].shape=(512, 256)
layers[2][forward_layer][recurrentKernels][i].shape=(256, 256)
layers[2][forward_layer][recurrentKernels][f].shape=(256, 256)
layers[2][forward_layer][recurrentKernels][c].shape=(256, 256)
layers[2][forward_layer][recurrentKernels][o].shape=(256, 256)
layers[2][forward_layer][bias][i].shape=(256,)
layers[2][forward_layer][bias][f].shape=(256,)
layers[2][forward_layer][bias][c].shape=(256,)
layers[2][forward_layer][bias][o].shape=(256,)
layers[3][backward_layer][kernels][i].shape=(512, 256)
layers[3][backward_layer][kernels][f].shape=(512, 256)
layers[3][backward_layer][kernels][c].shape=(512, 256)
layers[3][backward_layer][kernels][o].shape=(512, 256)
layers[3][backward_layer][recurrentKernels][i].shape=(256, 256)
layers[3][backward_layer][recurrentKernels][f].shape=(256, 256)
layers[3][backward_layer][recurrentKernels][c].shape=(256, 256)
layers[3][backward_layer][recurrentKernels][o].shape=(256, 256)
layers[3][backward_layer][bias][i].shape=(256,)
layers[3][backward_layer][bias][f].shape=(256,)
layers[3][backward_layer][bias][c].shape=(256,)
layers[3][backward_layer][bias][o].shape=(256,)
layers[4][forward_layer][kernels][i].shape=(512, 256)
layers[4][forward_layer][kernels][f].shape=(512, 256)
layers[4][forward_layer][kernels][c].shape=(512, 256)
layers[4][forward_layer][kernels][o].shape=(512, 256)
layers[4][forward_layer][recurrentKernels][i].shape=(256, 256)
layers[4][forward_layer][recurrentKernels][f].shape=(256, 256)
layers[4][forward_layer][recurrentKernels][c].shape=(256, 256)
layers[4][forward_layer][recurrentKernels][o].shape=(256, 256)
layers[4][forward_layer][bias][i].shape=(256,)
layers[4][forward_layer][bias][f].shape=(256,)
layers[4][forward_layer][bias][c].shape=(256,)
layers[4][forward_layer][bias][o].shape=(256,)
layers[5][backward_layer][kernels][i].shape=(512, 256)
layers[5][backward_layer][kernels][f].shape=(512, 256)
layers[5][backward_layer][kernels][c].shape=(512, 256)
layers[5][backward_layer][kernels][o].shape=(512, 256)
layers[5][backward_layer][recurrentKernels][i].shape=(256, 256)
layers[5][backward_layer][recurrentKernels][f].shape=(256, 256)
layers[5][backward_layer][recurrentKernels][c].shape=(256, 256)
layers[5][backward_layer][recurrentKernels][o].shape=(256, 256)
layers[5][backward_layer][bias][i].shape=(256,)
layers[5][backward_layer][bias][f].shape=(256,)
layers[5][backward_layer][bias][c].shape=(256,)
layers[5][backward_layer][bias][o].shape=(256,)
layers[6][dense][Weight].shape=(512, 50)
layers[6][dense][bias].shape=(50,)
n_params=3722290, bytes=29778320
```

### weight size

can the size of c in LSTM be different from the output
https://www.quora.com/In-LSTM-how-do-you-figure-out-what-size-the-weights-are-supposed-to-be/answer/Ashutosh-Choudhary-2

### peephole variations

how about feeding c in LSTM to the gates also not just feeding inputs and the previous outputs

## GRU
