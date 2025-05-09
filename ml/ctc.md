# Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Network

Plain RNN
- inputs require to be pre-segmented
- outputs require to be post-processed

HMM + RNN

CTC loss
- can be used as a loss function on top of various RNNs

## Glossary

- $\mathcal{Z}$
  - labelling
  - e.g. "hello", "cat"
- $\mathcal{X}$
  - $(\mathbb{R}^m)^*$
  - set of sequence of m dimensional vectors
  - an element is an instance of raw input e.g. spectrogram for speech recognition "Hello"
- $\mathcal{y} = \mathcal{N}_w(\mathcal{x})$
  - sequence of RNN outputs
  - e.g. for "$aa(blank)ab$", $\mathcal{y}$ can be like below

$$
\begin{bmatrix}
    \begin{pmatrix}0.85\\0\\0.4\\\vdots\\0.11\end{pmatrix}
    \begin{pmatrix}0.90\\0.01\\0.02\\\vdots\\0.07\end{pmatrix}
    \begin{pmatrix}0.24\\0\\0.01\\\vdots\\0.75\end{pmatrix}
    \begin{pmatrix}0.01\\0.92\\0.05\\\vdots\\0\end{pmatrix}
\end{bmatrix}
$$

- $\mathcal{z} = (z_1, z_2, ..., z_U)$
- $\mathcal{x} = (x_1, x_2, ..., x_T)$

- $m$
  - dimension of single input vector $\mathcal{x}$
- $n$
  - dimension of single output vector $\mathcal{y}$
- $\mathcal{D}_{\mathcal{X}\times\mathcal{Z}}$
  - whole data space
- $\mathcal{S}$
  - set of sampled examples $(x,z)$
- $L$
  - set of alphabets
  - e.g. $\lbrace{a,b,c,d,...,z\rbrace}$
- $L' = L \cup \lbrace blank \rbrace$
  - e.g. $\lbrace{a,b,c,d,...,z,(blank)\rbrace}$
- $l$
  - a labelling
- $\pi$
  - a path on the forward backward algorithm table

## Expressions

### (2)
$$
p(\pi|x) = \prod_{t=1}^{x}{y_{\pi_t}^k},\quad \forall \pi \in L'^{T} \tag{4}
$$
$$
h(\mathcal{x}) = \mathop{\arg\,\max}\limits_{\pi\in\mathcal{B}^{-1}(l)}{p(\pi|\mathcal{x})}
$$

