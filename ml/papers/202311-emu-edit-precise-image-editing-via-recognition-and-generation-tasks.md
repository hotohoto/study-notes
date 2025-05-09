# Emu Edit: Precise Image Editing via Recognition and Generation Tasks

https://emu-edit.metademolab.com/

- Methods are based on InstructPix2Pix

  - turn an input caption to the edited caption corresponding to a given instruction caption
    - by providing the exemplars to GPT-3
  - "approximates" masks for editing area by using cross attention maps
  - generate edited images by the edited caption

- the diffusion model is based on Emu

  

## Data preparation

- sample exemplars for data diversity
- don't assume the word-to-word alignment
  - e.g. "a cat riding a bicycle" and "a cat riding a car"

- extract masks before the editing process
- filtering out 70% samples by
  - the task predictor
  - CLIP metrics
  - L1 distance between the input image and the edited image
  - use image detectors
    - presence
    - absence
    - replacement

- final dataset
  - 10M samples 🫤
    - it's a big number but they're automatically generated.

### Instruction generation

- use the dialog optimized 70B Llama2 variant
  - temperature = 0.9
  - top-p = 0.9
  - in-context learning
    - (no finetuning)
- TODO: fig 15, 16

### Image pair generation

#### Grounded precise editing

#### Mask extraction

#### Region-based editing tasks

- local/texture
- add
- remove
- background

#### Free-form editing tasks

- global
- style
- text editing

#### Vision tasks

- detect/segment
- color
- image-to-image translation





## Training

- use 16 tasks ⭐
  - (using many tasks are helpful)
- $v_i$
  - $i$th task embedding 

- $\hat{y}=(c_I, c_T, x, i)$
- $c_I$
  - input image

- $c_T$
  - input instruction

- $x$
  - target image

- $i$
  - task index

- finetune the diffusion model and learn task embeddings

$$
\min\limits_{\theta, v_1, ..., v_k} \mathbb{E}_{\hat{y},\epsilon,t}
\left[
	\Vert
	\epsilon - \epsilon_\theta(z_t, t, E(c_I), c_T, v_i)
	\Vert_2^2
\right]
\tag{2}
$$

## Inference

- task inversion ⭐
  - the model parameters are fixed
  - optimize a new task embedding $v_\text{new}$
  - with just a few examples of a new task


$$
\min\limits_{v_\text{new}} \mathbb{E}_{y,\epsilon,t}
\left[
	\Vert
	\epsilon - \epsilon_\theta(z_t, t, E(c_I), c_T, v_\text{new})
	\Vert_2^2
\right]
\tag{3}
$$



- sequential edit thresholding ⭐
  - to avoid artifacts by the accumulated numerical errors
  - use L1 distance $d$ between two pixels
    - $d = \Vert c_I^{S} - c_I^{S+1} \Vert_1$
  - get $\bar{d}$ by passing $d$ though a low pass filter
  - keep the original value if the $\bar{d}$ is smaller than $\alpha = 0.03$
  
  
  $$
  c_I^{s+1}= \begin{cases}c_I^s & \text { if } \bar{d}<\alpha \\ c_I^{s+1} & \text { otherwise }\end{cases}
  \tag{4}
  $$
  
  
