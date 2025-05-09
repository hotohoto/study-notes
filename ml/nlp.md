# NLP(Natural Language Processing)

## TODO

- https://www.fastcampus.co.kr/data_camp_nlpadv/
- https://blog.floydhub.com/ten-trends-in-deep-learning-nlp/
- https://medium.com/datadriveninvestor/attention-in-rnns-321fbcd64f05
- https://hackernoon.com/chars2vec-character-based-language-model-for-handling-real-world-texts-with-spelling-errors-and-a3e4053a147d
- https://docs.google.com/spreadsheets/d/1pwkvIwf3T1bo2y7aXmSYPN6otlPKJl9kCJHaze0H3KY/edit#gid=697987516

## LLM papers



- Language Models are Few-Shot Learners

  - https://arxiv.org/abs/2005.14165
  - GPT-3
  - 175B parameters
  - 10000 GPUs
  - TODO

- The Power of Scale for Parameter-Efficient Prompt Tuning

  - https://arxiv.org/abs/2104.08691
  - "prompt tuning" to learn "soft prompts"

- LoRA: Low-Rank Adaptation of Large Language Models

  - https://arxiv.org/abs/2106.09685

- Finetuned Language Models Are Zero-Shot Learners

  - https://arxiv.org/abs/2109.01652
  - instruction tuning
  - more useful for larger models
  - make use of pre-existing NLP datasets

- (ChatGPT 3.5)

  - 202203
  - https://en.wikipedia.org/wiki/GPT-3#GPT-3.5

- Training language models to follow instructions with human feedback

  - https://arxiv.org/abs/2203.02155
  - RLHF
  - the term InstructGPT is coined
    - but InstructGPT refers to fine tuned GPT3.5 models (?) 🤔
  - TODO

- Visual Instruction Tuning

  - https://arxiv.org/abs/2304.08485
  - LLaVA
  - their original models are based on LLaMA1
  - but new models based on Llama2 have been released in September 2023.

- Llama 2: Open Foundation and Fine-Tuned Chat Models

  - https://arxiv.org/abs/2307.09288

  

## word-level embeddings

NPLM

- Learns embeddings as it predicts a next word given n - 1 previous words

word2vec

- The weights trained by word2vec actually consist of embedding vectors.
- Learns trying to minimize loss
- Disregards word order
- CBOW
  - Continuous Back Of Words
  - learns embeddings as it predicts missing word from the context
  - mixes context words
- skip-gram (2013a)
  - learns embeddings as it predicts context words
  - uses softmax
  - mixes context words
  - known to be better than CBOW for less frequent words
- skip-gram + negative sampling (2013b)
  - binary classification differentiates 2 types of pairs
    - a true pair
    - k randomly corrupted pairs
      - k = 5~20 is recommended for small corpora
      - k = 2~5 is recommended for big corpora
  - http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
  - https://stats.stackexchange.com/questions/244616/how-does-negative-sampling-work-in-word2vec

(References)

- http://www.goldenplanet.co.kr/blog/2021/05/10/%EB%B9%85%EB%8D%B0%EC%9D%B4%ED%84%B0-%EA%B3%B5%EB%B6%80-%ED%95%9C-%EA%B1%B8%EC%9D%8C-word2vec-%EC%9D%B4%EB%9E%80/

FastText

- Made by Facebook
- Utilizes character level information
- Good for Korean which is an agglutinative language
- https://kavita-ganesan.com/fasttext-vs-word2vec

GloVe

- learns as reducing dimensionality
- easier to parallelize than word2vec

Swivel

## Sentence / document level embeddings

(1900s)

Bag of words

- Works well for small data

tf-idf

- ${\displaystyle \mathrm {tfidf} (t,d,D)=\mathrm {tf} (t,d)\cdot \mathrm {idf} (t,D)}$
- tf
  - term frequency in a document
- idf
  - a measure of how much information the word provides
- https://en.wikipedia.org/wiki/Tf%E2%80%93idf

(2000)

LDA(Latent Dirichlet Allocation)

- https://towardsdatascience.com/a-friendly-introduction-to-text-clustering-fa996bcefd04
- our model
  - how to create a document?
    - Pick a number of words for this document
    - Pick topic distribution
    - For each word
      - pick a topic from the topic distribution
      - draw a word from the topic's words distribution
  - Drichlet distribution??
    - each corner corresponds to a topic chosen
    - we want to keep 2 hyperparameters alpha and beta below 1
      - so that topics can be dicernable

(2014)

Doc2Vec

(2016)

SIF(Smooth Inverse Frequency) embeddings
- https://openreview.net/forum?id=SyK00v5xxhttps://openreview.net/forum?id=SyK00v5xx
- https://github.com/PrincetonML/SIF
- https://towardsdatascience.com/fse-2b1ffa791cf9
- https://bab2min.tistory.com/631

GNMT
- Google's Neural Machine Translation system
- 380M parameters
- https://arxiv.org/abs/1609.08144
- GLEU
  - minimum of recall and precision
  - between 0 and 1
  - better than BLEU when they come to a single sentence rather than a corpus
- Objective function
  - step1
    - $\mathcal{O}_{\mathrm{ML}}(\boldsymbol{\theta})=\sum_{i=1}^{N} \log P_{\theta}\left(Y^{*(i)} \mid X^{(i)}\right)$
    - N: # of input-output sequence pairs
    - maximizing the sum of log probabilities of the ground-truth outputs given the corresponding inputs
    - until convergence
  - step2 (optional)
    - $\mathcal{O}_{\mathrm{Mixed}}(\boldsymbol{\theta})=\alpha * \mathcal{O}_{\mathrm{ML}}(\boldsymbol{\theta})+\mathcal{O}_{\mathrm{RL}}(\boldsymbol{\theta})$
    - $\mathcal{O}_{\mathrm{RL}}(\boldsymbol{\theta})=\sum_{i=1}^{N} \sum_{Y \in \mathcal{Y}} P_{\theta}\left(Y \mid X^{(i)}\right) r\left(Y, Y^{*(i)}\right)$
    - α = 0.017
    - references
      - [A Study of Reinforcement Learning for Neural Machine Translation](https://arxiv.org/abs/1808.08866)
      - [RL in NMT: The Good, the Bad and the Ugly](https://www.cl.uni-heidelberg.de/statnlpgroup/blog/rl4nmt)
- Inference time quantization
  - along with training time help to make the arithmeic operations stable under some constraints
- beam search modification
  - length normalization
  - coverage penalty
- vocabulary sizes
  - word-based using wordpiece model (WPM) tokenizer
  - character-based

(2017)

Transformer

- Ashishi Vaswani at al
- Google Brain, Google Research, University of Toronto
- Transformer base
  - 65M parameters
- Transformer big
  - 213M parameters
- encoder + decoder structure
  - the encoder reads the entire sentences at once
  - the decoder generates a word one by one autoregressively
- Q
  - comes from the target position
- K
  - comes from the position to attend
- V
  - comes from the position to attend
- structure
  - input
    - word embedding vectors of 512 dimension
  - encoders
    - encoder
      - self-attention
        - has dependency between word vectors
        - multi-heads
          - generates $Z$
          - single-head
            - generates $Z_0$
          - ...
      - feed-forward neural net
        - has no dependency between word vectors
    - ...
  - decoders
    - decoder
      - self-attention
      - decoder-encoder-attention
      - feed-forward neural net
      - bidirectional
    - ...
  - output
    - linear weight
      - to transform the decoder output to logits
      - for all words available including the 'end-of-sentence'
    - softmax to pick a word
- self-attention
  - the parallelism enables the model to leverage full power of SIMD hardware accelerators like GPUs/TPUs
  - http://jalammar.github.io/illustrated-transformer/
  - explanation by a single word
    - $x_1$
      - first word vector of 1 x 512
    - $W^Q$
      - weight matrix of 512 x 64 to produce query vectors
    - $W^K$
      - weight matrix of 512 x 64 to produce key vectors
    - $W^V$
      - weight matrix of 512 x 64 to produce value vectors
    - $q_1$
      - query vector of 1 x 64
      - $q_1 = x_1 W^Q$
    - $k_1$
      - key vector of 1 x 64
      - $k_1 = x_1 W^K$
    - $v_1$
      - value vector of 1 x 64
      - $v_1 = x_1 W^V$
    - $z_1$
      - output vector of 1 x 64
      - $z_1 = \text{softmax}({{q_1 \cdot k_1}\over{\sqrt{d_k}}}) \cdot v_1$
      - softmax is done with the results of other words
      - $d_k$ is dimension of key vector $k_1$
        - $\sqrt{d_k} = 8$
  - matrix calculation for single head
    - $X$
      - N x 512
    - $Q$
      - N x 64
      - $Q = X W^Q$
    - $K$
      - N x 64
      - $K = X W^Q$
    - $V$
      - N x 64
      - $V = X W^Q$
    - $Z$
      - N x 64
      - $Z = \text{softmax}({{QK^T}\over{\sqrt{d_k}}}) V$
  - matrix calculation for multi heads
    - $Q_0$
      - N x 64
      - $Q_0 = X W^Q_0$
    - $K_0$
      - N x 64
      - $K = X W^Q_0$
    - $V_0$
      - N x 64
      - $V = X W^Q_0$
    - $Z_0$
      - N x 64
      - $Z_0 = \text{softmax}({{Q_0 K^T_0}\over{\sqrt{d_k}}}) V$
    - $W_0$
      - weights matrix of N x 512
    - $Z$
      - output matrix of N x 512
      - $Z = W_0 \text{concatenate}(Z_0, Z_1, \cdots, Z_7)$
  - position encoding
    - https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
  - label smoothing

(2018)

ELMo

- Accounts for the context dependent meaning
- Train a model which can infer word embeddings given a sentence.
- Uses bidirectional LSTM.

GLoMo

- Unsupervisedly Learned Relational Graphs as Transferable Representations

ULM-FiT

- Applied transfer learning to NLP and had influence on many other pre-training models in NLP such as GPT, BERT
- LSTM based

BERT

- BERT base
  - 110M parameters
- BERT large
  - 345M parameters
- bidirectional
- uses the encoder part of the Transformer model
- outputs are fed into lineary layer and softmax
- MLM(Masked Language Model)
  - mask
    - needs some tokens to be replaced with a special [MASK] token.
  - generate
    - fill in the masks
    - non-autoregressively
    - every token at the output is computed at the same time, without any self-attention mask
    - conditioning on the non-masked tokens, which are present in the same input sequence as the masked tokens
- pretraining tasks
  - fill in the blanks
  - next sentence prediction
- WordPiece tokenizer used in the original paper
- loss
  - compute loss only on corrupted/masked tokens
- input embeddings
  - token embeddings
    - converts each token into a 768-dimensional vector representation.
  - segment embeddings
  - position embeddings
  - embedding size = hidden size
    - actually this could be a shortcoming
    - in Albert, factorized embedding parameterization is used instead to make the embedding size and the hidden size different each other
- special tokens
  - `[CLS]` for classification
  - `[SEP]` for concatenation of 2 sentences
- sentence embeddings
  - methods:
    - BERT embedding
      - average of the BERT output layer
    - output of the first token which is the `[CLS]` token
  - But the embedding quality of both methods is not that great, see Sentence-BERT instead
- drawbacks
  - `[CLS]` is not semantically meaningful. (As mentioned above.)
  - poly encoder has too large computational overhead for use-cases
  - score function is not symmetric
  - bad at setence comparision when using a single BERT, see Sentence-BERT instead

GPT
- autoregressive
- unidirectional
- uses the decoder part of the Transformer model
- 117M parameters
- uses transformer
- uses auxiliary training objectives
- semi-supervised learning for NLP
  - unsupervised pre-training
  - supervised fine-tuning
- paragraph generation
- position embeddings
- https://talktotransformer.com/


(2019)

GPT-2

- 1.5B parameters

KoGPT2

- 125M parameters
- https://github.com/SKT-AI/KoGPT2
- https://huggingface.co/skt/kogpt2-base-v2

ALBERT

- Use less memory by:
  - reusing an identical layer multiple times
  - factorized embedding parameterization

BART
- facebook
- BART-base
  - hidden size = 767
  - encoder
    - 6 layers
  - decoder
    - 6 layers
- BART-large
  - encoder
    - 12 layers
  - decoder
    - 12 layers
- initialization of parameters: N(0, 0.02)
- activation function: GeLU
- pretraining
  - token masking
    - `[MASK]`
  - token deletion
  - text infilling
    - span masking with a single `[MASK]` token
    - span length ~ Poisson(λ=3)
    - 0-length spans correspond to the insertion of `[MASK]` tokens
  - sentence permutation
  - document rotation
    - A token is chosen uniformly at random
    - the document is rotated so that it begins with that token
    - model is trained to find the start
  - fine tuning
    - sequence classification task
    - token classification task
    - sequence generation task
    - machine translation

KoBART
- KoBART-base
  - 124M parameters
  - encoder
    - 6 layers
    - 16 heads
  - decoder
    - 6 layers
    - 16 heads

RoBERTa

- A Robustly Optimized BERT Pretraining Approach

Sentence-BERT

- can generate a sentence embedding that can be compared using cosine similariity
- SBERT, SRoBERTa
- task
  - Semantic Textual Similarity
- siamese architecture
  - classification objective function
    - sentence A -> BERT -> pooling -> u
    - sentence B -> BERT -> pooling -> v
    - concat(u, v, |u-v|) -> softmax
  - inference
    - sentence A -> BERT -> pooling -> u
    - sentence B -> BERT -> pooling -> v
    - cosine-sim(u, v)
- pooling strategies
  - `[CLS]` token
  - mean pooling
  - max pooling
- concatenation
  - (u, v)
  - (|u - v|)
  - (u * v)
  - (|u - v|, u * v)
  - (u, v, u * v)
  - (u, v, |u - v|)
  - (u, v, |u - v|, u * v)
- objective functions
  - classification objective function
    - cross entropy
  - regression objective function
    - MSE
  - triplet objective function
    - max(|s_a - s_p| - |s_a - s_n| + e, 0)
- fine tuning
  - NLI
  - STS
- applications
  - clustering
  - information retrieval
- drawbacks
  - may lose words information so that it my worse than the original BERT depending on the problem
- References
  - https://youtu.be/izCeQOOuZpY

(2020)

Reformer

- LSH
- RevNet
  - recompute the input of each layer on-demand during back-propagation, rather than storing it in memory
- https://arxiv.org/abs/2001.04451

Longformer

- https://arxiv.org/abs/2004.05150
- a localized sliding window based mask
- few global mask to reduce computation
- extended BERT to longer sequence based tasks

ETC(Encoding Long and Structured Inputs in Transformers)
- https://arxiv.org/abs/2004.08483
- Google Research
- a noble global-local attention mechanism

ELECTRA

- 2020 march
- weight sharing between discriminator and generator
  - it's possible because input size can be the same compared to the other general GANs
- loglikelihood loss function
- 모든 단어를 2진분류,, 이 토큰이 진짜인가 아닌가..
- similar to GAN
- 효율성에 집중.
- 적대적 생성기..
- replaced token detection

Big bird

- RoBERTa-like
- Google Research
- relies on block sparse attention
  - attending some global tokens, sliding tokens, and random tokens instead of all tokens
  - linear in the number of tokens
  - in the paper the model can handle sequences up to a length of 4096
- might not be better than BERT, but more efficient
- universial approximator of sequence to sequence functions
- turing complete
  - meaning it can be used to simulate any turing machine
- extra global tokens preserve the expressive power of the model
- introduced novel applications to genomics data

- https://arxiv.org/abs/2007.14062
- https://huggingface.co/blog/big-bird

(2021)

gMLP

## auxilary algorithms

### beam search

- best-first search
  - expand the most promising node only
- beam search
  - expand the most promising beam_size nodes only
  - beam_size
  - α: length normalization coefficient
  - β: coverage normalization coefficient
  - γ: end of sentence normalization
- beam stack search
  - beam search + depth first search
- depth first beam search
  - beam search + depth first search
- BULB
  - beam search using limited discrepancy backtracking
- stochastic beam search
- flexible beam search
- recovery beam search
- breadth-first search
  - expand every thing without pruning
- references
  - https://opennmt.net/OpenNMT/translation/beam_search/

## Evaluation

BLEU-4 (Bilingual Evaluation Understudy score)

- between 0 and 1
- usually expressed as a percentage rather than a decimal
- the higher the better
- evaluation metric for translation
- takes account of more than one candidate reference sentences meaning the same
- has decent correlation with human evaluation
- easy to implement
- depending on the tokenizer
- $\mathrm{BP} \cdot \exp \left(\sum_{n=1}^{N} w_{n} \log p_{n}\right)$
  - $\mathrm{BP}= \begin{cases}1 & \text { if } c>r \\ \exp \left(1-\frac{r}{c}\right) & \text { if } c \leq r\end{cases}$
    - $r$: reference length
    - $c$: translation length
  - $N = 4$
  - $w_n = {1 \over N}$
  - $p_n$
    - $\begin{aligned} p_{n}=& {\sum_{C \in\{\text { Candidates }\}} \sum_{\text{n-gram} \in C} \text { Count clip }(\text{n-gram})} \over {\sum_{C^{\prime} \in\{\text { Candidates }\}} \sum_{\text{n-gram}^{\prime} \in C^{\prime}} \text { Count }\left(\text{n-gram}^{\prime}\right)} \end{aligned}$
- references
  - https://leimao.github.io/blog/BLEU-Score/
  - https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl
  - https://cloud.google.com/translate/automl/docs/evaluate#bleu

PPL(Perplexity)

- The lower the better.
  - A low perplexity indicates the probability distribution is good at predicting the sample.
  - https://en.wikipedia.org/wiki/Perplexity
  - https://huggingface.co/transformers/perplexity.html
- PPL >= 1
- depending on the tokenizer
- good at autoregressive model evaluation

WER

CTER
- CharacTER
- Wang at al., 2016
- https://github.com/roy-ht/pyter

CHRF
- Popović, 2015

## datasets / benchmarks

- GLUE
  - General Language Understanding Evaluation
  - https://gluebenchmark.com/
  - it is different from "GLEU" (which was mentioned in the GNMT paper)
- SST-2 Binary classification
  - https://paperswithcode.com/dataset/sst
- MultiNLI
  - https://cims.nyu.edu/~sbowman/multinli/
- SQuAD v1.1/v2.0
  - https://rajpurkar.github.io/SQuAD-explorer/
- C4
  - https://huggingface.co/datasets/c4
  - English, RealNews
- KorQuAD 1.0
  - https://korquad.github.io/KorQuad%201.0/
## etc

subsampling

- Include the only small amount of a word into the input data set for the efficiency of training

negative sampling

- selects the main word and other a few `noise` words to update the weights for them
- how?
  - make word table regarding the probabilities to be selected as a noise word
  - pick randomly from the table.

Seq2Seq + attention
- https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/


Stop words
