# Speech recognition

- https://www.youtube.com/watch?v=9dXiAecyJrY
  - Usually the length of samples to make spectrogram is 20ms.

## TODO

- curriculum learning
  - dictionary words by frequency without end-of-sentence
  - people names
- better dataset.
- find references for deciding where to put in the dropout layer

## BPTT

- seq2seq
- LSTM classifier
- LSTM generator

## CTC

- questions
  - derive the last equations
  - what is N^T
- seq2seq
  - hard to make data to train with
- attention

## DBN(Deep Belief Network)

known as outperforming the traditional Speech Recognition models.

- curriculum learning (from the short data, ...)
- batch normalization
- sortagrad

## bidirectional rnn

needs a lot of data..

data augmentation - sox
batch size matters

- nervana > cublas
- use 32, 64, 128, 192 when using GPU

- famous dataset:
- librispeech http://www.openslr.org/12
- wall street journal
