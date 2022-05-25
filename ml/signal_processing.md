# Digital Signal Processing

## terms

(common)

- $t$
  - continuous realtime
  - $t = nT$

(sampling)

- $T$ (or $T_s$)
  - sampling interval in seconds
- $n$, $m$
  - index of samples
- $f_s$
  - sampling rate
  - $f_s = {1 \over T}$

(signal)

- $P$ (or $T$)
  - period
- $f$
  - frequency
  - $f = {1 \over P}$
- $\omega$
  - radian frequency
  - $\omega = 2 \pi f = {2\pi \over P}$
- $A$
  - amplitude
- $\phi$
  - phase

(filters)

- low-pass filter
- high-pass filter
- notch filter
- Bessel filter

## Fourier analysis

(types of signals)

- continuous
  - periodic
    - FS(=CTFS)
  - aperiodic
    - FT(=CTFT)
- discrete
  - periodic
    - DTFS
  - aperiodic
    - finite or infinite length samples
      - DTFT
    - finite-length, and equally-spaced samples only
      - DFT, FFT
        - DFT
          - $O(N^2)$
        - FFT
          - $O(N\log(N))$
      - N inputs and N outputs
      - $- f_s \leq f \leq f_s$
        - $-0.5 \leq f \leq 0.5$
        - $- \pi \leq \omega \leq \pi$

## DFT/FFT

- whole data is considered as a single period
  - the result frequency should be translated with respect to that.
- if the input is pure real numbers, the output is mirrored
  - use the first half
- to optimize FFT, use the power of 2 for the data size

## References

- [Introduction to Digital Filters](https://www.dsprelated.com/freebooks/filters/)
- [Signal Representation and Notation](https://www.dsprelated.com/freebooks/filters/Signal_Representation_Notation.html)
- [3B1B - But what is a Fourier series? From heat flow to circle drawings | DE4](https://youtu.be/r6sGWTCMz2k)
- [comparison for the Fourier analysis variations](https://qr.ae/TQRYzq)
