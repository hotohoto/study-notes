# Camera



## Lenses

- fixed focus lens
  - https://en.wikipedia.org/wiki/Fixed-focus_lens
  - focus free or full focus
- fixed focal length lens
  - with fixed AFOV
  - https://en.wikipedia.org/wiki/Prime_lens
- zoom lenses
  - https://en.wikipedia.org/wiki/Zoom_lens
  - it's 4x zoom when the focal length is 100-400mm



## Focal length and FOV

https://www.edmundoptics.com/knowledge-center/application-notes/imaging/understanding-focal-length-and-field-of-view/

$$
\text{AFOV} = 2 \times \tan ^{-1} \left( {H \over 2f} \right)
$$

- f
  - focal length
  - the shorter f, the wider view
  - the longer f, the narrow view
- AFOV
  - angular field of view
  - angle which is corresponding to the width of an object plane

- H
  - width of image plane


$$
\text{AFOV} = 2 \times \tan^{-1}\left({\text{FOV} \over 2 \times \text{WD}}\right)
$$



- FOV
  - field of view
  - width of an object plane
- WD
  - working distance
  - the distance between an object plane and a camera

## Intrinsic parameters

- $f_x$:
  - focal length w.r.t x
- $f_y$:
  - focal length w.r.t y
- $c_x$:
  - mapped x coordinate of (0,0,1)
- $c_y$:
  - mapped y coordinate of (0,0,1)



## Distortion

- In OpenCV, $x$,$y$,$r$ seems to be normalized into [0, 1] first before applying distortion
  - meaning scaling an image doesn't affect to the distortion coefficients

- If it was not normalized into [0, 1]
  - when an image was enlarged $m$ times for both axes
  - meaning
    - $x^\prime = mx$
    - $y^\prime = my$
    - $r^\prime = mr = m\sqrt{x^2 + y^2}$
  - then the distortion coefficients could have been
    - $k_1^\prime = {k_1 \over m^2}$
    - $k_2^\prime = {k_2 \over m^4}$
    - $k_3^\prime = {k_3 \over m^6}$
    - $p_1^\prime = {p_1 \over m}$
    - $p_2^\prime = {p_2 \over m^2}$
  - (but this does not apply!!)
- the bigger $k_1, k_2, k_3$ the more round image we get (barrel distortion)
- the smaller $k_1, k_2, k_3$ the more sharp corners we get (Pincushion distortion) 

## Notes

- ISO
  - 어두운데서 선명하게 나오기 위해 올라감
  - ISO가 너무 높으면 사진이 지저분해보임

- 셔터 스피드 shutter speed
  - 짧으면
    - 사진이 어두워짐
    - jump 샷 같은거 찍기 좋음.
    - 밤에 너무 사진이 밝게 나오면 Tv로 놓고찍기도 함
  - 길면
    - 사진이 흔들리기 쉬움

- 조리개 aperture
  - AF로 놓고 조절할 수 있음.
  - f 값이 낮을 수록 아웃포커싱이 많이 됨. 예쁜 느낌
  - 단체 사진 찍을때는 어려울 수 있음.
  - 대상 뒤에 배경이 멀리 있을수록 아웃포커싱이 잘됨
  - 대상이 가까이 있을수록 아웃포커싱이 잘됨

- composition
  - 드라마나 영화에서 구도 보기
    - 시선 방향으로 공간 두기
    - 시선 방향 반대로 공간 두면 나쁜 사람인 경우
  - 가운데나 황금 비율에 초점 두기

- white balance
  - 색감 조절??

- light
  - 너무 명암비가 쌔면 안 예쁘게 나옴
  - 깊이감을 나타내도록 은은하게 한쪽에서 빛이 오는게 좋음.
  - 음식 사진은 음식 후방 약간 상단 약간 측면에서 빛이 들어오게 하는 경우가 많음.

- depth expression
  - 가까이 있는 물체 약간 나오게 하기
