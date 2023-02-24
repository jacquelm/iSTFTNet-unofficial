<div align="center">

# iSTFTNet : Light weight convolutional vocoder with iSTFT <!-- omit in toc -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook]
[![Paper](http://img.shields.io/badge/paper-arxiv.2203.02395-B31B1B.svg)][paper]

</div>

Clone of unofficial implementation of iSTFTNet.  
Model `C8C8I` is implemented.  

![](iSTFTnet.PNG)

## Features
* Audio quality: keeping good quality
* Training time: 30 % less time
* Inference speed: 60 % faster

## Training
```bash
python train.py --config config_v1.json
```

## Citations
```
@inproceedings{kaneko2022istftnet,
title={{iSTFTNet}: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse Short-Time Fourier Transform},
author={Takuhiro Kaneko and Kou Tanaka and Hirokazu Kameoka and Shogo Seki},
booktitle={ICASSP},
year={2022},
}
```

## References:
* https://github.com/jik876/hifi-gan

[paper]: https://arxiv.org/abs/2203.02395
[notebook]: https://colab.research.google.com/github/tarepan/iSTFTNet-pytorch/blob/main/istftnet.ipynb