<div align="center">

# iSTFTNet : Light weight convolutional vocoder with iSTFT <!-- omit in toc -->
[![OpenInColab]][notebook]
[![paper_badge]][paper]

</div>

Clone of the unofficial ***iSTFTNet*** implementation.  
Model `C8C8I` is implemented.  

![](iSTFTnet.PNG)

## Features
* Audio quality: keeping good quality
* Training time: 30% less time
* Inference speed: 60% faster

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
[paper_badge]: http://img.shields.io/badge/paper-arxiv.2203.02395-B31B1B.svg
[notebook]: https://colab.research.google.com/github/tarepan/iSTFTNet-unofficial/blob/main/istftnet.ipynb
[OpenInColab]: https://colab.research.google.com/assets/colab-badge.svg