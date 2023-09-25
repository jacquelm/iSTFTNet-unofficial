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
python scripts/train.py --config configs/config_v1.json
```
For more advanced user : 
```bash
python scripts/train.py --config configs/istftnet/config_v4.json --input_wavs_dir data/VCTK/wavs --input_training_file data/VCTK/train.txt --input_validation_file data/VCTK/val.txt --checkpoint_path checkpoints/cp_istftnet_16khz --ext flac --checkpoint_interval 100000


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