# FatFormer

This repository is an official implementation of the CVPR 2024 paper "[Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection](https://arxiv.org/abs/2312.16649)".

☀️ If you find this work useful for your research, please kindly star our repo and cite our paper! ☀️

### TODO
We are working hard on following items.

- [x] Release [arXiv paper](https://arxiv.org/abs/2312.16649)
- [ ] Release PaddlePaddle scripts
- [ ] Release inference scripts
- [ ] Release checkpoints 
- [x] Release datasets

## Introduction

In this paper, we study the problem of generalizable synthetic image detection, e.g., GANs and diffusion models. Cutting-edge solutions start to explore the benefits of pre-trained models, and mainly follow the fixed paradigm of solely training an attached classifier. However, our analysis shows that such a fixed paradigm is prone to yield detectors with insufficient learning regarding forgery representations. **We attribute the key challenge to the lack of forgery adaptation, and present a novel forgery-aware adaptive transformer approach, namely FatFormer.**

![FatFormer Structure](.github/overview.jpg "model structure")

Based on the pre-trained vision-language spaces of CLIP, FatFormer introduces two core designs for the adaption to build generalized forgery representations. First, motivated by the fact that both image and frequency analysis are essential for synthetic image detection, we develop a forgery-aware adapter to adapt image features to discern and integrate local forgery traces within image and frequency domains. Second, we find that considering the contrastive objectives between adapted image features and text prompt embeddings, a previously overlooked aspect, results in a nontrivial generalization improvement. Accordingly, we introduce language-guided alignment to supervise the forgery adaptation with image and text prompts in FatFormer. Experiments show that, by coupling these two designs, our approach tuned on 4-class ProGAN data attains a remarkable detection performance, achieving an average of 98% accuracy to unseen GANs, and surprisingly generalizes to unseen diffusion models with 95% accuracy.

## Datasets
### Training data
To train FatFormer, we adpot images generated by ProGAN with two training settings, consisting of 2-class (chair, horse) and 4-class (car, cat, chair, horse), following [CNNDetection](https://arxiv.org/abs/1912.11035). The original download link can be found in [here](https://github.com/peterwang512/CNNDetection#training-set). You can also download from our [mirror site](https://pan.baidu.com/s/1obzmrCsWvGyUlmH8MkSTLA?pwd=9i5w).

### Testing data
To evaluate FatFormer, we consider the synthetic images from both GANs and diffusion models (DMs). 

* GANs dataset

  For the GANs dataset, we utilize the 8 types of GANs for testing, including ProGAN, StyleGAN, StyleGAN2, BigGAN, CycleGAN, StarGAN, GauGAN and DeepFake, following [LGrad](https://openaccess.thecvf.com/content/CVPR2023/papers/Tan_Learning_on_Gradients_Generalized_Artifacts_Representation_for_GAN-Generated_Images_Detection_CVPR_2023_paper.pdf). The original download link can be found [here](https://github.com/peterwang512/CNNDetection#testset). You can also download from our [mirror site](https://pan.baidu.com/s/1aAiW8oMQcIZIaLYuQIOAjg?pwd=75cz).

* DMs dataset

  For the DMs dataset, we collect 6 types of SOTA DMs, including PNDM, Guided, DALL-E, VQ-Diffusion, LDM, and Glide, from [DIRE](https://arxiv.org/abs/2303.09295) and [UniversalFakeDetect](https://arxiv.org/abs/2302.10174). The original download link can be found [here](https://github.com/ZhendongWang6/DIRE#diffusionforensics-dataset) and [here](https://github.com/Yuheng-Li/UniversalFakeDetect#data). You can also download from our [mirror site](https://pan.baidu.com/s/1zoubPr5n_mGI27En9uyL8Q?pwd=a6sw).

## License
FatFormer is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

## Acknowledgement
This project is built on the open source repositories [CNNDetection](https://github.com/peterwang512/CNNDetection), [LGrad](https://github.com/chuangchuangtan/LGrad), [DIRE](https://github.com/ZhendongWang6/DIRE) and [UniversalFakeDetect](https://github.com/Yuheng-Li/UniversalFakeDetect).
Thanks them for their well-organized codes and datasets!

## Citation

```bibtex
@inproceedings{liu2024forgeryaware,
  title       = {Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection},
  author      = {Liu, Huan and Tan, Zichang and Tan, Chuangchuang and Wei, Yunchao and Wang, Jingdong and Zhao, Yao},
  booktitle   = {Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year        = {2024},
}
```
