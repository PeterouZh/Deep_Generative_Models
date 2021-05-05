# GAN-Inversion

A collection of papers I am interested in.

## Project

- [mmgeneration](https://github.com/open-mmlab/mmgeneration)
- [inr-gan](https://github.com/universome/inr-gan)
- [ADA](https://github.com/NVlabs/stylegan2-ada-pytorch)
- [awesome-image-translation](https://github.com/weihaox/awesome-image-translation)
- [awesome-gan-inversion](https://github.com/weihaox/awesome-gan-inversion)
- [naver-webtoon-faces](https://github.com/bryandlee/naver-webtoon-faces)
- [GAN Experiments](http://www.nathanshipley.com/gan/#gan-015-toonify-layer-blending)

## TOC
- [To be read](#to-be-read)
- [Disentanglement](#disentanglement)
- [Inversion](#inversion)
- [Encoder](#encoder)
- [Survey](#survey)
- [GANs](#gans)
- [Style transfer](#style-transfer)
- [Metric](#metric)
- [Spectrum](#spectrum)
- [Weakly Supervised Object Localization](#weakly-supervised-object-localization)
- [NeRF](#nerf)
- [3D](#3d)

## To be read
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:
| [InfinityGAN: Towards Infinite-Resolution Image Synthesis](http://arxiv.org/abs/2104.03963) | arXiv:2104.03963 [cs] |  | 2021 |
| [Aliasing Is Your Ally: End-to-End Super-Resolution from Raw Image Bursts](http://arxiv.org/abs/2104.06191) | arXiv:2104.06191 [cs, eess] |  | 2021 |
| [StylePeople: A Generative Model of Fullbody Human Avatars](http://arxiv.org/abs/2104.08363) | arXiv:2104.08363 [cs] |  | 2021 |
| [Encoding in Style: A StyleGAN Encoder for Image-to-Image Translation](http://arxiv.org/abs/2008.00951) | arXiv:2008.00951 [cs] | [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel) | 2020 |
| [Cross-Domain and Disentangled Face Manipulation with 3D Guidance](http://arxiv.org/abs/2104.11228) | arXiv:2104.11228 [cs] |  | 2021 |
| [On Buggy Resizing Libraries and Surprising Subtleties in FID Calculation](http://arxiv.org/abs/2104.11222) | arXiv:2104.11222 [cs] |  | 2021 |
| [FDA: Fourier Domain Adaptation for Semantic Segmentation](http://arxiv.org/abs/2004.05498) | arXiv:2004.05498 [cs] | [github](https://github.com/YanchaoYang/FDA) | 2020 |
| [Explaining in Style: Training a GAN to Explain a Classifier in StyleSpace](http://arxiv.org/abs/2104.13369) | arXiv:2104.13369 [cs, eess, stat] |  | 2021 |
| [StyleMapGAN: Exploiting Spatial Dimensions of Latent in GAN for Real-Time Image Editing](http://arxiv.org/abs/2104.14754) | CVPR |  | 2021 |
| [Learning a Deep Reinforcement Learning Policy Over the Latent Space of a Pre-Trained GAN for Semantic Age Manipulation](http://arxiv.org/abs/2011.00954) | arXiv:2011.00954 [cs] |  | 2021 |
| [Ensembling with Deep Generative Views](http://arxiv.org/abs/2104.14551) | arXiv:2104.14551 [cs] |  | 2021 |
| [GANalyze: Toward Visual Definitions of Cognitive Image Properties](http://arxiv.org/abs/1906.10112) | arXiv:1906.10112 [cs] |  | 2019 |
| [On the “Steerability” of Generative Adversarial Networks](http://arxiv.org/abs/1907.07171) | arXiv:1907.07171 [cs] |  | 2020 |
| [Editing in Style: Uncovering the Local Semantics of GANs](http://arxiv.org/abs/2004.14367) | CVPR |  | 2020 |
| [Using Latent Space Regression to Analyze and Leverage Compositionality in GANs](https://arxiv.org/abs/2103.10426v1) | ICLR |  | 2021 |
| [EigenGAN: Layer-Wise Eigen-Learning for GANs](http://arxiv.org/abs/2104.12476) | arXiv:2104.12476 [cs, stat] | [EigenGAN](https://github.com/LynnHo/EigenGAN-Tensorflow)  | 2021 |
| [Pose-Controllable Talking Face Generation by Implicitly Modularized Audio-Visual Representation](http://arxiv.org/abs/2104.11116) | arXiv:2104.11116 [cs, eess] |  | 2021 |
| [Unsupervised Image-to-Image Translation via Pre-Trained StyleGAN2 Network](http://arxiv.org/abs/2010.05713) | arXiv:2010.05713 [cs] | [github](https://github.com/HideUnderBush/UI2I_via_StyleGAN2) | 2020 |
| [DatasetGAN: Efficient Labeled Data Factory with Minimal Human Effort](http://arxiv.org/abs/2104.06490) | arXiv:2104.06490 [cs] |  | 2021 |
| [Anycost GANs for Interactive Image Synthesis and Editing](https://arxiv.org/abs/2103.03243v1) | CVPR |  | 2021 |
| [A Simple Baseline for StyleGAN Inversion](http://arxiv.org/abs/2104.07661) | arXiv:2104.07661 [cs] |  | 2021 |
| [Semantic Segmentation with Generative Models: Semi-Supervised Learning and Strong Out-of-Domain Generalization](http://arxiv.org/abs/2104.05833) | arXiv:2104.05833 [cs] |  | 2021 |
| [Positional Encoding as Spatial Inductive Bias in GANs](http://arxiv.org/abs/2012.05217) | arXiv:2012.05217 [cs] |  | 2020 |
| [MobileStyleGAN: A Lightweight Convolutional Neural Network for High-Fidelity Image Synthesis](http://arxiv.org/abs/2104.04767) | arXiv:2104.04767 [cs, eess] |  | 2021 |
| [An Empirical Study of the Effects of Sample-Mixing Methods for Efficient Training of Generative Adversarial Networks](https://arxiv.org/abs/2104.03535v1) | arXiv:2104.03535 [cs.CV] |  | 2021 |
| [Improved StyleGAN Embedding: Where Are the Good Latents?](http://arxiv.org/abs/2012.09036) | arXiv:2012.09036 [cs] |  | 2021 |
| [Regularizing Generative Adversarial Networks under Limited Data](http://arxiv.org/abs/2104.03310) | CVPR | [github](https://github.com/PeterouZh/lecam-gan) | 2021 |
| [Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks](https://ieeexplore.ieee.org/document/9150840/) | CVPRW | [github](https://github.com/haofanwang/Score-CAM) | 2020 |
| [Image Demoireing with Learnable Bandpass Filters](http://arxiv.org/abs/2004.00406) | arXiv:2004.00406 [cs] |  | 2020 |
| [Unveiling the Potential of Structure Preserving for Weakly Supervised Object Localization](http://arxiv.org/abs/2103.04523) | arXiv:2103.04523 [cs] |  | 2021 |
| [LatentCLR: A Contrastive Learning Approach for Unsupervised Discovery of Interpretable Directions](http://arxiv.org/abs/2104.00820) | arXiv:2104.00820 [cs] |  | 2021 |
| [Generating Images with Sparse Representations](http://arxiv.org/abs/2103.03841) | arXiv:2103.03841 [cs, stat] |  | 2021 |
| [PiCIE: Unsupervised Semantic Segmentation Using Invariance and Equivariance in Clustering](http://arxiv.org/abs/2103.17070) | CVPR |  | 2021 |
|FreezeG|| [github](https://github.com/bryandlee/FreezeG) ||
| [Dual Contrastive Loss and Attention for GANs](https://arxiv.org/abs/2103.16748v1) | arXiv:2103.16748 [cs.CV] |  | 2021 |
| [Unsupervised Disentanglement of Linear-Encoded Facial Semantics](https://arxiv.org/abs/2103.16605v1) | CVPR |  | 2021 |
| [Emergence of Object Segmentation in Perturbed Generative Models](http://arxiv.org/abs/1905.12663) | arXiv:1905.12663 [cs] | [github](https://github.com/adambielski/perturbed-seg) | 2019 |
| [Unsupervised Discovery of DisentangledManifolds in GANs](http://arxiv.org/abs/2011.11842) | arXiv:2011.11842 [cs] | [github](https://github.com/anvoynov/GANLatentDiscovery) | 2020 |
| [StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery](http://arxiv.org/abs/2103.17249) | arXiv:2103.17249 [cs] | [github](https://github.com/orpatashnik/StyleCLIP) | 2021 |
| [Few-Shot Semantic Image Synthesis Using StyleGAN Prior](http://arxiv.org/abs/2103.14877) | arXiv:2103.14877 [cs] |  | 2021 |

## Disentanglement
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [GANSpace: Discovering Interpretable GAN Controls](http://arxiv.org/abs/2004.02546) | arXiv:2004.02546 [cs] | [GANSpace](https://github.com/harskish/ganspace) | 2020 |
| [Interpreting the Latent Space of GANs for Semantic Face Editing](http://arxiv.org/abs/1907.10786) | CVPR | [InterFaceGAN](https://github.com/genforce/interfacegan) | 2020 |
| [Closed-Form Factorization of Latent Semantics in GANs](http://arxiv.org/abs/2007.06600) | arXiv:2007.06600 [cs] | [sefa](https://github.com/genforce/sefa) | 2020 |
| [StyleSpace Analysis: Disentangled Controls for StyleGAN Image Generation](http://arxiv.org/abs/2011.12799) | arXiv:2011.12799 [cs] | [StyleSpace](https://github.com/xrenaa/StyleSpace-pytorch) | 2020 |
| [Unsupervised Image Transformation Learning via Generative Adversarial Networks](http://arxiv.org/abs/2103.07751) | arXiv:2103.07751 [cs] | [github](https://github.com/genforce/trgan) | 2021 |
| [Resolution Dependent GAN Interpolation for Controllable Image Synthesis Between Domains](http://arxiv.org/abs/2010.05334) | arXiv:2010.05334 [cs] | [toonify](https://github.com/justinpinkney/toonify) | 2020 |

## Inversion
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models](http://openaccess.thecvf.com/content_CVPR_2020/html/Menon_PULSE_Self-Supervised_Photo_Upsampling_via_Latent_Space_Exploration_of_Generative_CVPR_2020_paper.html) | CVPR |[PULSE]()  | 2020 |
| [Image2StyleGAN++: How to Edit the Embedded Images?](http://arxiv.org/abs/1911.11544) | arXiv:1911.11544 [cs] |  | 2020 |
| [Image2StyleGAN: How to Embed Images Into the StyleGAN Latent Space?](http://arxiv.org/abs/1904.03189) | ICCV |  | 2019 |
| [Inverting The Generator Of A Generative Adversarial Network](http://arxiv.org/abs/1611.05644) | arXiv:1611.05644 [cs] |  | 2016 |
| [Understanding Deep Image Representations by Inverting Them](http://arxiv.org/abs/1412.0035) | CVPR |  | 2015 |
| [Feature-Based Metrics for Exploring the Latent Space of Generative Models](https://openreview.net/forum?id=BJslDBkwG) | ICLRW |  | 2018 |
| [Dreaming to Distill: Data-Free Knowledge Transfer via DeepInversion](http://arxiv.org/abs/1912.08795) | arXiv:1912.08795 [cs, stat] | [DeepInversion](https://github.com/NVlabs/DeepInversion) | 2020 |
| [IMAGINE: Image Synthesis by Image-Guided Model Inversion](http://arxiv.org/abs/2104.05895) | arXiv:2104.05895 [cs] |  | 2021 |
| [Image Processing Using Multi-Code GAN Prior](http://arxiv.org/abs/1912.07116) | CVPR | [mGANprior](https://github.com/genforce/mganprior) | 2020 |
| [Exploiting Deep Generative Prior for Versatile Image Restoration and Manipulation](http://arxiv.org/abs/2003.13659) | ECCV | [DGP](https://github.com/XingangPan/deep-generative-prior) | 2020 |
| [Generative Visual Manipulation on the Natural Image Manifold](http://arxiv.org/abs/1609.03552) | ECCV |  | 2018 |


## Encoder
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution](http://arxiv.org/abs/2012.00739) | arXiv:2012.00739 [cs] |[GLEAN]()  | 2020 |
| [Swapping Autoencoder for Deep Image Manipulation](http://arxiv.org/abs/2007.00653) | arXiv:2007.00653 [cs] | [github](https://github.com/rosinality/swapping-autoencoder-pytorch) | 2020 |
| [In-Domain GAN Inversion for Real Image Editing](http://arxiv.org/abs/2004.00049) | ECCV |  | 2020 |
| [Designing an Encoder for StyleGAN Image Manipulation](http://arxiv.org/abs/2102.02766) | arXiv:2102.02766 [cs] | [encoder4editing](https://github.com/omertov/encoder4editing) | 2021 |
| [ReStyle: A Residual-Based StyleGAN Encoder via Iterative Refinement](http://arxiv.org/abs/2104.02699) | arXiv:2104.02699 [cs] | [ReStyle](https://github.com/yuval-alaluf/restyle-encoder) | 2021 |

## Survey
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [GAN Inversion: A Survey](http://arxiv.org/abs/2101.05278) | arXiv:2101.05278 [cs] |  | 2021 |

## GANs
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [Sampling Generative Networks](http://arxiv.org/abs/1609.04468) | arXiv:1609.04468 [cs, stat] |  | 2016 |
| [A Style-Based Generator Architecture for Generative Adversarial Networks](http://arxiv.org/abs/1812.04948) | CVPR |  | 2019 |
| [Analyzing and Improving the Image Quality of StyleGAN](http://arxiv.org/abs/1912.04958) | arXiv:1912.04958 [cs, eess, stat] |  | 2019 |
| [Training Generative Adversarial Networks with Limited Data](http://arxiv.org/abs/2006.06676) | arXiv:2006.06676 [cs, stat] |  | 2020 |

## Style transfer
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization](http://arxiv.org/abs/1703.06868) | ICCV |  | 2017 |
| [Texture Synthesis Using Convolutional Neural Networks](http://arxiv.org/abs/1505.07376) | NeurIPS |  | 2015 |
| [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) | arXiv:1508.06576 [cs, q-bio] |  | 2015 |
| [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html) | CVPR |  | 2016 |
| [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://arxiv.org/abs/1603.08155) | ECCV |  | 2016 |
| [Texture Networks: Feed-Forward Synthesis of Textures and Stylized Images](http://arxiv.org/abs/1603.03417) | ICML |  | 2016 |


## Metric
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](http://arxiv.org/abs/1801.03924) | arXiv:1801.03924 [cs] | [lpips-pytorch](https://github.com/S-aiueo32/lpips-pytorch) | 2018 |
| [Generating Images with Perceptual Similarity Metrics Based on Deep Networks](http://arxiv.org/abs/1602.02644) | NeurIPS | Perceptual Similarity | 2016 |


## Spectrum
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [Reproducibility of "FDA: Fourier Domain Adaptation ForSemantic Segmentation](http://arxiv.org/abs/2104.14749) | arXiv:2104.14749 [cs] |  | 2021 |
| [A Closer Look at Fourier Spectrum Discrepancies for CNN-Generated Images Detection](http://arxiv.org/abs/2103.17195) | CVPR |  | 2021 |

## Weakly Supervised Object Localization
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [Labels4Free: Unsupervised Segmentation Using StyleGAN](http://arxiv.org/abs/2103.14968) | arXiv:2103.14968 [cs] |  | 2021 |
| [TS-CAM: Token Semantic Coupled Attention Map for Weakly Supervised Object Localization](http://arxiv.org/abs/2103.14862) | arXiv:2103.14862 [cs] |  | 2021 |


## NeRF
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://arxiv.org/abs/2003.08934) | ECCV | [github](https://github.com/yenchenlin/nerf-pytorch)  | 2020 |
| [Neural Volume Rendering: NeRF And Beyond](http://arxiv.org/abs/2101.05204) | arXiv:2101.05204 [cs] | [github](https://github.com/yenchenlin/awesome-NeRF) | 2021 |


## 3D
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [Image GANs Meet Differentiable Rendering for Inverse Graphics and Interpretable 3D Neural Rendering](http://arxiv.org/abs/2010.09125) | arXiv:2010.09125 [cs] |  | 2021 |
| [Unsupervised Learning of Probably Symmetric Deformable 3D Objects from Images in the Wild](http://arxiv.org/abs/1911.11130) | arXiv:1911.11130 [cs] |[unsup3d](https://github.com/elliottwu/unsup3d)  | 2020 |
| [Do 2D GANs Know 3D Shape? Unsupervised 3D Shape Reconstruction from 2D Image GANs](http://arxiv.org/abs/2011.00844) | arXiv:2011.00844 [cs] | [GAN2Shape](https://github.com/XingangPan/GAN2Shape) | 2021 |
| [Neural 3D Mesh Renderer](http://arxiv.org/abs/1711.07566) | CVPR |  | 2018 |














