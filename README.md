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
- [timm](https://github.com/rwightman/pytorch-image-models)
- [fun-with-computer-graphics](https://github.com/zheng95z/fun-with-computer-graphics)

### Face
- [StyleGAN-nada](https://github.com/rinongal/StyleGAN-nada)
- [RetrieveInStyle](https://github.com/mchong6/RetrieveInStyle)

### 3D 
- [pi-GAN](https://github.com/marcoamonteiro/pi-GAN)
- [face3d](https://github.com/YadiraF/face3d)

### Tools
- [bokeh](https://github.com/bokeh/bokeh)
- [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)
- [label-studio](https://github.com/heartexlabs/label-studio)
- [streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas)
- [face-alignment](https://github.com/1adrianb/face-alignment)


### Style transfer
- [style-transfer-pytorch](https://github.com/crowsonkb/style-transfer-pytorch)
- [Stylebank-exp](https://github.com/PeterouZh/Stylebank-exp)

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

## arXiv
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:
| [Perceptual Gradient Networks](http://arxiv.org/abs/2105.01957) | arXiv:2105.01957 [cs] |  | 2021 |
| [Unconstrained Scene Generation with Locally Conditioned Radiance Fields](https://arxiv.org/abs/2104.00670v1) | arXiv:2104.00670 [cs.CV] |  | 2021 |
| [InfinityGAN: Towards Infinite-Resolution Image Synthesis](http://arxiv.org/abs/2104.03963) | arXiv:2104.03963 [cs] |  | 2021 |
| [Aliasing Is Your Ally: End-to-End Super-Resolution from Raw Image Bursts](http://arxiv.org/abs/2104.06191) | arXiv:2104.06191 [cs, eess] |  | 2021 |
| [StylePeople: A Generative Model of Fullbody Human Avatars](http://arxiv.org/abs/2104.08363) | arXiv:2104.08363 [cs] |  | 2021 |
| [Cross-Domain and Disentangled Face Manipulation with 3D Guidance](http://arxiv.org/abs/2104.11228) | arXiv:2104.11228 [cs] |  | 2021 |
| [On Buggy Resizing Libraries and Surprising Subtleties in FID Calculation](http://arxiv.org/abs/2104.11222) | arXiv:2104.11222 [cs] |  | 2021 |
| [FDA: Fourier Domain Adaptation for Semantic Segmentation](http://arxiv.org/abs/2004.05498) | arXiv:2004.05498 [cs] | [github](https://github.com/YanchaoYang/FDA) | 2020 |
| [Explaining in Style: Training a GAN to Explain a Classifier in StyleSpace](http://arxiv.org/abs/2104.13369) | arXiv:2104.13369 [cs, eess, stat] |  | 2021 |
| [StyleMapGAN: Exploiting Spatial Dimensions of Latent in GAN for Real-Time Image Editing](http://arxiv.org/abs/2104.14754) | CVPR |  | 2021 |
| [Learning a Deep Reinforcement Learning Policy Over the Latent Space of a Pre-Trained GAN for Semantic Age Manipulation](http://arxiv.org/abs/2011.00954) | arXiv:2011.00954 [cs] |  | 2021 |
| [Ensembling with Deep Generative Views](http://arxiv.org/abs/2104.14551) | arXiv:2104.14551 [cs] |  | 2021 |
| [GANalyze: Toward Visual Definitions of Cognitive Image Properties](http://arxiv.org/abs/1906.10112) | arXiv:1906.10112 [cs] |  | 2019 |
| [On the “Steerability” of Generative Adversarial Networks](http://arxiv.org/abs/1907.07171) | arXiv:1907.07171 [cs] |  | 2020 |
| [Using Latent Space Regression to Analyze and Leverage Compositionality in GANs](https://arxiv.org/abs/2103.10426v1) | ICLR |  | 2021 |
| [Pose-Controllable Talking Face Generation by Implicitly Modularized Audio-Visual Representation](http://arxiv.org/abs/2104.11116) | arXiv:2104.11116 [cs, eess] |  | 2021 |
| [Unsupervised Image-to-Image Translation via Pre-Trained StyleGAN2 Network](http://arxiv.org/abs/2010.05713) | arXiv:2010.05713 [cs] | [github](https://github.com/HideUnderBush/UI2I_via_StyleGAN2) | 2020 |
| [DatasetGAN: Efficient Labeled Data Factory with Minimal Human Effort](http://arxiv.org/abs/2104.06490) | arXiv:2104.06490 [cs] |  | 2021 |
| [Anycost GANs for Interactive Image Synthesis and Editing](https://arxiv.org/abs/2103.03243v1) | CVPR |  | 2021 |
| [Semantic Segmentation with Generative Models: Semi-Supervised Learning and Strong Out-of-Domain Generalization](http://arxiv.org/abs/2104.05833) | arXiv:2104.05833 [cs] |  | 2021 |
| [Positional Encoding as Spatial Inductive Bias in GANs](http://arxiv.org/abs/2012.05217) | arXiv:2012.05217 [cs] |  | 2020 |
| [An Empirical Study of the Effects of Sample-Mixing Methods for Efficient Training of Generative Adversarial Networks](https://arxiv.org/abs/2104.03535v1) | arXiv:2104.03535 [cs.CV] |  | 2021 |
| [Improved StyleGAN Embedding: Where Are the Good Latents?](http://arxiv.org/abs/2012.09036) | arXiv:2012.09036 [cs] |  | 2021 |
| [Regularizing Generative Adversarial Networks under Limited Data](http://arxiv.org/abs/2104.03310) | CVPR | [github](https://github.com/PeterouZh/lecam-gan) | 2021 |
| [Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks](https://ieeexplore.ieee.org/document/9150840/) | CVPRW | [github](https://github.com/haofanwang/Score-CAM) | 2020 |
| [Image Demoireing with Learnable Bandpass Filters](http://arxiv.org/abs/2004.00406) | arXiv:2004.00406 [cs] |  | 2020 |
| [Unveiling the Potential of Structure Preserving for Weakly Supervised Object Localization](http://arxiv.org/abs/2103.04523) | arXiv:2103.04523 [cs] |  | 2021 |
| [LatentCLR: A Contrastive Learning Approach for Unsupervised Discovery of Interpretable Directions](http://arxiv.org/abs/2104.00820) | arXiv:2104.00820 [cs] |  | 2021 |
| [Generating Images with Sparse Representations](http://arxiv.org/abs/2103.03841) | arXiv:2103.03841 [cs, stat] |  | 2021 |
| [PiCIE: Unsupervised Semantic Segmentation Using Invariance and Equivariance in Clustering](http://arxiv.org/abs/2103.17070) | CVPR |  | 2021 |
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

### Semantic hierarchy
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [Semantic Hierarchy Emerges in Deep Generative Representations for Scene Synthesis](http://arxiv.org/abs/1911.09267) | arXiv:1911.09267 [cs] |  | 2020 |


## Inversion

###  Optimization
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models](http://openaccess.thecvf.com/content_CVPR_2020/html/Menon_PULSE_Self-Supervised_Photo_Upsampling_via_Latent_Space_Exploration_of_Generative_CVPR_2020_paper.html) | CVPR |[PULSE]()  | 2020 |
| [Image2StyleGAN++: How to Edit the Embedded Images?](http://arxiv.org/abs/1911.11544) | arXiv:1911.11544 [cs] |  | 2020 |
| [Image2StyleGAN: How to Embed Images Into the StyleGAN Latent Space?](http://arxiv.org/abs/1904.03189) | ICCV |  | 2019 |
| [Inverting The Generator Of A Generative Adversarial Network](http://arxiv.org/abs/1611.05644) | arXiv:1611.05644 [cs] |  | 2016 |
| [Feature-Based Metrics for Exploring the Latent Space of Generative Models](https://openreview.net/forum?id=BJslDBkwG) | ICLRW |  | 2018 |
| [Understanding Deep Image Representations by Inverting Them](http://arxiv.org/abs/1412.0035) | CVPR |  | 2015 |
| [Dreaming to Distill: Data-Free Knowledge Transfer via DeepInversion](http://arxiv.org/abs/1912.08795) | arXiv:1912.08795 [cs, stat] | [DeepInversion](https://github.com/NVlabs/DeepInversion) | 2020 |
| [IMAGINE: Image Synthesis by Image-Guided Model Inversion](http://arxiv.org/abs/2104.05895) | arXiv:2104.05895 [cs] |  | 2021 |
| [Image Processing Using Multi-Code GAN Prior](http://arxiv.org/abs/1912.07116) | CVPR | [mGANprior](https://github.com/genforce/mganprior) | 2020 |
| [Exploiting Deep Generative Prior for Versatile Image Restoration and Manipulation](http://arxiv.org/abs/2003.13659) | ECCV | [DGP](https://github.com/XingangPan/deep-generative-prior) | 2020 |
| [Generative Visual Manipulation on the Natural Image Manifold](http://arxiv.org/abs/1609.03552) | ECCV |  | 2018 |
| [GAN Dissection: Visualizing and Understanding Generative Adversarial Networks](http://arxiv.org/abs/1811.10597) | arXiv:1811.10597 [cs] |  | 2018 |
| [GAN-Based Projector for Faster Recovery with Convergence Guarantees in Linear Inverse Problems](http://arxiv.org/abs/1902.09698) | arXiv:1902.09698 [cs, eess, stat] |  | 2019 |
| [Your Local GAN: Designing Two Dimensional Local Attention Mechanisms for Generative Models](http://openaccess.thecvf.com/content_CVPR_2020/html/Daras_Your_Local_GAN_Designing_Two_Dimensional_Local_Attention_Mechanisms_for_CVPR_2020_paper.html) | CVPR |  | 2020 |
| [Rewriting a Deep Generative Model](http://arxiv.org/abs/2007.15646) | arXiv:2007.15646 [cs] |  | 2020 |
| [Transforming and Projecting Images into Class-Conditional Generative Networks](http://arxiv.org/abs/2005.01703) | arXiv:2005.01703 [cs] |  | 2020 |
| [StyleGAN2 Distillation for Feed-Forward Image Manipulation](https://arxiv.org/abs/2003.03581v2) | arXiv:2003.03581 [cs.CV] |  | 2020 |
| [On the “Steerability” of Generative Adversarial Networks](http://arxiv.org/abs/1907.07171) | arXiv:1907.07171 [cs] |  | 2020 |
| [Unsupervised Discovery of DisentangledManifolds in GANs](http://arxiv.org/abs/2011.11842) | arXiv:2011.11842 [cs] |  | 2020 |
| [PIE: Portrait Image Embedding for Semantic Control](http://arxiv.org/abs/2009.09485) | arXiv:2009.09485 [cs] |  | 2020 |
| [GANSpace: Discovering Interpretable GAN Controls](http://arxiv.org/abs/2004.02546) | NeurIPS |  | 2020 |
| [When and How Can Deep Generative Models Be Inverted?](http://arxiv.org/abs/2006.15555) | arXiv:2006.15555 [cs, stat] |  | 2020 |
| [Style Intervention: How to Achieve Spatial Disentanglement with Style-Based Generators?](http://arxiv.org/abs/2011.09699) | arXiv:2011.09699 [cs] |  | 2020 |
| [StyleSpace Analysis: Disentangled Controls for StyleGAN Image Generation](http://arxiv.org/abs/2011.12799) | arXiv:2011.12799 [cs] |  | 2020 |
| [Navigating the GAN Parameter Space for Semantic Image Editing](http://arxiv.org/abs/2011.13786) | arXiv:2011.13786 [cs] |  | 2021 |
| [Mask-Guided Discovery of Semantic Manifolds in Generative Models](http://arxiv.org/abs/2105.07273) | arXiv:2105.07273 [cs] | [masked-gan-manifold](https://github.com/bmolab/masked-gan-manifold) | 2021 |
| [StyleFlow: Attribute-Conditioned Exploration of StyleGAN-Generated Images Using Conditional Continuous Normalizing Flows](http://arxiv.org/abs/2008.02401) | arXiv:2008.02401 [cs] | [StyleFlow](https://github.com/RameenAbdal/StyleFlow) | 2020 |
| [Disentangled Face Attribute Editing via Instance-Aware Latent Space Search](http://arxiv.org/abs/2105.12660) | arXiv:2105.12660 [cs] |  | 2021 |
| [Barbershop: GAN-Based Image Compositing Using Segmentation Masks](http://arxiv.org/abs/2106.01505) | arXiv:2106.01505 [cs] |  | 2021 |
| [Unsupervised Discovery of Interpretable Directions in the GAN Latent Space](http://arxiv.org/abs/2002.03754) | arXiv:2002.03754 [cs, stat] | [GANLatentDiscovery](https://github.com/anvoynov/GANLatentDiscovery) | 2020 |
| [Pivotal Tuning for Latent-Based Editing of Real Images](http://arxiv.org/abs/2106.05744) | arXiv:2106.05744 [cs] | [PTI](https://github.com/danielroich/PTI) | 2021 |
| [Fine-Tuning StyleGAN2 For Cartoon Face Generation](http://arxiv.org/abs/2106.12445) | arXiv:2106.12445 [cs, eess] | [Cartoon-StyleGAN](https://github.com/happy-jihye/Cartoon-StyleGAN) | 2021 |
| [Editing in Style: Uncovering the Local Semantics of GANs](http://arxiv.org/abs/2004.14367) | CVPR |  | 2020 |
| [Retrieve in Style: Unsupervised Facial Feature Transfer and Retrieval](http://arxiv.org/abs/2107.06256) | arXiv:2107.06256 [cs] | [RetrieveInStyle](https://github.com/mchong6/RetrieveInStyle) | 2021 |
| [StyleCariGAN: Caricature Generation via StyleGAN Feature Map Modulation](http://arxiv.org/abs/2107.04331) | arXiv:2107.04331 [cs] |  | 2021 |
| [A Simple Baseline for StyleGAN Inversion](http://arxiv.org/abs/2104.07661) | arXiv:2104.07661 [cs] |  | 2021 |


### Encoder
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution](http://arxiv.org/abs/2012.00739) | arXiv:2012.00739 [cs] |[GLEAN]()  | 2020 |
| [Swapping Autoencoder for Deep Image Manipulation](http://arxiv.org/abs/2007.00653) | arXiv:2007.00653 [cs] | [github](https://github.com/rosinality/swapping-autoencoder-pytorch) | 2020 |
| [In-Domain GAN Inversion for Real Image Editing](http://arxiv.org/abs/2004.00049) | ECCV |  | 2020 |
| [ReStyle: A Residual-Based StyleGAN Encoder via Iterative Refinement](http://arxiv.org/abs/2104.02699) | arXiv:2104.02699 [cs] | [ReStyle](https://github.com/yuval-alaluf/restyle-encoder) | 2021 |
| [Interpreting the Latent Space of GANs for Semantic Face Editing](http://arxiv.org/abs/1907.10786) | CVPR |  | 2020 |
| [Face Identity Disentanglement via Latent Space Mapping](http://arxiv.org/abs/2005.07728) | arXiv:2005.07728 [cs] |  | 2020 |
| [Collaborative Learning for Faster StyleGAN Embedding](http://arxiv.org/abs/2007.01758) | arXiv:2007.01758 [cs] |  | 2020 |
| [Unsupervised Discovery of DisentangledManifolds in GANs](http://arxiv.org/abs/2011.11842) | arXiv:2011.11842 [cs] |  | 2020 |
| [Generative Hierarchical Features from Synthesizing Images](http://arxiv.org/abs/2007.10379) | arXiv:2007.10379 [cs] |  | 2020 |
| [One Shot Face Swapping on Megapixels](http://arxiv.org/abs/2105.04932) | arXiv:2105.04932 [cs] |  | 2021 |
| [GAN Prior Embedded Network for Blind Face Restoration in the Wild](https://arxiv.org/abs/2105.06070v1) | 2021 |
| [Adversarial Latent Autoencoders](http://openaccess.thecvf.com/content_CVPR_2020/html/Pidhorskyi_Adversarial_Latent_Autoencoders_CVPR_2020_paper.html) | CVPR | [ALAE](https://github.com/podgorskiy/ALAE) | 2020 |
| [Encoding in Style: A StyleGAN Encoder for Image-to-Image Translation](http://arxiv.org/abs/2008.00951) | arXiv:2008.00951 [cs] | [psp](https://github.com/eladrich/pixel2style2pixel) | 2021 |
| [Designing an Encoder for StyleGAN Image Manipulation](http://arxiv.org/abs/2102.02766) | arXiv:2102.02766 [cs] | [encoder4editing](https://github.com/omertov/encoder4editing) | 2021 |
| [A Latent Transformer for Disentangled and Identity-Preserving Face Editing](http://arxiv.org/abs/2106.11895) | arXiv:2106.11895 [cs] |  | 2021 |
| [ShapeEditer: A StyleGAN Encoder for Face Swapping](http://arxiv.org/abs/2106.13984) | arXiv:2106.13984 [cs] |  | 2021 |
| [Force-in-Domain GAN Inversion](http://arxiv.org/abs/2107.06050) | arXiv:2107.06050 [cs, eess] |  | 2021 |
| [StyleFusion: A Generative Model for Disentangling Spatial Segments](http://arxiv.org/abs/2107.07437) | arXiv:2107.07437 [cs] |  | 2021 |
| [Perceptually Validated Precise Local Editing for Facial Action Units with StyleGAN](http://arxiv.org/abs/2107.12143) | arXiv:2107.12143 [cs] |  | 2021 |
| [StyleGAN2 Distillation for Feed-Forward Image Manipulation](https://arxiv.org/abs/2003.03581v2) | arXiv:2003.03581 [cs.CV] |  | 2020 |

### Hybrid optimization
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [Generative Visual Manipulation on the Natural Image Manifold](http://arxiv.org/abs/1609.03552) | ECCV |  | 2018 |
| [Semantic Photo Manipulation with a Generative Image Prior](https://arxiv.org/abs/2005.07727) | ACM Transactions on Graphics |  | 2019 |
| [Seeing What a GAN Cannot Generate](http://arxiv.org/abs/1910.11626) | arXiv:1910.11626 [cs, eess] |  | 2019 |
| [In-Domain GAN Inversion for Real Image Editing](http://arxiv.org/abs/2004.00049) | ECCV |  | 2020 |



### Without optimization
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [Closed-Form Factorization of Latent Semantics in GANs](http://arxiv.org/abs/2007.06600) | arXiv:2007.06600 [cs] |  | 2020 |
| [GAN “Steerability” without Optimization](http://arxiv.org/abs/2012.05328) | arXiv:2012.05328 [cs] |  | 2021 |
| [Low-Rank Subspaces in GANs](http://arxiv.org/abs/2106.04488) | arXiv:2106.04488 [cs] |  | 2021 |
| [LARGE: Latent-Based Regression through GAN Semantics](http://arxiv.org/abs/2107.11186) | arXiv:2107.11186 [cs] |  | 2021 |


### Cls
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [Contrastive Model Inversion for Data-Free Knowledge Distillation](http://arxiv.org/abs/2105.08584) | arXiv:2105.08584 [cs] |  | 2021 |
| [Generative Models as a Data Source for Multiview Representation Learning](http://arxiv.org/abs/2106.05258) | arXiv:2106.05258 [cs] |  | 2021 |
| [Inverting and Understanding Object Detectors](http://arxiv.org/abs/2106.13933) | arXiv:2106.13933 [cs] |  | 2021 |
| [Deep Neural Networks Are Surprisingly Reversible: A Baseline for Zero-Shot Inversion](http://arxiv.org/abs/2107.06304) | arXiv:2107.06304 [cs] |  | 2021 |


## Survey
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [GAN Inversion: A Survey](http://arxiv.org/abs/2101.05278) | arXiv:2101.05278 [cs] |  | 2021 |

## GANs


### Regs
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [The Hessian Penalty: A Weak Prior for Unsupervised Disentanglement](http://arxiv.org/abs/2008.10599) | ECCV |  | 2020 |


### StyleGANs
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [A Style-Based Generator Architecture for Generative Adversarial Networks](http://arxiv.org/abs/1812.04948) | CVPR |  | 2019 |
| [Analyzing and Improving the Image Quality of StyleGAN](http://arxiv.org/abs/1912.04958) | arXiv:1912.04958 [cs, eess, stat] |  | 2019 |
| [Training Generative Adversarial Networks with Limited Data](http://arxiv.org/abs/2006.06676) | arXiv:2006.06676 [cs, stat] |  | 2020 |
| [Alias-Free Generative Adversarial Networks](http://arxiv.org/abs/2106.12423) | arXiv:2106.12423 [cs, stat] | [alias-free-gan](https://github.com/NVlabs/alias-free-gan) | 2021 |
| [Transforming the Latent Space of StyleGAN for Real Face Editing](http://arxiv.org/abs/2105.14230) | arXiv:2105.14230 [cs] | [TransStyleGAN](https://github.com/AnonSubm2021/TransStyleGAN) | 2021 |
| [MobileStyleGAN: A Lightweight Convolutional Neural Network for High-Fidelity Image Synthesis](http://arxiv.org/abs/2104.04767) | arXiv:2104.04767 [cs, eess] | [MobileStyleGAN](https://github.com/bes-dev/MobileStyleGAN.pytorch) | 2021 |
| [Few-Shot Image Generation via Cross-Domain Correspondence](http://arxiv.org/abs/2104.06820) | CVPR | [few-shot-gan-adaptation](https://github.com/utkarshojha/few-shot-gan-adaptation) | 2021 |
| [EigenGAN: Layer-Wise Eigen-Learning for GANs](http://arxiv.org/abs/2104.12476) | arXiv:2104.12476 [cs, stat] | [EigenGAN](https://github.com/LynnHo/EigenGAN-Tensorflow)  | 2021 |
| [Toward Spatially Unbiased Generative Models](http://arxiv.org/abs/2108.01285) | ICCV | [toward_spatial_unbiased](https://github.com/jychoi118/toward_spatial_unbiased) | 2021 |


### SinGAN
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [ExSinGAN: Learning an Explainable Generative Model from a Single Image](http://arxiv.org/abs/2105.07350) | arXiv:2105.07350 [cs] |  | 2021 |


### GANs
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [Sampling Generative Networks](http://arxiv.org/abs/1609.04468) | arXiv:1609.04468 [cs, stat] |  | 2016 |
| [Combining Transformer Generators with Convolutional Discriminators](http://arxiv.org/abs/2105.10189) | arXiv:2105.10189 [cs] |  | 2021 |
| [Improving Generation and Evaluation of Visual Stories via Semantic Consistency](http://arxiv.org/abs/2105.10026) | arXiv:2105.10026 [cs] |  | 2021 |
| [Towards Faster and Stabilized GAN Training for High-Fidelity Few-Shot Image Synthesis](https://openreview.net/forum?id=1Fqg133qRaI) | ICLR2021 | [github](https://github.com/lucidrains/lightweight-gan) | 2021 |
| [TediGAN: Text-Guided Diverse Face Image Generation and Manipulation](http://arxiv.org/abs/2012.03308) | CVPR |  | 2021 |
| [Data-Efficient Instance Generation from Instance Discrimination](http://arxiv.org/abs/2106.04566) | arXiv:2106.04566 [cs] |  | 2021 |
| [Styleformer: Transformer Based Generative Adversarial Networks with Style Vector](http://arxiv.org/abs/2106.07023) | arXiv:2106.07023 [cs, eess] |  | 2021 |
| [FBC-GAN: Diverse and Flexible Image Synthesis via Foreground-Background Composition](http://arxiv.org/abs/2107.03166) | arXiv:2107.03166 [cs] |  | 2021 |
| [ViTGAN: Training GANs with Vision Transformers](http://arxiv.org/abs/2107.04589) | arXiv:2107.04589 [cs, eess] |  | 2021 |
| [Learning Efficient GANs for Image Translation via Differentiable Masks and Co-Attention Distillation](http://arxiv.org/abs/2011.08382) | arXiv:2011.08382 [cs] |  | 2021 |
| [CGANs with Auxiliary Discriminative Classifier](http://arxiv.org/abs/2107.10060) | arXiv:2107.10060 [cs] |  | 2021 |
|FreezeG|| [github](https://github.com/bryandlee/FreezeG) ||
| :white_check_mark: [Freeze the Discriminator: A Simple Baseline for Fine-Tuning GANs](http://arxiv.org/abs/2002.10964) | arXiv:2002.10964 [cs, stat] | [FreezeD](https://github.com/sangwoomo/FreezeD) | 2020 |
| [A Good Image Generator Is What You Need for High-Resolution Video Synthesis](http://arxiv.org/abs/2104.15069) | ICLR |  | 2021 |


## GAN application
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [SC-FEGAN: Face Editing Generative Adversarial Network with User’s Sketch and Color](http://arxiv.org/abs/1902.06838) | arXiv:1902.06838 [cs] |  | 2019 |
| [Semantic Text-to-Face GAN -ST^2FG](http://arxiv.org/abs/2107.10756) | arXiv:2107.10756 [cs] |  | 2021 |
| [CRD-CGAN: Category-Consistent and Relativistic Constraints for Diverse Text-to-Image Generation](http://arxiv.org/abs/2107.13516) | arXiv:2107.13516 [cs] |  | 2021 |


## Image-to-Image Translation
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [Image-to-Image Translation with Conditional Adversarial Networks](http://arxiv.org/abs/1611.07004) | CVPR | [pix2pix]()  | 2017 |
| [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](http://arxiv.org/abs/1711.11585) | CVPR | [pix2pix-HD]() | 2018 |
| [Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks](http://arxiv.org/abs/1703.10593) | ICCV | [CycleGAN]() | 2017 |
| [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](http://arxiv.org/abs/1711.09020) | CVPR |  | 2018 |
| [StarGAN v2: Diverse Image Synthesis for Multiple Domains](http://arxiv.org/abs/1912.01865) | CVPR |  | 2020 |
| [Multimodal Unsupervised Image-to-Image Translation](http://arxiv.org/abs/1804.04732) | arXiv:1804.04732 [cs, stat] | [MUNIT]()  | 2018 |
| [High-Resolution Photorealistic Image Translation in Real-Time: A Laplacian Pyramid Translation Network](http://arxiv.org/abs/2105.09188) | arXiv:2105.09188 [cs] |  | 2021 |
| [MixerGAN: An MLP-Based Architecture for Unpaired Image-to-Image Translation](http://arxiv.org/abs/2105.14110) | arXiv:2105.14110 [cs] |  | 2021 |
| [GANs N’ Roses: Stable, Controllable, Diverse Image to Image Translation (Works for Videos Too!)](http://arxiv.org/abs/2106.06561) | arXiv:2106.06561 [cs] |  | 2021 |
| [Sketch Your Own GAN](http://arxiv.org/abs/2108.02774) | ICCV |  | 2021 |




## Style transfer
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization](http://arxiv.org/abs/1703.06868) | ICCV |  | 2017 |
| [Texture Synthesis Using Convolutional Neural Networks](http://arxiv.org/abs/1505.07376) | NeurIPS |  | 2015 |
| [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) | arXiv:1508.06576 [cs, q-bio] |  | 2015 |
| [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html) | CVPR |  | 2016 |
| [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://arxiv.org/abs/1603.08155) | ECCV |  | 2016 |
| [Texture Networks: Feed-Forward Synthesis of Textures and Stylized Images](http://arxiv.org/abs/1603.03417) | ICML |  | 2016 |
| [Attention-Based Stylisation for Exemplar Image Colourisation](http://arxiv.org/abs/2105.01705) | arXiv:2105.01705 [cs, eess] |  | 2021 |
| [StyleBank: An Explicit Representation for Neural Image Style Transfer](https://arxiv.org/abs/1703.09210v2) | | [Stylebank](https://github.com/jxcodetw/Stylebank) | 2017 |
| [Rethinking and Improving the Robustness of Image Style Transfer](http://arxiv.org/abs/2104.05623) | arXiv:2104.05623 [cs, eess] |  | 2021 |
| [Paint Transformer: Feed Forward Neural Painting with Stroke Prediction](http://arxiv.org/abs/2108.03798) | ICCV |  | 2021 |


## Metric
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](http://arxiv.org/abs/1801.03924) | arXiv:1801.03924 [cs] | [lpips-pytorch](https://github.com/S-aiueo32/lpips-pytorch) | 2018 |
| [Generating Images with Perceptual Similarity Metrics Based on Deep Networks](http://arxiv.org/abs/1602.02644) | NeurIPS | Perceptual Similarity | 2016 |
| [Generic Perceptual Loss for Modeling Structured Output Dependencies](http://arxiv.org/abs/2103.10571) | CVPR |  | 2021 |
| [Inverting Adversarially Robust Networks for Image Synthesis](http://arxiv.org/abs/2106.06927) | arXiv:2106.06927 [cs] |  | 2021 |


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
| [Finding an Unsupervised Image Segmenter in Each of Your Deep Generative Models](http://arxiv.org/abs/2105.08127) | arXiv:2105.08127 [cs] |  | 2021 |
| [Segmentation in Style: Unsupervised Semantic Image Segmentation with Stylegan and CLIP](http://arxiv.org/abs/2107.12518) | arXiv:2107.12518 [cs] |  | 2021 |


## Implicit Neural Representations

- [https://github.com/vsitzmann/awesome-implicit-representations](https://github.com/vsitzmann/awesome-implicit-representations)

|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|


## NeRF
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| Ray Tracing Volume Densities |  SIGGRAPH  |  | 1984 |
| Efficient Ray Tracing of Volume Data | ACM Transactions on Graphics |  | 1990 |
| [Surface Light Fields for 3D Photography](https://doi.org/10.1145/344779.344925) | SIGGRAPH |  |  2000 |
| [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://arxiv.org/abs/2003.08934) | ECCV | [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)  | 2020 |
| [NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections](http://arxiv.org/abs/2008.02268) | arXiv:2008.02268 [cs] | [nerfw](https://github.com/PeterouZh/nerf_pl/tree/nerfw) | 2021 |
| [Modulated Periodic Activations for Generalizable Local Functional Representations](http://arxiv.org/abs/2104.03960) | arXiv:2104.03960 [cs] |  | 2021 |
| [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](http://arxiv.org/abs/2006.10739) | arXiv:2006.10739 [cs] |  | 2020 |
| [NeRF++: Analyzing and Improving Neural Radiance Fields](http://arxiv.org/abs/2010.07492) | arXiv:2010.07492 [cs] | [nerfplusplus](https://github.com/Kai-46/nerfplusplus) | 2020 |
| [Neural Volume Rendering: NeRF And Beyond](http://arxiv.org/abs/2101.05204) | arXiv:2101.05204 [cs] | [awesome-NeRF](https://github.com/yenchenlin/awesome-NeRF) | 2021 |
| [Editing Conditional Radiance Fields](http://arxiv.org/abs/2105.06466) | arXiv:2105.06466 [cs] | [editnerf](https://github.com/stevliu/editnerf) | 2021 |
| [Recursive-NeRF: An Efficient and Dynamically Growing NeRF](http://arxiv.org/abs/2105.09103) | arXiv:2105.09103 [cs] |  | 2021 |
| [Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes](http://arxiv.org/abs/2011.13084) | arXiv:2011.13084 [cs] |  | 2021 |
| [MVSNeRF: Fast Generalizable Radiance Field Reconstruction from Multi-View Stereo](http://arxiv.org/abs/2103.15595) | arXiv:2103.15595 [cs] | [mvsnerf](https://github.com/apchenstu/mvsnerf) | 2021 |
| [Neural Sparse Voxel Fields](http://arxiv.org/abs/2007.11571) | arXiv:2007.11571 [cs] | [NSVF](https://github.com/facebookresearch/NSVF) | 2021 |
| [Depth-Supervised NeRF: Fewer Views and Faster Training for Free](http://arxiv.org/abs/2107.02791) | arXiv:2107.02791 [cs] |  | 2021 |
| [Rethinking Positional Encoding](http://arxiv.org/abs/2107.02561) | arXiv:2107.02561 [cs] |  | 2021 |
| [Nerfies: Deformable Neural Radiance Fields](https://arxiv.org/abs/2011.12948v4) | 	arXiv:2011.12948 | [nerfies](https://github.com/google/nerfies) | 2020 |
| [D-NeRF: Neural Radiance Fields for Dynamic Scenes](http://arxiv.org/abs/2011.13961) | arXiv:2011.13961 [cs] | [D-NeRF](https://github.com/albertpumarola/D-NeRF) | 2020 |
| :heart: [NeRF--: Neural Radiance Fields Without Known Camera Parameters](http://arxiv.org/abs/2102.07064) | arXiv:2102.07064 [cs] | [improved-nerfmm](https://github.com/ventusff/improved-nerfmm)  | 2021 |


### Sine
|  Title  |   Venue  |Code|Year|Cite|
|:--------|:--------:|:--------:|:--------:|:--------:|
| :white_check_mark: [Implicit Neural Representations with Periodic Activation Functions](http://arxiv.org/abs/2006.09661) | NeurIPS |  | 2020 |
| :white_check_mark: [Modulated Periodic Activations for Generalizable Local Functional Representations](http://arxiv.org/abs/2104.03960) | arXiv:2104.03960 [cs] |  | 2021 |
| [Learned Initializations for Optimizing Coordinate-Based Neural Representations](http://arxiv.org/abs/2012.02189) | arXiv:2012.02189 [cs] | [nerf-meta](https://github.com/sanowar-raihan/nerf-meta) | 2021 |

### INR
|  Title  |   Venue  |Code|Year|Cite|
|:--------|:--------:|:--------:|:--------:|:--------:|
| [Adversarial Generation of Continuous Images](http://arxiv.org/abs/2011.12026) | arXiv:2011.12026 [cs] | [inr-gan](https://github.com/universome/inr-gan) | 2020 |
| [Image Generators with Conditionally-Independent Pixel Synthesis](http://arxiv.org/abs/2011.13775) | arXiv:2011.13775 [cs] | [CIPS](https://github.com/saic-mdal/CIPS) | 2020 |

### NeRF
|  Title  |   Venue  |Code|Year|Cite|
|:--------|:--------:|:--------:|:--------:|:--------:|
| [Pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis](http://arxiv.org/abs/2012.00926) | arXiv:2012.00926 [cs] | [pi-GAN](https://github.com/marcoamonteiro/pi-GAN) | 2021 | 19|
| [GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields](http://arxiv.org/abs/2011.12100) | CVPR | [giraffe](https://github.com/autonomousvision/giraffe) | 2021 |
| [GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis](http://arxiv.org/abs/2007.02442) | arXiv:2007.02442 [cs] |  | 2021 |
| [CAMPARI: Camera-Aware Decomposed Generative Neural Radiance Fields](http://arxiv.org/abs/2103.17269) | arXiv:2103.17269 [cs] |  | 2021 |
| [GNeRF: GAN-Based Neural Radiance Field without Posed Camera](http://arxiv.org/abs/2103.15606) | arXiv:2103.15606 [cs] |  | 2021 |


### 3D
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [StyleRig: Rigging StyleGAN for 3D Control over Portrait Images](http://arxiv.org/abs/2004.00121) | arXiv:2004.00121 [cs] |  | 2020 |
| [Exemplar-Based 3D Portrait Stylization](http://arxiv.org/abs/2104.14559) | arXiv:2104.14559 [cs] |  | 2021 |
| [Landmark Detection and 3D Face Reconstruction for Caricature Using a Nonlinear Parametric Model](http://arxiv.org/abs/2004.09190) | arXiv:2004.09190 [cs] |  | 2021 |
| [SofGAN: A Portrait Image Generator with Dynamic Styling](http://arxiv.org/abs/2007.03780) | arXiv:2007.03780 [cs] | [sofgan](https://github.com/apchenstu/sofgan) | 2021 |

### Face
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [Neural Head Reenactment with Latent Pose Descriptors](http://arxiv.org/abs/2004.12000) | CVPR | [latent-pose-reenactment](https://github.com/shrubb/latent-pose-reenactment) | 2020 |


## sdf
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](http://arxiv.org/abs/1901.05103) | arXiv:1901.05103 [cs] |  | 2019 |
| [Occupancy Networks: Learning 3D Reconstruction in Function Space](http://arxiv.org/abs/1812.03828) | arXiv:1812.03828 [cs] |  | 2019 |
| [PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization](http://arxiv.org/abs/1905.05172) | arXiv:1905.05172 [cs] |  | 2019 |



## 3D
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [Unsupervised Learning of Probably Symmetric Deformable 3D Objects from Images in the Wild](http://arxiv.org/abs/1911.11130) | arXiv:1911.11130 [cs] |[unsup3d](https://github.com/elliottwu/unsup3d)  | 2020 |
| [Do 2D GANs Know 3D Shape? Unsupervised 3D Shape Reconstruction from 2D Image GANs](http://arxiv.org/abs/2011.00844) | arXiv:2011.00844 [cs] | [GAN2Shape](https://github.com/XingangPan/GAN2Shape) | 2021 |
| [Lifting 2D StyleGAN for 3D-Aware Face Generation](http://arxiv.org/abs/2011.13126) | CVPR |  | 2021 |
| [Image GANs Meet Differentiable Rendering for Inverse Graphics and Interpretable 3D Neural Rendering](http://arxiv.org/abs/2010.09125) | arXiv:2010.09125 [cs] |  | 2021 |
| [Neural 3D Mesh Renderer](http://arxiv.org/abs/1711.07566) | CVPR |  | 2018 |
| [Fast-GANFIT: Generative Adversarial Network for High Fidelity 3D Face Reconstruction](http://arxiv.org/abs/2105.07474) | arXiv:2105.07474 [cs] |  | 2021 |
| [Learning to Stylize Novel Views](http://arxiv.org/abs/2105.13509) | arXiv:2105.13509 [cs] |  | 2021 |
| [Inverting Generative Adversarial Renderer for Face Reconstruction](http://arxiv.org/abs/2105.02431) | CVPR | [StyleRenderer](https://github.com/WestlyPark/StyleRenderer) | 2021 |
| [Learning to Aggregate and Personalize 3D Face from In-the-Wild Photo Collection](http://arxiv.org/abs/2106.07852) | arXiv:2106.07852 [cs] |  | 2021 |
| [Subdivision-Based Mesh Convolution Networks](http://arxiv.org/abs/2106.02285) | arXiv:2106.02285 [cs] |  | 2021 |
| [Learning to Aggregate and Personalize 3D Face from In-the-Wild Photo Collection](http://arxiv.org/abs/2106.07852) | CVPR |  | 2021 |
| [To Fit or Not to Fit: Model-Based Face Reconstruction and Occlusion Segmentation from Weak Supervision](http://arxiv.org/abs/2106.09614) | arXiv:2106.09614 [cs] |  | 2021 |
| [Unsupervised Learning of Depth and Depth-of-Field Effect from Natural Images with Aperture Rendering Generative Adversarial Networks](http://arxiv.org/abs/2106.13041) | arXiv:2106.13041 [cs, eess, stat] |  | 2021 |
| [DOVE: Learning Deformable 3D Objects by Watching Videos](http://arxiv.org/abs/2107.10844) | arXiv:2107.10844 [cs] |  | 2021 |


## Transformer
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [Training Data-Efficient Image Transformers & Distillation through Attention](http://arxiv.org/abs/2012.12877) | arXiv:2012.12877 [cs] | [deit](https://github.com/facebookresearch/deit) | 2020 |
| [Intriguing Properties of Vision Transformers](http://arxiv.org/abs/2105.10497) | arXiv:2105.10497 [cs] |  | 2021 |
| [CogView: Mastering Text-to-Image Generation via Transformers](http://arxiv.org/abs/2105.13290) | arXiv:2105.13290 [cs] |  | 2021 |
| [An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale](http://arxiv.org/abs/2010.11929) | arXiv:2010.11929 [cs] |  | 2021 |
| [Scaling Vision Transformers](http://arxiv.org/abs/2106.04560) | arXiv:2106.04560 [cs] |  | 2021 |
| [IA-RED$^2$: Interpretability-Aware Redundancy Reduction for Vision Transformers](http://arxiv.org/abs/2106.12620) | arXiv:2106.12620 [cs] |  | 2021 |
| [Rethinking and Improving Relative Position Encoding for Vision Transformer]() | ICCV |  | 2021 |
| [Go Wider Instead of Deeper](http://arxiv.org/abs/2107.11817) | arXiv:2107.11817 [cs] |  | 2021 |
| [A Unified Efficient Pyramid Transformer for Semantic Segmentation](http://arxiv.org/abs/2107.14209) | arXiv:2107.14209 [cs] |  | 2021 |
| :heart: [Conditional DETR for Fast Training Convergence](http://arxiv.org/abs/2108.06152) | ICCV |  | 2021 |


## ssl
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [Emerging Properties in Self-Supervised Vision Transformers](http://arxiv.org/abs/2104.14294) | arXiv:2104.14294 [cs] | [dino](https://github.com/facebookresearch/dino) | 2021 |
| [What Is Considered Complete for Visual Recognition?](http://arxiv.org/abs/2105.13978) | arXiv:2105.13978 [cs] |  | 2021 |


## DA
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [Semi-Supervised Domain Adaptation via Adaptive and Progressive Feature Alignment](http://arxiv.org/abs/2106.02845) | arXiv:2106.02845 [cs] |  | 2021 |
| [Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Semantic Segmentation](http://arxiv.org/abs/2101.10979) | arXiv:2101.10979 [cs] |  | 2021 |

## Data
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [Semi-Supervised Active Learning with Temporal Output Discrepancy](http://arxiv.org/abs/2107.14153) | ICCV |  | 2021 |


## CNN & BN
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [Beyond BatchNorm: Towards a General Understanding of Normalization in Deep Learning](http://arxiv.org/abs/2106.05956) | arXiv:2106.05956 [cs] |  | 2021 |
| [R-Drop: Regularized Dropout for Neural Networks](http://arxiv.org/abs/2106.14448) | arXiv:2106.14448 [cs] |  | 2021 |
| [Switchable Whitening for Deep Representation Learning](http://arxiv.org/abs/1904.09739) | ICCV |  | 2019 |
| [Positional Normalization](http://arxiv.org/abs/1907.04312) | arXiv:1907.04312 [cs] |  | 2019 |
| [On Feature Normalization and Data Augmentation](http://arxiv.org/abs/2002.11102) | arXiv:2002.11102 [cs, stat] |  | 2021 |
| [Channel Equilibrium Networks for Learning Deep Representation](http://arxiv.org/abs/2003.00214) | arXiv:2003.00214 [cs] |  | 2020 |
| [Representative Batch Normalization with Feature Calibration]() | CVPR |  | 2021 |
| [EPSANet: An Efficient Pyramid Squeeze Attention Block on Convolutional Neural Network](http://arxiv.org/abs/2105.14447) | arXiv:2105.14447 [cs] |  | 2021 |
| [Bias Loss for Mobile Neural Networks](http://arxiv.org/abs/2107.11170) | arXiv:2107.11170 [cs] |  | 2021 |
| [Compositional Models: Multi-Task Learning and Knowledge Transfer with Modular Networks](http://arxiv.org/abs/2107.10963) | arXiv:2107.10963 [cs] |  | 2021 |
| [Parametric Contrastive Learning](http://arxiv.org/abs/2107.12028) | ICCV |  | 2021 |
| [Log-Polar Space Convolution for Convolutional Neural Networks](http://arxiv.org/abs/2107.11943) | arXiv:2107.11943 [cs] |  | 2021 |
| [Decoupled Dynamic Filter Networks](http://arxiv.org/abs/2104.14107) | arXiv:2104.14107 [cs] |  | 2021 |
| [Spectral Leakage and Rethinking the Kernel Size in CNNs](http://arxiv.org/abs/2101.10143) | arXiv:2101.10143 [cs] |  | 2021 |

## Finetune
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| :heart: [How Transferable Are Features in Deep Neural Networks?](http://arxiv.org/abs/1411.1792) | arXiv:1411.1792 [cs] |  | 2014 |


## Positional Encoding
|  Title  |   Venue  |Code|Year|
|:--------|:--------:|:--------:|:--------:|
| [Positional Encoding as Spatial Inductive Bias in GANs](http://arxiv.org/abs/2012.05217) | arXiv:2012.05217 [cs] |  | 2020 |
| [Mind the Pad -- CNNs Can Develop Blind Spots](http://arxiv.org/abs/2010.02178) | arXiv:2010.02178 [cs, stat] |  | 2020 |
| [How Much Position Information Do Convolutional Neural Networks Encode?](http://arxiv.org/abs/2001.08248) | ICLR |  | 2020 |
| [On Translation Invariance in CNNs: Convolutional Layers Can Exploit Absolute Spatial Location](http://arxiv.org/abs/2003.07064) | CVPR |  | 2020 |
| [Rethinking and Improving Relative Position Encoding for Vision Transformer]() | ICCV |  | 2021 |
