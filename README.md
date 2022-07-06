# GAN-Inversion

A collection of papers I am interested in.

## Awesome

- https://ait.ethz.ch/index.php
- https://liuyebin.com/student.html
- https://virtualhumans.mpi-inf.mpg.de/
- https://ps.is.mpg.de/publications
- https://www.mpi-inf.mpg.de/departments/visual-computing-and-artificial-intelligence/publications
- https://ait.ethz.ch/people/hilliges/
- https://vlg.inf.ethz.ch/publications.html

## Renderer

- https://github.com/eth-ait/aitviewer

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
- [View_Neural_Talking_Head_Synthesis](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis)
- [Anime2Sketch](https://github.com/Mukosame/Anime2Sketch)

### 3D

- [face3d](https://github.com/YadiraF/face3d)
- [DECA](https://github.com/YadiraF/DECA)

### Tools

- [bokeh](https://github.com/bokeh/bokeh)
- [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)
- [label-studio](https://github.com/heartexlabs/label-studio)
- [streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas)
- [face-alignment](https://github.com/1adrianb/face-alignment)
- [remove images background](https://github.com/danielgatis/rembg)

### GUI

- https://github.com/gradio-app/gradio

### StyleGAN

- https://github.com/justinpinkney/awesome-pretrained-stylegan2
- https://github.com/justinpinkney/awesome-pretrained-stylegan3
- [generative-evaluation-prdc](https://github.com/clovaai/generative-evaluation-prdc)

### Style transfer

- [style-transfer-pytorch](https://github.com/crowsonkb/style-transfer-pytorch)
- [Stylebank-exp](https://github.com/PeterouZh/Stylebank-exp)

### Art

- https://github.com/fogleman/primitive

### Anime

- https://github.com/TachibanaYoshino/AnimeGAN
- https://github.com/TachibanaYoshino/AnimeGANv2

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

| Title                                                                                                                                                      |            Venue            |                             Code                              | Year |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------: | :-----------------------------------------------------------: | :--: |
| [Perceptual Gradient Networks](http://arxiv.org/abs/2105.01957)                                                                                            |    arXiv:2105.01957 [cs]    |                                                               | 2021 |
| [InfinityGAN: Towards Infinite-Resolution Image Synthesis](http://arxiv.org/abs/2104.03963)                                                                |    arXiv:2104.03963 [cs]    |                                                               | 2021 |
| [Aliasing Is Your Ally: End-to-End Super-Resolution from Raw Image Bursts](http://arxiv.org/abs/2104.06191)                                                | arXiv:2104.06191 [cs, eess] |                                                               | 2021 |
| [StylePeople: A Generative Model of Fullbody Human Avatars](http://arxiv.org/abs/2104.08363)                                                               |    arXiv:2104.08363 [cs]    |                                                               | 2021 |
| [Cross-Domain and Disentangled Face Manipulation with 3D Guidance](http://arxiv.org/abs/2104.11228)                                                        |    arXiv:2104.11228 [cs]    |                                                               | 2021 |
| [On Buggy Resizing Libraries and Surprising Subtleties in FID Calculation](http://arxiv.org/abs/2104.11222)                                                |    arXiv:2104.11222 [cs]    |                                                               | 2021 |
| [FDA: Fourier Domain Adaptation for Semantic Segmentation](http://arxiv.org/abs/2004.05498)                                                                |    arXiv:2004.05498 [cs]    |         [github](https://github.com/YanchaoYang/FDA)          | 2020 |
| [StyleMapGAN: Exploiting Spatial Dimensions of Latent in GAN for Real-Time Image Editing](http://arxiv.org/abs/2104.14754)                                 |            CVPR             |                                                               | 2021 |
| [Learning a Deep Reinforcement Learning Policy Over the Latent Space of a Pre-Trained GAN for Semantic Age Manipulation](http://arxiv.org/abs/2011.00954)  |    arXiv:2011.00954 [cs]    |                                                               | 2021 |
| [GANalyze: Toward Visual Definitions of Cognitive Image Properties](http://arxiv.org/abs/1906.10112)                                                       |    arXiv:1906.10112 [cs]    |                                                               | 2019 |
| [On the “Steerability” of Generative Adversarial Networks](http://arxiv.org/abs/1907.07171)                                                                |    arXiv:1907.07171 [cs]    |                                                               | 2020 |
| [Pose-Controllable Talking Face Generation by Implicitly Modularized Audio-Visual Representation](http://arxiv.org/abs/2104.11116)                         | arXiv:2104.11116 [cs, eess] |                                                               | 2021 |
| [Unsupervised Image-to-Image Translation via Pre-Trained StyleGAN2 Network](http://arxiv.org/abs/2010.05713)                                               |    arXiv:2010.05713 [cs]    | [github](https://github.com/HideUnderBush/UI2I_via_StyleGAN2) | 2020 |
| [DatasetGAN: Efficient Labeled Data Factory with Minimal Human Effort](http://arxiv.org/abs/2104.06490)                                                    |    arXiv:2104.06490 [cs]    |                                                               | 2021 |
| [Anycost GANs for Interactive Image Synthesis and Editing](https://arxiv.org/abs/2103.03243v1)                                                             |            CVPR             |                                                               | 2021 |
| [Semantic Segmentation with Generative Models: Semi-Supervised Learning and Strong Out-of-Domain Generalization](http://arxiv.org/abs/2104.05833)          |    arXiv:2104.05833 [cs]    |                                                               | 2021 |
| [Positional Encoding as Spatial Inductive Bias in GANs](http://arxiv.org/abs/2012.05217)                                                                   |    arXiv:2012.05217 [cs]    |                                                               | 2020 |
| [An Empirical Study of the Effects of Sample-Mixing Methods for Efficient Training of Generative Adversarial Networks](https://arxiv.org/abs/2104.03535v1) |  arXiv:2104.03535 [cs.CV]   |                                                               | 2021 |
| [Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks](https://ieeexplore.ieee.org/document/9150840/)                           |            CVPRW            |       [github](https://github.com/haofanwang/Score-CAM)       | 2020 |
| [Image Demoireing with Learnable Bandpass Filters](http://arxiv.org/abs/2004.00406)                                                                        |    arXiv:2004.00406 [cs]    |                                                               | 2020 |
| [Unveiling the Potential of Structure Preserving for Weakly Supervised Object Localization](http://arxiv.org/abs/2103.04523)                               |    arXiv:2103.04523 [cs]    |                                                               | 2021 |
| [LatentCLR: A Contrastive Learning Approach for Unsupervised Discovery of Interpretable Directions](http://arxiv.org/abs/2104.00820)                       |    arXiv:2104.00820 [cs]    |                                                               | 2021 |
| [Generating Images with Sparse Representations](http://arxiv.org/abs/2103.03841)                                                                           | arXiv:2103.03841 [cs, stat] |                                                               | 2021 |
| [PiCIE: Unsupervised Semantic Segmentation Using Invariance and Equivariance in Clustering](http://arxiv.org/abs/2103.17070)                               |            CVPR             |                                                               | 2021 |
| [Dual Contrastive Loss and Attention for GANs](https://arxiv.org/abs/2103.16748v1)                                                                         |  arXiv:2103.16748 [cs.CV]   |                                                               | 2021 |
| [Unsupervised Disentanglement of Linear-Encoded Facial Semantics](https://arxiv.org/abs/2103.16605v1)                                                      |            CVPR             |                                                               | 2021 |
| [Emergence of Object Segmentation in Perturbed Generative Models](http://arxiv.org/abs/1905.12663)                                                         |    arXiv:1905.12663 [cs]    |    [github](https://github.com/adambielski/perturbed-seg)     | 2019 |
| [Unsupervised Discovery of DisentangledManifolds in GANs](http://arxiv.org/abs/2011.11842)                                                                 |    arXiv:2011.11842 [cs]    |   [github](https://github.com/anvoynov/GANLatentDiscovery)    | 2020 |
| [StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery](http://arxiv.org/abs/2103.17249)                                                                 |    arXiv:2103.17249 [cs]    |      [github](https://github.com/orpatashnik/StyleCLIP)       | 2021 |
| [Few-Shot Semantic Image Synthesis Using StyleGAN Prior](http://arxiv.org/abs/2103.14877)                                                                  |    arXiv:2103.14877 [cs]    |                                                               | 2021 |

## Disentanglement

| Title                                                                                                                      |         Venue         |                            Code                            | Year |
| :------------------------------------------------------------------------------------------------------------------------- | :-------------------: | :--------------------------------------------------------: | :--: |
| [GANSpace: Discovering Interpretable GAN Controls](http://arxiv.org/abs/2004.02546)                                        | arXiv:2004.02546 [cs] |      [GANSpace](https://github.com/harskish/ganspace)      | 2020 |
| [Interpreting the Latent Space of GANs for Semantic Face Editing](http://arxiv.org/abs/1907.10786)                         |         CVPR          |  [InterFaceGAN](https://github.com/genforce/interfacegan)  | 2020 |
| [Closed-Form Factorization of Latent Semantics in GANs](http://arxiv.org/abs/2007.06600)                                   | arXiv:2007.06600 [cs] |          [sefa](https://github.com/genforce/sefa)          | 2020 |
| [StyleSpace Analysis: Disentangled Controls for StyleGAN Image Generation](http://arxiv.org/abs/2011.12799)                | arXiv:2011.12799 [cs] | [StyleSpace](https://github.com/xrenaa/StyleSpace-pytorch) | 2020 |
| [Unsupervised Image Transformation Learning via Generative Adversarial Networks](http://arxiv.org/abs/2103.07751)          | arXiv:2103.07751 [cs] |        [github](https://github.com/genforce/trgan)         | 2021 |
| [Resolution Dependent GAN Interpolation for Controllable Image Synthesis Between Domains](http://arxiv.org/abs/2010.05334) | arXiv:2010.05334 [cs] |    [toonify](https://github.com/justinpinkney/toonify)     | 2020 |
| [WarpedGANSpace: Finding Non-Linear RBF Paths in GAN Latent Space](http://arxiv.org/abs/2109.13357)                        | arXiv:2109.13357 [cs] |                                                            | 2021 |
| [Discovering Interpretable Latent Space Directions of GANs beyond Binary Attributes] CVPR                                  |                       |                            2021                            |

### Semantic hierarchy

| Title                                                                                                                |         Venue         | Code | Year |
| :------------------------------------------------------------------------------------------------------------------- | :-------------------: | :--: | :--: |
| [Semantic Hierarchy Emerges in Deep Generative Representations for Scene Synthesis](http://arxiv.org/abs/1911.09267) | arXiv:1911.09267 [cs] |      | 2020 |

## Inversion

### Optimization

| Title                                                                                                                                                                                                                                                |                     Venue                     |                                 Code                                 | Year |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------: | :------------------------------------------------------------------: | :--: |
| [Image2StyleGAN++: How to Edit the Embedded Images?](http://arxiv.org/abs/1911.11544)                                                                                                                                                                |             arXiv:1911.11544 [cs]             |                                                                      | 2020 |
| [Image2StyleGAN: How to Embed Images Into the StyleGAN Latent Space?](http://arxiv.org/abs/1904.03189)                                                                                                                                               |                     ICCV                      |                                                                      | 2019 |
| [Inverting The Generator Of A Generative Adversarial Network](http://arxiv.org/abs/1611.05644)                                                                                                                                                       |             arXiv:1611.05644 [cs]             |                                                                      | 2016 |
| [Feature-Based Metrics for Exploring the Latent Space of Generative Models](https://openreview.net/forum?id=BJslDBkwG)                                                                                                                               |                     ICLRW                     |                                                                      | 2018 |
| [Understanding Deep Image Representations by Inverting Them](http://arxiv.org/abs/1412.0035)                                                                                                                                                         |                     CVPR                      |                                                                      | 2015 |
| [Dreaming to Distill: Data-Free Knowledge Transfer via DeepInversion](http://arxiv.org/abs/1912.08795)                                                                                                                                               |          arXiv:1912.08795 [cs, stat]          |       [DeepInversion](https://github.com/NVlabs/DeepInversion)       | 2020 |
| [IMAGINE: Image Synthesis by Image-Guided Model Inversion](http://arxiv.org/abs/2104.05895)                                                                                                                                                          |             arXiv:2104.05895 [cs]             |                                                                      | 2021 |
| [Image Processing Using Multi-Code GAN Prior](http://arxiv.org/abs/1912.07116)                                                                                                                                                                       |                     CVPR                      |          [mGANprior](https://github.com/genforce/mganprior)          | 2020 |
| [Generative Visual Manipulation on the Natural Image Manifold](http://arxiv.org/abs/1609.03552)                                                                                                                                                      |                     ECCV                      |                                                                      | 2018 |
| [GAN Dissection: Visualizing and Understanding Generative Adversarial Networks](http://arxiv.org/abs/1811.10597)                                                                                                                                     |             arXiv:1811.10597 [cs]             |                                                                      | 2018 |
| [GAN-Based Projector for Faster Recovery with Convergence Guarantees in Linear Inverse Problems](http://arxiv.org/abs/1902.09698)                                                                                                                    |       arXiv:1902.09698 [cs, eess, stat]       |                                                                      | 2019 |
| [Your Local GAN: Designing Two Dimensional Local Attention Mechanisms for Generative Models](http://openaccess.thecvf.com/content_CVPR_2020/html/Daras_Your_Local_GAN_Designing_Two_Dimensional_Local_Attention_Mechanisms_for_CVPR_2020_paper.html) |                     CVPR                      |                                                                      | 2020 |
| [Rewriting a Deep Generative Model](http://arxiv.org/abs/2007.15646)                                                                                                                                                                                 |             arXiv:2007.15646 [cs]             |                                                                      | 2020 |
| [Transforming and Projecting Images into Class-Conditional Generative Networks](http://arxiv.org/abs/2005.01703)                                                                                                                                     |             arXiv:2005.01703 [cs]             |                                                                      | 2020 |
| [StyleGAN2 Distillation for Feed-Forward Image Manipulation](https://arxiv.org/abs/2003.03581v2)                                                                                                                                                     |           arXiv:2003.03581 [cs.CV]            |                                                                      | 2020 |
| [On the “Steerability” of Generative Adversarial Networks](http://arxiv.org/abs/1907.07171)                                                                                                                                                          |             arXiv:1907.07171 [cs]             |                                                                      | 2020 |
| [Unsupervised Discovery of DisentangledManifolds in GANs](http://arxiv.org/abs/2011.11842)                                                                                                                                                           |             arXiv:2011.11842 [cs]             |                                                                      | 2020 |
| [PIE: Portrait Image Embedding for Semantic Control](http://arxiv.org/abs/2009.09485)                                                                                                                                                                |             arXiv:2009.09485 [cs]             |                                                                      | 2020 |
| [GANSpace: Discovering Interpretable GAN Controls](http://arxiv.org/abs/2004.02546)                                                                                                                                                                  |                    NeurIPS                    |                                                                      | 2020 |
| [When and How Can Deep Generative Models Be Inverted?](http://arxiv.org/abs/2006.15555)                                                                                                                                                              |          arXiv:2006.15555 [cs, stat]          |                                                                      | 2020 |
| [Style Intervention: How to Achieve Spatial Disentanglement with Style-Based Generators?](http://arxiv.org/abs/2011.09699)                                                                                                                           |             arXiv:2011.09699 [cs]             |                                                                      | 2020 |
| [StyleSpace Analysis: Disentangled Controls for StyleGAN Image Generation](http://arxiv.org/abs/2011.12799)                                                                                                                                          |             arXiv:2011.12799 [cs]             |                                                                      | 2020 |
| [Navigating the GAN Parameter Space for Semantic Image Editing](http://arxiv.org/abs/2011.13786)                                                                                                                                                     |             arXiv:2011.13786 [cs]             |                                                                      | 2021 |
| [Mask-Guided Discovery of Semantic Manifolds in Generative Models](http://arxiv.org/abs/2105.07273)                                                                                                                                                  |             arXiv:2105.07273 [cs]             | [masked-gan-manifold](https://github.com/bmolab/masked-gan-manifold) | 2021 |
| [StyleFlow: Attribute-Conditioned Exploration of StyleGAN-Generated Images Using Conditional Continuous Normalizing Flows](http://arxiv.org/abs/2008.02401)                                                                                          |             arXiv:2008.02401 [cs]             |        [StyleFlow](https://github.com/RameenAbdal/StyleFlow)         | 2020 |
| [Disentangled Face Attribute Editing via Instance-Aware Latent Space Search](http://arxiv.org/abs/2105.12660)                                                                                                                                        |             arXiv:2105.12660 [cs]             |                                                                      | 2021 |
| [Barbershop: GAN-Based Image Compositing Using Segmentation Masks](http://arxiv.org/abs/2106.01505)                                                                                                                                                  |             arXiv:2106.01505 [cs]             |                                                                      | 2021 |
| [Unsupervised Discovery of Interpretable Directions in the GAN Latent Space](http://arxiv.org/abs/2002.03754)                                                                                                                                        |          arXiv:2002.03754 [cs, stat]          | [GANLatentDiscovery](https://github.com/anvoynov/GANLatentDiscovery) | 2020 |
| [Pivotal Tuning for Latent-Based Editing of Real Images](http://arxiv.org/abs/2106.05744)                                                                                                                                                            |             arXiv:2106.05744 [cs]             |              [PTI](https://github.com/danielroich/PTI)               | 2021 |
| [Editing in Style: Uncovering the Local Semantics of GANs](http://arxiv.org/abs/2004.14367)                                                                                                                                                          |                     CVPR                      |                                                                      | 2020 |
| [Retrieve in Style: Unsupervised Facial Feature Transfer and Retrieval](http://arxiv.org/abs/2107.06256)                                                                                                                                             |             arXiv:2107.06256 [cs]             |    [RetrieveInStyle](https://github.com/mchong6/RetrieveInStyle)     | 2021 |
| [StyleCariGAN: Caricature Generation via StyleGAN Feature Map Modulation](http://arxiv.org/abs/2107.04331)                                                                                                                                           |             arXiv:2107.04331 [cs]             |                                                                      | 2021 |
| [A Simple Baseline for StyleGAN Inversion](http://arxiv.org/abs/2104.07661)                                                                                                                                                                          |             arXiv:2104.07661 [cs]             |                                                                      | 2021 |
| [From Continuity to Editability: Inverting GANs with Consecutive Images](http://arxiv.org/abs/2107.13812)                                                                                                                                            |             arXiv:2107.13812 [cs]             |                                                                      | 2021 |
| [AgileGAN: Stylizing Portraits by Inversion-Consistent Transfer Learning]()                                                                                                                                                                          | ACM Transactions on Graphics (Proc. SIGGRAPH) |                                                                      | 2021 |
| [Talk-to-Edit: Fine-Grained Facial Editing via Dialog](http://arxiv.org/abs/2109.04425)                                                                                                                                                              |                     ICCV                      |       [Talk-to-Edit](https://github.com/yumingj/Talk-to-Edit)        | 2021 |
| [Improved StyleGAN Embedding: Where Are the Good Latents?](http://arxiv.org/abs/2012.09036)                                                                                                                                                          |             arXiv:2012.09036 [cs]             |                [II2S](https://github.com/ZPdesu/II2S)                | 2021 |
| [EditGAN: High-Precision Semantic Image Editing](https://arxiv.org/abs/2111.03186v1)                                                                                                                                                                 |                                               |    [editGAN_release](https://github.com/nv-tlabs/editGAN_release)    | 2021 |
| [Grasping the Arrow of Time from the Singularity: Decoding Micromotion in Low-Dimensional Latent Spaces from StyleGAN](http://arxiv.org/abs/2204.12696)                                                                                              |             arXiv:2204.12696 [cs]             |                                                                      | 2022 |
| [Spatially-Adaptive Multilayer Selection for GAN Inversion and Editing](http://arxiv.org/abs/2206.08357) | CVPR | [sam_inversion](https://github.com/adobe-research/sam_inversion)  | arXiv. 2022 |


### Encoder

| Title                                                                                                                                                  |            Venue            |                                 Code                                 | Year |
| :----------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------: | :------------------------------------------------------------------: | :--: |
| [GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution](http://arxiv.org/abs/2012.00739)                                               |    arXiv:2012.00739 [cs]    |                              [GLEAN]()                               | 2020 |
| [Swapping Autoencoder for Deep Image Manipulation](http://arxiv.org/abs/2007.00653)                                                                    |    arXiv:2007.00653 [cs]    | [github](https://github.com/rosinality/swapping-autoencoder-pytorch) | 2020 |
| [In-Domain GAN Inversion for Real Image Editing](http://arxiv.org/abs/2004.00049)                                                                      |            ECCV             |                                                                      | 2020 |
| [ReStyle: A Residual-Based StyleGAN Encoder via Iterative Refinement](http://arxiv.org/abs/2104.02699)                                                 |    arXiv:2104.02699 [cs]    |      [ReStyle](https://github.com/yuval-alaluf/restyle-encoder)      | 2021 |
| [Interpreting the Latent Space of GANs for Semantic Face Editing](http://arxiv.org/abs/1907.10786)                                                     |            CVPR             |                                                                      | 2020 |
| [Face Identity Disentanglement via Latent Space Mapping](http://arxiv.org/abs/2005.07728)                                                              |    arXiv:2005.07728 [cs]    |                                                                      | 2020 |
| [Collaborative Learning for Faster StyleGAN Embedding](http://arxiv.org/abs/2007.01758)                                                                |    arXiv:2007.01758 [cs]    |                                                                      | 2020 |
| [Unsupervised Discovery of DisentangledManifolds in GANs](http://arxiv.org/abs/2011.11842)                                                             |    arXiv:2011.11842 [cs]    |                                                                      | 2020 |
| [Generative Hierarchical Features from Synthesizing Images](http://arxiv.org/abs/2007.10379)                                                           |    arXiv:2007.10379 [cs]    |                                                                      | 2020 |
| [One Shot Face Swapping on Megapixels](http://arxiv.org/abs/2105.04932)                                                                                |    arXiv:2105.04932 [cs]    |                                                                      | 2021 |
| [GAN Prior Embedded Network for Blind Face Restoration in the Wild](https://arxiv.org/abs/2105.06070v1)                                                |            2021             |
| [Adversarial Latent Autoencoders](http://openaccess.thecvf.com/content_CVPR_2020/html/Pidhorskyi_Adversarial_Latent_Autoencoders_CVPR_2020_paper.html) |            CVPR             |              [ALAE](https://github.com/podgorskiy/ALAE)              | 2020 |
| [Encoding in Style: A StyleGAN Encoder for Image-to-Image Translation](http://arxiv.org/abs/2008.00951)                                                |    arXiv:2008.00951 [cs]    |         [psp](https://github.com/eladrich/pixel2style2pixel)         | 2021 |
| [Designing an Encoder for StyleGAN Image Manipulation](http://arxiv.org/abs/2102.02766)                                                                |    arXiv:2102.02766 [cs]    |    [encoder4editing](https://github.com/omertov/encoder4editing)     | 2021 |
| [A Latent Transformer for Disentangled and Identity-Preserving Face Editing](http://arxiv.org/abs/2106.11895)                                          |    arXiv:2106.11895 [cs]    |                                                                      | 2021 |
| [ShapeEditer: A StyleGAN Encoder for Face Swapping](http://arxiv.org/abs/2106.13984)                                                                   |    arXiv:2106.13984 [cs]    |                                                                      | 2021 |
| [Force-in-Domain GAN Inversion](http://arxiv.org/abs/2107.06050)                                                                                       | arXiv:2107.06050 [cs, eess] |                                                                      | 2021 |
| [StyleFusion: A Generative Model for Disentangling Spatial Segments](http://arxiv.org/abs/2107.07437)                                                  |    arXiv:2107.07437 [cs]    |                                                                      | 2021 |
| [Perceptually Validated Precise Local Editing for Facial Action Units with StyleGAN](http://arxiv.org/abs/2107.12143)                                  |    arXiv:2107.12143 [cs]    |                                                                      | 2021 |
| [StyleGAN2 Distillation for Feed-Forward Image Manipulation](https://arxiv.org/abs/2003.03581v2)                                                       |  arXiv:2003.03581 [cs.CV]   |                                                                      | 2020 |
| [GAN Inversion for Out-of-Range Images with Geometric Transformations](http://arxiv.org/abs/2108.08998)                                                |            ICCV             |                                                                      | 2021 |
| :heart: [DyStyle: Dynamic Neural Network for Multi-Attribute-Conditioned Style Editing](http://arxiv.org/abs/2109.10737)                               |    arXiv:2109.10737 [cs]    |            [DyStyle](https://github.com/phycvgan/DyStyle)            | 2021 |
| [High-Fidelity GAN Inversion for Image Attribute Editing](http://arxiv.org/abs/2109.06590)                                                             |    arXiv:2109.06590 [cs]    |                                                                      | 2021 |
| :heart: [Few-Shot Knowledge Transfer for Fine-Grained Cartoon Face Generation](http://arxiv.org/abs/2007.13332)                                        |    arXiv:2007.13332 [cs]    |                                                                      | 2020 |

### Hybrid optimization

| Title                                                                                           |            Venue             | Code | Year |
| :---------------------------------------------------------------------------------------------- | :--------------------------: | :--: | :--: |
| [Generative Visual Manipulation on the Natural Image Manifold](http://arxiv.org/abs/1609.03552) |             ECCV             |      | 2018 |
| [Semantic Photo Manipulation with a Generative Image Prior](https://arxiv.org/abs/2005.07727)   | ACM Transactions on Graphics |      | 2019 |
| [Seeing What a GAN Cannot Generate](http://arxiv.org/abs/1910.11626)                            | arXiv:1910.11626 [cs, eess]  |      | 2019 |
| [In-Domain GAN Inversion for Real Image Editing](http://arxiv.org/abs/2004.00049)               |             ECCV             |      | 2020 |

### Without optimization

| Title                                                                                                                      |         Venue         |                   Code                   | Year |
| :------------------------------------------------------------------------------------------------------------------------- | :-------------------: | :--------------------------------------: | :--: |
| [Closed-Form Factorization of Latent Semantics in GANs](http://arxiv.org/abs/2007.06600)                                   | arXiv:2007.06600 [cs] |                                          | 2020 |
| [GAN “Steerability” without Optimization](http://arxiv.org/abs/2012.05328)                                                 | arXiv:2012.05328 [cs] |                                          | 2021 |
| [Low-Rank Subspaces in GANs](http://arxiv.org/abs/2106.04488)                                                              | arXiv:2106.04488 [cs] |                                          | 2021 |
| [LARGE: Latent-Based Regression through GAN Semantics](http://arxiv.org/abs/2107.11186)                                    | arXiv:2107.11186 [cs] |                                          | 2021 |
| [Orthogonal Jacobian Regularization for Unsupervised Disentanglement in Image Generation](http://arxiv.org/abs/2108.07668) |         ICCV          |                                          | 2021 |
| [Controllable and Compositional Generation with Latent-Space Energy-Based Models](http://arxiv.org/abs/2110.10873)         |        NeurIPS        |  [LACE](https://github.com/NVlabs/LACE)  | 2021 |
| [Do Generative Models Know Disentanglement? Contrastive Learning Is All You Need](http://arxiv.org/abs/2102.10543)         | arXiv:2102.10543 [cs] | [DisCo](https://github.com/xrenaa/DisCo) | 2021 |

### DGP

| Title                                                                                                                                                                                                                                                         |            Venue             |                                           Code                                           | Year |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------: | :--------------------------------------------------------------------------------------: | :--: |
| :heavy_check_mark: [Exploiting Deep Generative Prior for Versatile Image Restoration and Manipulation](http://arxiv.org/abs/2003.13659)                                                                                                                                          |             ECCV             |                [DGP](https://github.com/XingangPan/deep-generative-prior)                | 2020 |
| :heavy_check_mark: [PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models](http://openaccess.thecvf.com/content_CVPR_2020/html/Menon_PULSE_Self-Supervised_Photo_Upsampling_via_Latent_Space_Exploration_of_Generative_CVPR_2020_paper.html) |             CVPR             |                                        [PULSE]()                                         | 2020 |
| :heavy_check_mark: [GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution](http://arxiv.org/abs/2012.00739)                                                                                                                                                      |    arXiv:2012.00739 [cs]     |                                                                                          | 2020 |
| [Unsupervised Portrait Shadow Removal via Generative Priors](http://arxiv.org/abs/2108.03466)                                                                                                                                                                 |    arXiv:2108.03466 [cs]     |                                                                                          | 2021 |
| [Towards Real-World Blind Face Restoration with Generative Facial Prior](http://arxiv.org/abs/2101.04061)                                                                                                                                                     |             CVPR             |                      [GFPGAN](https://github.com/TencentARC/GFPGAN)                      | 2021 |
| [Towards Vivid and Diverse Image Colorization with Generative Color Prior](http://arxiv.org/abs/2108.08826)                                                                                                                                                   |             ICCV             |                                                                                          | 2021 |
| [Self-Validation: Early Stopping for Single-Instance Deep Generative Priors]()                                                                                                                                                                                |   arXiv:2110.12271 [cs.CV]   |                                                                                          | 2021 |
| [One-Shot Generative Domain Adaptation](http://arxiv.org/abs/2111.09876)                                                                                                                                                                                      |    arXiv:2111.09876 [cs]     |                                                                                          | 2021 |
| :heart: [Time-Travel Rephotography]()                                                                                                                                                                                                                         | ACM Transactions on Graphics | [code](https://github.com/Time-Travel-Rephotography/Time-Travel-Rephotography.github.io) | 2021 |

### Cls

| Title                                                                                                                   |         Venue         | Code | Year |
| :---------------------------------------------------------------------------------------------------------------------- | :-------------------: | :--: | :--: |
| [Contrastive Model Inversion for Data-Free Knowledge Distillation](http://arxiv.org/abs/2105.08584)                     | arXiv:2105.08584 [cs] |      | 2021 |
| [Generative Models as a Data Source for Multiview Representation Learning](http://arxiv.org/abs/2106.05258)             | arXiv:2106.05258 [cs] |      | 2021 |
| [Inverting and Understanding Object Detectors](http://arxiv.org/abs/2106.13933)                                         | arXiv:2106.13933 [cs] |      | 2021 |
| [Deep Neural Networks Are Surprisingly Reversible: A Baseline for Zero-Shot Inversion](http://arxiv.org/abs/2107.06304) | arXiv:2107.06304 [cs] |      | 2021 |
| [Ensembling with Deep Generative Views](http://arxiv.org/abs/2104.14551)                                                | arXiv:2104.14551 [cs] |      | 2021 |

### Change pose implicitly

| Title                                                                                                                |         Venue         |                       Code                       | Year |
| :------------------------------------------------------------------------------------------------------------------- | :-------------------: | :----------------------------------------------: | :--: |
| [On the “Steerability” of Generative Adversarial Networks](http://arxiv.org/abs/1907.07171)                          | arXiv:1907.07171 [cs] |                                                  | 2020 |
| [Interpreting the Latent Space of GANs for Semantic Face Editing](http://arxiv.org/abs/1907.10786)                   |         CVPR          |                                                  | 2020 |
| [GANSpace: Discovering Interpretable GAN Controls](http://arxiv.org/abs/2004.02546)                                  | arXiv:2004.02546 [cs] | [GANSpace](https://github.com/harskish/ganspace) | 2020 |
| [Closed-Form Factorization of Latent Semantics in GANs](http://arxiv.org/abs/2007.06600)                             | arXiv:2007.06600 [cs] |     [sefa](https://github.com/genforce/sefa)     | 2020 |
| [StyleGAN of All Trades: Image Manipulation with Only Pretrained StyleGAN](http://arxiv.org/abs/2111.01619)          | arXiv:2111.01619 [cs] |                                                  | 2021 |
| [Using Latent Space Regression to Analyze and Leverage Compositionality in GANs](https://arxiv.org/abs/2103.10426v1) |         ICLR          |                                                  | 2021 |

## Survey

| Title                                                      |         Venue         | Code | Year |
| :--------------------------------------------------------- | :-------------------: | :--: | :--: |
| [GAN Inversion: A Survey](http://arxiv.org/abs/2101.05278) | arXiv:2101.05278 [cs] |      | 2021 |

## GANs

### NeurIPS 2021

| Title                                                                                              |  Venue  | Code | Year |
| :------------------------------------------------------------------------------------------------- | :-----: | :--: | :--: |
| [Rebooting ACGAN: Auxiliary Classifier GANs with Stable Training](http://arxiv.org/abs/2111.01118) | NeurIPS |      | 2021 |

### Theory

| Title                                                                                                       |               Venue               | Code | Year |
| :---------------------------------------------------------------------------------------------------------- | :-------------------------------: | :--: | :--: |
| :white_check_mark: [Towards a Better Global Loss Landscape of GANs](http://arxiv.org/abs/2011.04926)        |              NeurIPS              |      | 2020 |
| [On the Benefit of Width for Neural Networks: Disappearance of Bad Basins](http://arxiv.org/abs/1812.11039) | arXiv:1812.11039 [cs, math, stat] |      | 2021 |

### Regs

| Title                                                                                                 | Venue | Code | Year |
| :---------------------------------------------------------------------------------------------------- | :---: | :--: | :--: |
| [The Hessian Penalty: A Weak Prior for Unsupervised Disentanglement](http://arxiv.org/abs/2008.10599) | ECCV  |      | 2020 |

### Detection

| Title                                                                                              |         Venue         | Code | Year |
| :------------------------------------------------------------------------------------------------- | :-------------------: | :--: | :--: |
| [Self-Supervised Object Detection via Generative Image Synthesis](http://arxiv.org/abs/2110.09848) | arXiv:2110.09848 [cs] |      | 2021 |

### StyleGANs

| Title                                                                                                                                                   |               Venue               |                                                       Code                                                       | Year |
| :------------------------------------------------------------------------------------------------------------------------------------------------------ | :-------------------------------: | :--------------------------------------------------------------------------------------------------------------: | :--: |
| [A Style-Based Generator Architecture for Generative Adversarial Networks](http://arxiv.org/abs/1812.04948)                                             |               CVPR                |                                                                                                                  | 2019 |
| [Analyzing and Improving the Image Quality of StyleGAN](http://arxiv.org/abs/1912.04958)                                                                | arXiv:1912.04958 [cs, eess, stat] |                                                                                                                  | 2019 |
| [Training Generative Adversarial Networks with Limited Data](http://arxiv.org/abs/2006.06676)                                                           |    arXiv:2006.06676 [cs, stat]    |                                                                                                                  | 2020 |
| [Deceive D: Adaptive Pseudo Augmentation for GAN Training with Limited Data](https://openreview.net/forum?id=spjlJ4jeM_)                                |              NeurIPS              |                                                                                                                  | 2021 |
| [Alias-Free Generative Adversarial Networks](http://arxiv.org/abs/2106.12423)                                                                           |    arXiv:2106.12423 [cs, stat]    | [alias-free-gan](https://github.com/NVlabs/alias-free-gan), [rep2](https://github.com/duskvirkus/alias-free-gan) | 2021 |
| [Transforming the Latent Space of StyleGAN for Real Face Editing](http://arxiv.org/abs/2105.14230)                                                      |       arXiv:2105.14230 [cs]       |                          [TransStyleGAN](https://github.com/AnonSubm2021/TransStyleGAN)                          | 2021 |
| [MobileStyleGAN: A Lightweight Convolutional Neural Network for High-Fidelity Image Synthesis](http://arxiv.org/abs/2104.04767)                         |    arXiv:2104.04767 [cs, eess]    |                       [MobileStyleGAN](https://github.com/bes-dev/MobileStyleGAN.pytorch)                        | 2021 |
| [Few-Shot Image Generation via Cross-Domain Correspondence](http://arxiv.org/abs/2104.06820)                                                            |               CVPR                |                [few-shot-gan-adaptation](https://github.com/utkarshojha/few-shot-gan-adaptation)                 | 2021 |
| [EigenGAN: Layer-Wise Eigen-Learning for GANs](http://arxiv.org/abs/2104.12476)                                                                         |    arXiv:2104.12476 [cs, stat]    |                            [EigenGAN](https://github.com/LynnHo/EigenGAN-Tensorflow)                             | 2021 |
| :heart: [Toward Spatially Unbiased Generative Models](http://arxiv.org/abs/2108.01285)                                                                  |               ICCV                |                 [toward_spatial_unbiased](https://github.com/jychoi118/toward_spatial_unbiased)                  | 2021 |
| [Interpreting Generative Adversarial Networks for Interactive Image Generation](http://arxiv.org/abs/2108.04896)                                        |       arXiv:2108.04896 [cs]       |                                                                                                                  | 2021 |
| [Explaining in Style: Training a GAN to Explain a Classifier in StyleSpace](http://arxiv.org/abs/2104.13369)                                            |               ICCV                |                       [explaining-in-style](https://github.com/google/explaining-in-style)                       | 2021 |
| :white_check_mark: [Projected GANs Converge Faster]()                                                                                                   |              NeurIPS              |                        [projected_gan](https://github.com/autonomousvision/projected_gan)                        | 2021 |
| :white_check_mark: [Towards Faster and Stabilized GAN Training for High-Fidelity Few-Shot Image Synthesis](https://openreview.net/forum?id=1Fqg133qRaI) |             ICLR2021              |                             [github](https://github.com/lucidrains/lightweight-gan)                              | 2021 |
| :heart: [Ensembling Off-the-Shelf Models for GAN Training](http://arxiv.org/abs/2112.09130)                                                             |       arXiv:2112.09130 [cs]       |                        [vision-aided-gan](https://github.com/nupurkmr9/vision-aided-gan)                         | 2021 |
| :heart: [StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](http://arxiv.org/abs/2202.00273)                                                      |       arXiv:2202.00273 [cs]       |                                                                                                                  | 2022 |
| [When, Why, and Which Pretrained GANs Are Useful?](http://arxiv.org/abs/2202.08937)                                                                     |               ICLR                |                                                                                                                  | 2022 |
| [A U-Net Based Discriminator for Generative Adversarial Networks](http://arxiv.org/abs/2002.12655)                                                      |               CVPR                |                                                                                                                  | 2020 |

### Transformer

| Title                                                                                                    |         Venue         |                         Code                         | Year |
| :------------------------------------------------------------------------------------------------------- | :-------------------: | :--------------------------------------------------: | :--: |
| [Compositional Transformers for Scene Generation](http://arxiv.org/abs/2111.08960)                       |        NeurIPS        |                                                      | 2021 |
| :heart: [GAN-Supervised Dense Visual Alignment](http://arxiv.org/abs/2112.05143)                         | arXiv:2112.05143 [cs] | [gangealing](https://github.com/wpeebles/gangealing) | 2021 |
| [Improved Transformer for High-Resolution GANs](http://arxiv.org/abs/2106.07631)                         | arXiv:2106.07631 [cs] |                                                      | 2021 |
| [MaskGIT: Masked Generative Image Transformer](http://arxiv.org/abs/2202.04200)                          | arXiv:2202.04200 [cs] |                                                      | 2022 |
| [StyleSwin: Transformer-Based GAN for High-Resolution Image Generation](http://arxiv.org/abs/2112.10762) |         CVPR          |                                                      | 2022 |

### SinGAN

| Title                                                                                                     |         Venue         | Code | Year |
| :-------------------------------------------------------------------------------------------------------- | :-------------------: | :--: | :--: |
| [ExSinGAN: Learning an Explainable Generative Model from a Single Image](http://arxiv.org/abs/2105.07350) | arXiv:2105.07350 [cs] |      | 2021 |

### Video

| Title                                                                                           |         Venue         | Code | Year |
| :---------------------------------------------------------------------------------------------- | :-------------------: | :--: | :--: |
| :heart: [Diverse Generation from a Single Video Made Possible](http://arxiv.org/abs/2109.08591) | arXiv:2109.08591 [cs] |      | 2021 |

### GANs

| Title                                                                                                                                   |            Venue            |                         Code                         | Year |
| :-------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------: | :--------------------------------------------------: | :--: |
| :white_check_mark: [Differentiable Augmentation for Data-Efficient GAN Training](http://arxiv.org/abs/2006.10738)                       |    arXiv:2006.10738 [cs]    |                                                      | 2020 |
| [Sampling Generative Networks](http://arxiv.org/abs/1609.04468)                                                                         | arXiv:1609.04468 [cs, stat] |                                                      | 2016 |
| [Combining Transformer Generators with Convolutional Discriminators](http://arxiv.org/abs/2105.10189)                                   |    arXiv:2105.10189 [cs]    |                                                      | 2021 |
| [Improving Generation and Evaluation of Visual Stories via Semantic Consistency](http://arxiv.org/abs/2105.10026)                       |    arXiv:2105.10026 [cs]    |                                                      | 2021 |
| [TediGAN: Text-Guided Diverse Face Image Generation and Manipulation](http://arxiv.org/abs/2012.03308)                                  |            CVPR             |                                                      | 2021 |
| [Data-Efficient Instance Generation from Instance Discrimination](http://arxiv.org/abs/2106.04566)                                      |    arXiv:2106.04566 [cs]    |                                                      | 2021 |
| [Styleformer: Transformer Based Generative Adversarial Networks with Style Vector](http://arxiv.org/abs/2106.07023)                     | arXiv:2106.07023 [cs, eess] |                                                      | 2021 |
| [FBC-GAN: Diverse and Flexible Image Synthesis via Foreground-Background Composition](http://arxiv.org/abs/2107.03166)                  |    arXiv:2107.03166 [cs]    |                                                      | 2021 |
| [ViTGAN: Training GANs with Vision Transformers](http://arxiv.org/abs/2107.04589)                                                       | arXiv:2107.04589 [cs, eess] |                                                      | 2021 |
| [Learning Efficient GANs for Image Translation via Differentiable Masks and Co-Attention Distillation](http://arxiv.org/abs/2011.08382) |    arXiv:2011.08382 [cs]    |                                                      | 2021 |
| [CGANs with Auxiliary Discriminative Classifier](http://arxiv.org/abs/2107.10060)                                                       |    arXiv:2107.10060 [cs]    |                                                      | 2021 |
| [A Good Image Generator Is What You Need for High-Resolution Video Synthesis](http://arxiv.org/abs/2104.15069)                          |            ICLR             |                                                      | 2021 |
| [Dual Projection Generative Adversarial Networks for Conditional Image Generation](http://arxiv.org/abs/2108.09016)                     |            ICCV             |                                                      | 2021 |
| [Your GAN Is Secretly an Energy-Based Model and You Should Use Discriminator Driven Latent Sampling](http://arxiv.org/abs/2003.06060)   | arXiv:2003.06060 [cs, stat] | [CGAN-DDLS](https://github.com/JHpark1677/CGAN-DDLS) | 2021 |
| [Manifold-Preserved GANs](http://arxiv.org/abs/2109.08955)                                                                              |    arXiv:2109.08955 [cs]    |                                                      | 2021 |
| [Latent Reweighting, an Almost Free Improvement for GANs](http://arxiv.org/abs/2110.09803)                                              |    arXiv:2110.09803 [cs]    |                                                      | 2021 |
| [STRANSGAN: AN EMPIRICAL STUDY ON TRANS- FORMER IN GANS]()                                                                              |  arXiv:2110.13107 [cs.CV]   |                                                      | 2021 |
| [Self-Supervised GANs with Label Augmentation](http://arxiv.org/abs/2106.08601)                                                         |    arXiv:2106.08601 [cs]    |                                                      | 2021 |
| [Regularizing Generative Adversarial Networks under Limited Data](http://arxiv.org/abs/2104.03310)                                      |            CVPR             |   [github](https://github.com/PeterouZh/lecam-gan)   | 2021 |

### cGANs

| Title                                                                           |            Venue            | Code | Year |
| :------------------------------------------------------------------------------ | :-------------------------: | :--: | :--: |
| [Unbiased Auxiliary Classifier GANs with MINE](http://arxiv.org/abs/2006.07567) |    arXiv:2006.07567 [cs]    |      | 2020 |
| [Twin Auxiliary Classifiers GAN](http://arxiv.org/abs/1907.02690)               | arXiv:1907.02690 [cs, stat] |      | 2019 |

### Finetune

| Title                                                                                                                                                                                                                                    |            Venue            |                                Code                                 | Year |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------: | :-----------------------------------------------------------------: | :--: |
| FreezeG                                                                                                                                                                                                                                  |                             |           [github](https://github.com/bryandlee/FreezeG)            |      |
| :white_check_mark: [Freeze the Discriminator: A Simple Baseline for Fine-Tuning GANs](http://arxiv.org/abs/2002.10964)                                                                                                                   | arXiv:2002.10964 [cs, stat] |           [FreezeD](https://github.com/sangwoomo/FreezeD)           | 2020 |
| [Fine-Tuning StyleGAN2 For Cartoon Face Generation](http://arxiv.org/abs/2106.12445)                                                                                                                                                     | arXiv:2106.12445 [cs, eess] | [Cartoon-StyleGAN](https://github.com/happy-jihye/Cartoon-StyleGAN) | 2021 |
| [Transferring GANs: Generating Images from Limited Data](http://arxiv.org/abs/1805.01677)                                                                                                                                                |            ECCV             |                                                                     | 2018 |
| [Image Generation From Small Datasets via Batch Statistics Adaptation](http://arxiv.org/abs/1904.01774)                                                                                                                                  |            ICCV             |                                                                     | 2019 |
| [MineGAN: Effective Knowledge Transfer From GANs to Target Domains With Few Images](http://openaccess.thecvf.com/content_CVPR_2020/html/Wang_MineGAN_Effective_Knowledge_Transfer_From_GANs_to_Target_Domains_With_CVPR_2020_paper.html) |            CVPR             |                                                                     | 2020 |

### Compression

| Title                                                                                                                                                                                                                             |         Venue         |                Code                 | Year |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------: | :---------------------------------: | :--: |
| [GAN Compression: Efficient Architectures for Interactive Conditional GANs](http://openaccess.thecvf.com/content_CVPR_2020/html/Li_GAN_Compression_Efficient_Architectures_for_Interactive_Conditional_GANs_CVPR_2020_paper.html) |         CVPR          |                                     | 2020 |
| [Online Multi-Granularity Distillation for GAN Compression](http://arxiv.org/abs/2108.06908)                                                                                                                                      |         ICCV          |                                     | 2021 |
| [Revisiting Discriminator in GAN Compression: A Generator-Discriminator Cooperative Compression Scheme](http://arxiv.org/abs/2110.14439)                                                                                          | arXiv:2110.14439 [cs] | [GCC](https://github.com/SJLeo/GCC) | 2021 |

### Detection fake

| Title                                                                                                    |         Venue         | Code | Year |
| :------------------------------------------------------------------------------------------------------- | :-------------------: | :--: | :--: |
| [Robust Attentive Deep Neural Network for Exposing GAN-Generated Faces](http://arxiv.org/abs/2109.02167) | arXiv:2109.02167 [cs] |      | 2021 |

### Segmentation

| Title                                                                                    |         Venue         | Code | Year |
| :--------------------------------------------------------------------------------------- | :-------------------: | :--: | :--: |
| [Labels4Free: Unsupervised Segmentation Using StyleGAN](http://arxiv.org/abs/2103.14968) | arXiv:2103.14968 [cs] |      | 2021 |
| [BigDatasetGAN: Synthesizing ImageNet with Pixel-Wise Annotations](http://arxiv.org/abs/2201.04684) | ArXiv:2201.04684 [Cs] |  | arXiv. 2022 |


### Datasets

| Title                                                                                                                            | Venue  |                                               Code                                               |    Year     |
| :------------------------------------------------------------------------------------------------------------------------------- | :----: | :----------------------------------------------------------------------------------------------: | :---------: |
| [Learning Hybrid Image Templates (HIT) by Information Projection]()                                                              | TPAMI  |           [AnimalFace](https://vcla.stat.ucla.edu/people/zhangzhang-si/HiT/exp5.html)            |    2012     |
| [A Style-Based Generator Architecture for Generative Adversarial Networks](http://arxiv.org/abs/1812.04948)                      |  CVPR  |                                             [FFHQ]()                                             |    2019     |
| [StarGAN v2: Diverse Image Synthesis for Multiple Domains](http://arxiv.org/abs/1912.01865)                                      |  CVPR  | [AFHQ](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq) |    2020     |
| [Automated Flower Classification over a Large Number of Classes]()                                                               |        |            [102Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)            |    2008     |
| [XGAN: Unsupervised Image-to-Image Translation for Many-to-Many Mappings](http://arxiv.org/abs/1711.05139)                       |  ICML  |                        [CartoonSet](https://google.github.io/cartoonset/)                        |    2018     |
| [Anime Faces Sourced from Safebooru Resized to 256x256](https://www.kaggle.com/scribbless/another-anime-face-dataset)            | Kaggle |        [AnimeFace](https://www.kaggle.com/scribbless/another-anime-face-dataset/metadata)        |             |
| [Facial Expressions of Manga (Japanese Comic) Character Faces](https://www.kaggle.com/mertkkl/manga-facial-expressions)          | Kaggle |           [MangaExpressions](https://www.kaggle.com/mertkkl/manga-facial-expressions)            |             |
| [Open-Source Cartoon Dataset](https://www.kaggle.com/arnaud58/photo2cartoon/version/1?select=trainB)                             | Kaggle |      [photo2cartoon](https://www.kaggle.com/arnaud58/photo2cartoon/version/1?select=trainB)      |
| [Simpsons Faces: A Lot of Images of Your Favourite Characters](https://www.kaggle.com/kostastokis/simpsons-faces?select=cropped) | Kaggle |        [SimpsonsFaces](https://www.kaggle.com/kostastokis/simpsons-faces?select=cropped)         |             |
| [Bitmoji Faces](https://www.kaggle.com/mostafamozafari/bitmoji-faces)                                                            | Kaggle |               [BitmojiFaces](https://www.kaggle.com/mostafamozafari/bitmoji-faces)               |             |
| [BlendGAN: Implicitly GAN Blending for Arbitrary Stylized Face Generation](https://arxiv.org/abs/2110.11728v1)                   |        |                        [AAHQ](https://github.com/onion-liu/aahq-dataset)                         |    2021     |
| :heart: [Fake It Till You Make It: Face Analysis in the Wild Using Synthetic Data Alone](http://arxiv.org/abs/2109.15102)        |  ICCV  |                  [FaceSynthetics](https://github.com/microsoft/FaceSynthetics)                   |    2021     |
| [Seeing 3D Chairs: Exemplar Part-Based 2D-3D Alignment Using a Large Dataset of CAD Models]()                                    |  CVPR  |                                              chair                                               |    2014     |
| [A Large-Scale Car Dataset for Fine-Grained Categorization and Verification](http://arxiv.org/abs/1506.08959)                    |  CVPR  |                                            [CompCars]                                            | arXiv. 2015 |
| [The ArtBench Dataset: Benchmarking Generative Models with Artworks](https://github.com/liaopeiyuan/artbench) | 2022 |


### alias (ref)

| Title                                                                                                       |            Venue            | Code | Year |
| :---------------------------------------------------------------------------------------------------------- | :-------------------------: | :--: | :--: |
| [Alias-Free Generative Adversarial Networks](http://arxiv.org/abs/2106.12423)                               | arXiv:2106.12423 [cs, stat] |      | 2021 |
| [On Buggy Resizing Libraries and Surprising Subtleties in FID Calculation](http://arxiv.org/abs/2104.11222) |    arXiv:2104.11222 [cs]    |      | 2021 |


## GAN application

| Title                                                                                                                              |         Venue         | Code | Year |
| :--------------------------------------------------------------------------------------------------------------------------------- | :-------------------: | :--: | :--: |
| [SC-FEGAN: Face Editing Generative Adversarial Network with User’s Sketch and Color](http://arxiv.org/abs/1902.06838)              | arXiv:1902.06838 [cs] |      | 2019 |
| [Semantic Text-to-Face GAN -ST^2FG](http://arxiv.org/abs/2107.10756)                                                               | arXiv:2107.10756 [cs] |      | 2021 |
| [CRD-CGAN: Category-Consistent and Relativistic Constraints for Diverse Text-to-Image Generation](http://arxiv.org/abs/2107.13516) | arXiv:2107.13516 [cs] |      | 2021 |

## Image-to-Image Translation

| Title                                                                                                                                     |            Venue            |                                               Code                                               | Year |
| :---------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------: | :----------------------------------------------------------------------------------------------: | :--: |
| [Image-to-Image Translation with Conditional Adversarial Networks](http://arxiv.org/abs/1611.07004)                                       |            CVPR             |                                           [pix2pix]()                                            | 2017 |
| [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](http://arxiv.org/abs/1711.11585)                        |            CVPR             |                                          [pix2pix-HD]()                                          | 2018 |
| [Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks](http://arxiv.org/abs/1703.10593)                        |            ICCV             |                                           [CycleGAN]()                                           | 2017 |
| [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](http://arxiv.org/abs/1711.09020)           |            CVPR             |                                                                                                  | 2018 |
| [StarGAN v2: Diverse Image Synthesis for Multiple Domains](http://arxiv.org/abs/1912.01865)                                               |            CVPR             |                                                                                                  | 2020 |
| [Multimodal Unsupervised Image-to-Image Translation](http://arxiv.org/abs/1804.04732)                                                     | arXiv:1804.04732 [cs, stat] |                                            [MUNIT]()                                             | 2018 |
| [High-Resolution Photorealistic Image Translation in Real-Time: A Laplacian Pyramid Translation Network](http://arxiv.org/abs/2105.09188) |    arXiv:2105.09188 [cs]    |                                                                                                  | 2021 |
| [MixerGAN: An MLP-Based Architecture for Unpaired Image-to-Image Translation](http://arxiv.org/abs/2105.14110)                            |    arXiv:2105.14110 [cs]    |                                                                                                  | 2021 |
| [GANs N’ Roses: Stable, Controllable, Diverse Image to Image Translation (Works for Videos Too!)](http://arxiv.org/abs/2106.06561)        |    arXiv:2106.06561 [cs]    |                                                                                                  | 2021 |
| :heart: [Sketch Your Own GAN](http://arxiv.org/abs/2108.02774)                                                                            |            ICCV             |                                                                                                  | 2021 |
| [Contrastive Learning for Unpaired Image-to-Image Translation](http://arxiv.org/abs/2007.15651)                                           |            ECCV             | [contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation) | 2020 |
| [The Animation Transformer: Visual Correspondence via Segment Matching](http://arxiv.org/abs/2109.02614)                                  |    arXiv:2109.02614 [cs]    |                                                                                                  | 2021 |
| [Image Synthesis via Semantic Composition](http://arxiv.org/abs/2109.07053)                                                               |            ICCV             |                                                                                                  | 2021 |
| [You Only Need Adversarial Supervision for Semantic Image Synthesis](http://arxiv.org/abs/2012.04781)                                     | arXiv:2012.04781 [cs, eess] |                                                                                                  | 2020 |

## Style transfer

| Title                                                                                                                                                                       |               Venue                |                                          Code                                           | Year |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------: | :-------------------------------------------------------------------------------------: | :--: |
| [Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization](http://arxiv.org/abs/1703.06868)                                                               |                ICCV                |                                                                                         | 2017 |
| [Texture Synthesis Using Convolutional Neural Networks](http://arxiv.org/abs/1505.07376)                                                                                    |              NeurIPS               |                                                                                         | 2015 |
| [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)                                                                                                     |    arXiv:1508.06576 [cs, q-bio]    |                                                                                         | 2015 |
| [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html) |                CVPR                |                                                                                         | 2016 |
| [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://arxiv.org/abs/1603.08155)                                                                      |                ECCV                |                                                                                         | 2016 |
| [Texture Networks: Feed-Forward Synthesis of Textures and Stylized Images](http://arxiv.org/abs/1603.03417)                                                                 |                ICML                |                                                                                         | 2016 |
| [Attention-Based Stylisation for Exemplar Image Colourisation](http://arxiv.org/abs/2105.01705)                                                                             |    arXiv:2105.01705 [cs, eess]     |                                                                                         | 2021 |
| [StyleBank: An Explicit Representation for Neural Image Style Transfer](https://arxiv.org/abs/1703.09210v2)                                                                 |                                    |                   [Stylebank](https://github.com/jxcodetw/Stylebank)                    | 2017 |
| [Rethinking and Improving the Robustness of Image Style Transfer](http://arxiv.org/abs/2104.05623)                                                                          |    arXiv:2104.05623 [cs, eess]     |                                                                                         | 2021 |
| [Paint Transformer: Feed Forward Neural Painting with Stroke Prediction](http://arxiv.org/abs/2108.03798)                                                                   |                ICCV                |                                                                                         | 2021 |
| :heart: [AdaAttN: Revisit Attention Mechanism in Arbitrary Neural Style Transfer](http://arxiv.org/abs/2108.03647)                                                          |                ICCV                |                                                                                         | 2021 |
| [ZiGAN: Fine-Grained Chinese Calligraphy Font Generation via a Few-Shot Style Transfer Approach](http://arxiv.org/abs/2108.03596)                                           |       arXiv:2108.03596 [cs]        |                                                                                         | 2021 |
| [Domain-Aware Universal Style Transfer](http://arxiv.org/abs/2108.04441)                                                                                                    |                ICCV                |                                                                                         | 2021 |
| [Aesthetics and Neural Network Image Representations](http://arxiv.org/abs/2109.08103)                                                                                      | arXiv:2109.08103 [cs, eess, q-bio] |                                                                                         | 2021 |
| :heart: [Collaborative Distillation for Ultra-Resolution Universal Style Transfer](http://arxiv.org/abs/2003.08436)                                                         |                CVPR                | [collaborative-distillation](https://github.com/mingsun-tse/collaborative-distillation) | 2020 |
| [Adaptive Convolutions for Structure-Aware Style Transfer]()                                                                                                                |                CVPR                |             [ada-conv-pytorch](https://github.com/RElbers/ada-conv-pytorch)             | 2021 |

## Metric & perceptual loss

| Title                                                                                                                |         Venue         |                                        Code                                         | Year |
| :------------------------------------------------------------------------------------------------------------------- | :-------------------: | :---------------------------------------------------------------------------------: | :--: |
| [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](http://arxiv.org/abs/1801.03924)            | arXiv:1801.03924 [cs] |             [lpips-pytorch](https://github.com/S-aiueo32/lpips-pytorch)             | 2018 |
| [Generating Images with Perceptual Similarity Metrics Based on Deep Networks](http://arxiv.org/abs/1602.02644)       |        NeurIPS        |                                Perceptual Similarity                                | 2016 |
| [Generic Perceptual Loss for Modeling Structured Output Dependencies](http://arxiv.org/abs/2103.10571)               |         CVPR          |                                      [random]                                       | 2021 |
| [Inverting Adversarially Robust Networks for Image Synthesis](http://arxiv.org/abs/2106.06927)                       | arXiv:2106.06927 [cs] |                                                                                     | 2021 |
| [Demystifying MMD GANs](http://arxiv.org/abs/1801.01401)                                                             |         ICLR          |                           Kernel Inception Distance (KID)                           | 2018 |
| [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](http://arxiv.org/abs/1706.08500) |        NeurIPS        |                          Fréchet Inception Distance (FID)                           | 2017 |
| [Improved Techniques for Training GANs](http://papers.nips.cc/paper/6125-improved-techniques-for-training-gans.pdf)  |        NeurIPS        |                                Inception Score (IS)                                 | 2016 |
| [High-Fidelity Performance Metrics for Generative Models in PyTorch](https://github.com/toshas/torch-fidelity)       |                       |                                   torch-fidelity                                    | 2020 |
| [Reliable Fidelity and Diversity Metrics for Generative Models](http://arxiv.org/abs/2002.09797)                     |         ICML          | [generative-evaluation-prdc](https://github.com/clovaai/generative-evaluation-prdc) | 2020 |
| [The Contextual Loss for Image Transformation with Non-Aligned Data](http://arxiv.org/abs/1803.02077) | ECCV | [contextualLoss](https://github.com/roimehrez/contextualLoss) | arXiv. 2018 |
| [Maintaining Natural Image Statistics with the Contextual Loss](http://arxiv.org/abs/1803.04626) | ArXiv:1803.04626 [Cs] |  | 2018 |


## Spectrum

| Title                                                                                                                 |         Venue         | Code | Year |
| :-------------------------------------------------------------------------------------------------------------------- | :-------------------: | :--: | :--: |
| [Reproducibility of "FDA: Fourier Domain Adaptation ForSemantic Segmentation](http://arxiv.org/abs/2104.14749)        | arXiv:2104.14749 [cs] |      | 2021 |
| [A Closer Look at Fourier Spectrum Discrepancies for CNN-Generated Images Detection](http://arxiv.org/abs/2103.17195) |         CVPR          |      | 2021 |

## Weakly Supervised Object Localization

| Title                                                                                                                     |         Venue         | Code | Year |
| :------------------------------------------------------------------------------------------------------------------------ | :-------------------: | :--: | :--: |
| [TS-CAM: Token Semantic Coupled Attention Map for Weakly Supervised Object Localization](http://arxiv.org/abs/2103.14862) | arXiv:2103.14862 [cs] |      | 2021 |
| [Finding an Unsupervised Image Segmenter in Each of Your Deep Generative Models](http://arxiv.org/abs/2105.08127)         | arXiv:2105.08127 [cs] |      | 2021 |
| [Segmentation in Style: Unsupervised Semantic Image Segmentation with Stylegan and CLIP](http://arxiv.org/abs/2107.12518) | arXiv:2107.12518 [cs] |      | 2021 |

## Implicit Neural Representations

- [https://github.com/vsitzmann/awesome-implicit-representations](https://github.com/vsitzmann/awesome-implicit-representations)

| Title                                                                                                               |         Venue         | Code | Year |
| :------------------------------------------------------------------------------------------------------------------ | :-------------------: | :--: | :--: |
| [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](http://arxiv.org/abs/1901.05103)  | arXiv:1901.05103 [cs] |      | 2019 |
| [Occupancy Networks: Learning 3D Reconstruction in Function Space](http://arxiv.org/abs/1812.03828)                 | arXiv:1812.03828 [cs] |      | 2019 |
| :heart: [Neural Image Representations for Multi-Image Fusion and Layer Separation](http://arxiv.org/abs/2108.01199) | arXiv:2108.01199 [cs] |      | 2021 |
| [Learning Continuous Image Representation with Local Implicit Image Function](http://arxiv.org/abs/2012.09161)      |         CVPR          |      | 2021 |


## DDPM

| Title                                                                                                         |            Venue            |                                   Code                                    | Year |
| :------------------------------------------------------------------------------------------------------------ | :-------------------------: | :-----------------------------------------------------------------------: | :--: |
| [Denoising Diffusion Probabilistic Models](http://arxiv.org/abs/2006.11239)                                   | arXiv:2006.11239 [cs, stat] |                 [diffusion](https://github.com/hojonathanho/diffusion)                                                          | 2020 |
| [ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models](http://arxiv.org/abs/2108.02938)     |            ICCV             |                                                                           | 2021 |
| [Diffusion Models Beat GANs on Image Synthesis](http://arxiv.org/abs/2105.05233)                              | arXiv:2105.05233 [cs, stat] |      [guided-diffusion](https://github.com/openai/guided-diffusion)       | 2021 |
| [SDEdit: Image Synthesis and Editing with Stochastic Differential Equations](http://arxiv.org/abs/2108.01073) |    arXiv:2108.01073 [cs]    |              [SDEdit](https://github.com/ermongroup/SDEdit)               | 2021 |
| [D2C: Diffusion-Denoising Models for Few-Shot Conditional Generation](http://arxiv.org/abs/2106.06819)        |    arXiv:2106.06819 [cs]    |                                                                           | 2021 |
| [Label-Efficient Semantic Segmentation with Diffusion Models](https://arxiv.org/abs/2112.03126v1)             |                             | [ddpm-segmentation](https://github.com/yandex-research/ddpm-segmentation) | 2021 |

## Text-to-image
| Title                                                                                                                  |            Venue             |                            Code                            | Year |
| :--------------------------------------------------------------------------------------------------------------------- | :--------------------------: | :--------------------------------------------------------: | :--: |
| [Hierarchical Text-Conditional Image Generation with CLIP Latents](http://arxiv.org/abs/2204.06125) | | [DALLE2-pytorch](https://github.com/lucidrains/DALLE2-pytorch) | 2022 |
| [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487v1) | | [imagen-pytorch](https://github.com/lucidrains/imagen-pytorch),  [Imagen-pytorch](https://github.com/cene555/Imagen-pytorch) | 2022 |
| [Scaling Autoregressive Models for Content-Rich Text-to-Image Generation] n.d. | | [https://github.com/google-research/parti](parti) | |


## 3D & NeRF

- https://www.meshlab.net/

| Title                                                                                                                  |            Venue             |                            Code                            | Year |
| :--------------------------------------------------------------------------------------------------------------------- | :--------------------------: | :--------------------------------------------------------: | :--: |
| Efficient Ray Tracing of Volume Data                                                                                   | ACM Transactions on Graphics |                                                            | 1990 |
| [Surface Light Fields for 3D Photography](https://doi.org/10.1145/344779.344925)                                       |           SIGGRAPH           |                                                            | 2000 |
| [NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections](http://arxiv.org/abs/2008.02268)        |    arXiv:2008.02268 [cs]     |  [nerfw](https://github.com/PeterouZh/nerf_pl/tree/nerfw)  | 2021 |
| [Modulated Periodic Activations for Generalizable Local Functional Representations](http://arxiv.org/abs/2104.03960)   |    arXiv:2104.03960 [cs]     |                                                            | 2021 |
| [Neural Volume Rendering: NeRF And Beyond](http://arxiv.org/abs/2101.05204)                                            |    arXiv:2101.05204 [cs]     | [awesome-NeRF](https://github.com/yenchenlin/awesome-NeRF) | 2021 |
| [Editing Conditional Radiance Fields](http://arxiv.org/abs/2105.06466)                                                 |    arXiv:2105.06466 [cs]     |      [editnerf](https://github.com/stevliu/editnerf)       | 2021 |
| [Recursive-NeRF: An Efficient and Dynamically Growing NeRF](http://arxiv.org/abs/2105.09103)                           |    arXiv:2105.09103 [cs]     |                                                            | 2021 |
| [MVSNeRF: Fast Generalizable Radiance Field Reconstruction from Multi-View Stereo](http://arxiv.org/abs/2103.15595)    |    arXiv:2103.15595 [cs]     |      [mvsnerf](https://github.com/apchenstu/mvsnerf)       | 2021 |
| [Depth-Supervised NeRF: Fewer Views and Faster Training for Free](http://arxiv.org/abs/2107.02791)                     |    arXiv:2107.02791 [cs]     |                                                            | 2021 |
| [Rethinking Positional Encoding](http://arxiv.org/abs/2107.02561)                                                      |    arXiv:2107.02561 [cs]     |                                                            | 2021 |
| [Nerfies: Deformable Neural Radiance Fields](https://arxiv.org/abs/2011.12948v4)                                       |       arXiv:2011.12948       |        [nerfies](https://github.com/google/nerfies)        | 2020 |
| [Self-Calibrating Neural Radiance Fields](http://arxiv.org/abs/2108.13826)                                             |             ICCV             |                                                            | 2021 |
| [Light Field Networks: Neural Scene Representations with Single-Evaluation Rendering](http://arxiv.org/abs/2106.02634) |    arXiv:2106.02634 [cs]     |                                                            | 2021 |

### Sine

| Title                                                                                                                                   |         Venue         |                           Code                           | Year | Cite |
| :-------------------------------------------------------------------------------------------------------------------------------------- | :-------------------: | :------------------------------------------------------: | :--: | :--: |
| [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](http://arxiv.org/abs/2006.10739)              | arXiv:2006.10739 [cs] |                                                          | 2020 |
| :white_check_mark: [Implicit Neural Representations with Periodic Activation Functions](http://arxiv.org/abs/2006.09661)                |        NeurIPS        |                                                          | 2020 |
| :white_check_mark: [Modulated Periodic Activations for Generalizable Local Functional Representations](http://arxiv.org/abs/2104.03960) | arXiv:2104.03960 [cs] |                                                          | 2021 |
| [Learned Initializations for Optimizing Coordinate-Based Neural Representations](http://arxiv.org/abs/2012.02189)                       | arXiv:2012.02189 [cs] | [nerf-meta](https://github.com/sanowar-raihan/nerf-meta) | 2021 |
| [Seeing Implicit Neural Representations as Fourier Series](http://arxiv.org/abs/2109.00249)                                             | arXiv:2109.00249 [cs] |                                                          | 2021 |

### INR

| Title                                                                                                     |         Venue         |                       Code                       | Year | Cite |
| :-------------------------------------------------------------------------------------------------------- | :-------------------: | :----------------------------------------------: | :--: | :--: |
| [Adversarial Generation of Continuous Images](http://arxiv.org/abs/2011.12026)                            | arXiv:2011.12026 [cs] | [inr-gan](https://github.com/universome/inr-gan) | 2020 |
| [Image Generators with Conditionally-Independent Pixel Synthesis](http://arxiv.org/abs/2011.13775)        | arXiv:2011.13775 [cs] |    [CIPS](https://github.com/saic-mdal/CIPS)     | 2020 |
| [A Structured Dictionary Perspective on Implicit Neural Representations](http://arxiv.org/abs/2112.01917) | arXiv:2112.01917 [cs] |                                                  | 2021 |

### 3D & NeRF GANs

| Title                                                                                                                                             |            Venue            |                          Code                          | Year | Cite |
| :------------------------------------------------------------------------------------------------------------------------------------------------ | :-------------------------: | :----------------------------------------------------: | :--: | :--: |
| :heavy_check_mark: [HoloGAN: Unsupervised Learning of 3D Representations from Natural Images](http://arxiv.org/abs/1904.01326)                    |            ICCV             |                                                        | 2019 |
| [BlockGAN: Learning 3D Object-Aware Scene Representations from Unlabelled Images](http://arxiv.org/abs/2002.08988)                                |           NeurIPS           |                                                        | 2020 |
| :heavy_check_mark: [GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis](http://arxiv.org/abs/2007.02442)                                                  |    arXiv:2007.02442 [cs]    |                                                        | 2021 |
| :heavy_check_mark: [Pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis](http://arxiv.org/abs/2012.00926)                         |    arXiv:2012.00926 [cs]    |   [pi-GAN](https://github.com/marcoamonteiro/pi-GAN)   | 2021 |  19  |
| :heavy_check_mark: [GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields](http://arxiv.org/abs/2011.12100)              |            CVPR             | [giraffe](https://github.com/autonomousvision/giraffe) | 2021 |
| :heavy_check_mark: [GIRAFFE HD: A High-Resolution 3D-Aware Generative Model](http://arxiv.org/abs/2203.14954)                                     |            CVPR             |                                                        | 2022 |
| :heart: [StyleNeRF: A Style-Based 3D-Aware Generator for High-Resolution Image Synthesis](http://arxiv.org/abs/2110.08985)                        | arXiv:2110.08985 [cs, stat] |                                                        | 2021 |
| [CAMPARI: Camera-Aware Decomposed Generative Neural Radiance Fields](http://arxiv.org/abs/2103.17269)                                             |    arXiv:2103.17269 [cs]    |                                                        | 2021 |
| :heavy_check_mark: [GNeRF: GAN-Based Neural Radiance Field without Posed Camera](http://arxiv.org/abs/2103.15606)                                                    |    arXiv:2103.15606 [cs]    |         [gnerf](https://github.com/MQ66/gnerf)         | 2021 |
| :heart: [Unconstrained Scene Generation with Locally Conditioned Radiance Fields](http://arxiv.org/abs/2104.00670)                                |            ICCV             |       [ml-gsn](https://github.com/apple/ml-gsn)        | 2021 |
| [Learning Object-Compositional Neural Radiance Field for Editable Scene Rendering](http://arxiv.org/abs/2109.01847)                               |            ICCV             |                                                        | 2021 |
| :heavy_check_mark: [A Shading-Guided Generative Implicit Model for Shape-Accurate 3D-Aware Image Synthesis](http://arxiv.org/abs/2110.15678)                         |           NeurIPS           |                                                        | 2021 |
| :heavy_check_mark: [Generative Occupancy Fields for 3D Surface-Aware Image Synthesis](http://arxiv.org/abs/2111.00969)                                               |           NeurIPS           |                                                        | 2021 |
| :heavy_check_mark: [Efficient Geometry-Aware 3D Generative Adversarial Networks](http://arxiv.org/abs/2112.07945)                                 |    arXiv:2112.07945 [cs]    |         [eg3d](https://github.com/NVlabs/eg3d)         | 2021 |
| :heavy_check_mark: [3D-Aware Image Synthesis via Learning Structural and Textural Representations](http://arxiv.org/abs/2112.10759)               |    arXiv:2112.10759 [cs]    |                     [VolumeGAN]()                      | 2021 |
| :heavy_check_mark: [GRAM: Generative Radiance Manifolds for 3D-Aware Image Generation](http://arxiv.org/abs/2112.08867)                           |    arXiv:2112.08867 [cs]    |                                                        | 2021 |
| [CoordGAN: Self-Supervised Dense Correspondences Emerge from GANs](http://arxiv.org/abs/2203.16521)                                               |            CVPR             |                                                        | 2022 |
| :heavy_check_mark: [Disentangled3D: Learning a 3D Generative Model with Disentangled Geometry and Appearance from Monocular Images](http://arxiv.org/abs/2203.15926) |            CVPR             |                                                        | 2022 |
| :heavy_check_mark: [Multi-View Consistent Generative Adversarial Networks for 3D-Aware Image Synthesis](http://arxiv.org/abs/2204.06307)          |            CVPR             |   [MVCGAN](https://github.com/Xuanmeng-Zhang/MVCGAN)   | 2022 |
| :heavy_check_mark: [FENeRF: Face Editing in Neural Radiance Fields]()                                                                             |            CVPR             |    [FENeRF](https://github.com/MrTornado24/FENeRF)     | 2022 |
| :heavy_check_mark: [IDE-3D: Interactive Disentangled Editing for High-Resolution 3D-Aware Portrait Synthesis](http://arxiv.org/abs/2205.15517) | arXiv:2205.15517 |  | 2022 |
| :heavy_check_mark: [EpiGRAF: Rethinking Training of 3D GANs](http://arxiv.org/abs/2206.10535) | ArXiv:2206.10535 [Cs] | [epigraf](https://github.com/universome/epigraf)  | arXiv. 2022 |
| https://github.com/rethinking-3d-gans/code | | | |

### NeRF

- https://github.com/ActiveVisionLab/nerfmm
- https://github.com/ventusff/improved-nerfmm
- https://github.com/Kai-46/nerfplusplus
- https://github.com/kwea123/nerf_pl
- https://github.com/NVlabs/instant-ngp

| Title                                                                                                                                  |          Venue           |                                                     Code                                                      | Year | Cite |
| :------------------------------------------------------------------------------------------------------------------------------------- | :----------------------: | :-----------------------------------------------------------------------------------------------------------: | :--: | :--: |
| Ray Tracing Volume Densities                                                                                                           |         SIGGRAPH         |                                                                                                               | 1984 |
| :white_check_mark: [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://arxiv.org/abs/2003.08934)           |           ECCV           |                          [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)                           | 2020 |
| :white_check_mark: [NeRF--: Neural Radiance Fields Without Known Camera Parameters](http://arxiv.org/abs/2102.07064)                   |  arXiv:2102.07064 [cs]   | [nerfmm](https://github.com/PeterouZh/nerfmm), [improved-nerfmm](https://github.com/ventusff/improved-nerfmm) | 2021 |
| :white_check_mark: [NeRF++: Analyzing and Improving Neural Radiance Fields](http://arxiv.org/abs/2010.07492)                           |  arXiv:2010.07492 [cs]   |                            [nerfplusplus](https://github.com/Kai-46/nerfplusplus)                             | 2020 |
| :white_check_mark: [FastNeRF: High-Fidelity Neural Rendering at 200FPS](http://arxiv.org/abs/2103.10380)                               |  arXiv:2103.10380 [cs]   |                                                                                                               | 2021 |
| :white_check_mark: [KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs](http://arxiv.org/abs/2103.13744)         |           ICCV           |                                                                                                               | 2021 |
| [Plenoxels: Radiance Fields without Neural Networks](http://arxiv.org/abs/2112.05131)                                                  |  arXiv:2112.05131 [cs]   |                                    [svox2](https://github.com/sxyu/svox2)                                     | 2021 |
| [Mega-NeRF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs](http://arxiv.org/abs/2112.10703)                      |  arXiv:2112.10703 [cs]   |                             [mega-nerf](https://github.com/cmusatyalab/mega-nerf)                             | 2021 |
| :heart: [Neural Sparse Voxel Fields](http://arxiv.org/abs/2007.11571)                                                                  |  arXiv:2007.11571 [cs]   |                               [NSVF](https://github.com/facebookresearch/NSVF)                                | 2021 |
| [Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields](http://arxiv.org/abs/2103.13415)                      |           ICCV           |                                                                                                               | 2021 |
| :heart: [Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields](https://arxiv.org/abs/2111.12077v2)                              | arXiv:2111.12077 [cs.CV] |                                                                                                               | 2021 |
| :heart: [IBRNet: Learning Multi-View Image-Based Rendering](http://arxiv.org/abs/2102.13090)                                           |  arXiv:2102.13090 [cs]   |                                                                                                               | 2021 |
| :heart: [Neural Actor: Neural Free-View Synthesis of Human Actors with Pose Control](http://arxiv.org/abs/2106.02019)                  |  arXiv:2106.02019 [cs]   |                                                                                                               | 2022 |
| [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding]()                                                            |                          |                             [instant-ngp](https://github.com/NVlabs/instant-ngp)                              |      |      |
| :heart: [Point-NeRF: Point-Based Neural Radiance Fields](http://arxiv.org/abs/2201.08845)                                                      |  arXiv:2201.08845 [cs]   |                               [pointnerf](https://github.com/Xharlie/pointnerf)                               | 2022 |
| [MoFaNeRF: Morphable Facial Neural Radiance Field](http://arxiv.org/abs/2112.02308)                                                    |  arXiv:2112.02308 [cs]   |                                                                                                               | 2021 |
| [Object-Centric Neural Scene Rendering](https://arxiv.org/abs/2012.08503v1)                                                            |           2020           |
| [Semantic View Synthesis](https://arxiv.org/abs/2008.10598v1)                                                                          |           2020           |
| [NeRS: Neural Reflectance Surfaces for Sparse-View 3D Reconstruction in the Wild](https://arxiv.org/abs/2110.07604v3)                  |           2021           |
| [MINE: Towards Continuous Depth MPI with NeRF for Novel View Synthesis](http://arxiv.org/abs/2103.14910)                               |  arXiv:2103.14910 [cs]   |                                                                                                               | 2021 |
| :white_check_mark: [CodeNeRF: Disentangled Neural Radiance Fields for Object Categories](http://arxiv.org/abs/2109.01750)              |           ICCV           |                               [code-nerf](https://github.com/wbjang/code-nerf)                                | 2021 |
| [NeRF-SR: High-Quality Neural Radiance Fields Using Super-Sampling](http://arxiv.org/abs/2112.01759)                                   |  arXiv:2112.01759 [cs]   |                                                                                                               | 2021 |
| :heart: [TensoRF: Tensorial Radiance Fields](http://arxiv.org/abs/2203.09517)                                                          |  arXiv:2203.09517 [cs]   |                                                                                                               | 2022 |
| [Sem2NeRF: Converting Single-View Semantic Masks to Neural Radiance Fields](http://arxiv.org/abs/2203.10821)                           |  arXiv:2203.10821 [cs]   |                                                                                                               | 2022 |
| [Pix2NeRF: Unsupervised Conditional $\pi$-GAN for Single Image to Neural Radiance Fields Translation](http://arxiv.org/abs/2202.13162) |  arXiv:2202.13162 [cs]   |                                                                                                               | 2022 |
| [CLIP-NeRF: Text-and-Image Driven Manipulation of Neural Radiance Fields](http://arxiv.org/abs/2112.05139)                             |  arXiv:2112.05139 [cs]   |                                                                                                               | 2022 |
| [BARF: Bundle-Adjusting Neural Radiance Fields](http://arxiv.org/abs/2104.06405)                                                       |  arXiv:2104.06405 [cs]   |                                                                                                               | 2021 |
| [Unified Implicit Neural Stylization](http://arxiv.org/abs/2204.01943)                                                                 |  arXiv:2204.01943 [cs]   |                                                                                                               | 2022 |
| [SinNeRF: Training Neural Radiance Fields on Complex Scenes from a Single Image](http://arxiv.org/abs/2204.00928)                      |  arXiv:2204.00928 [cs]   |                                                                                                               | 2022 |


### 3D inversion

| Title                                                                                                                                      |         Venue         |                                       Code                                        | Year | Cite |
| :----------------------------------------------------------------------------------------------------------------------------------------- | :-------------------: | :-------------------------------------------------------------------------------: | :--: | :--: |
| [Unsupervised 3D Shape Completion through GAN Inversion](http://arxiv.org/abs/2104.13366) | CVPR |  | 2021 |
| [3D GAN Inversion for Controllable Portrait Image Animation](http://arxiv.org/abs/2203.13441) | ArXiv:2203.13441 [Cs] |  | arXiv. 2022 |
| [Pix2NeRF: Unsupervised Conditional $\pi$-GAN for Single Image to Neural Radiance Fields Translation](http://arxiv.org/abs/2202.13162) | ArXiv:2202.13162 [Cs] |  | arXiv. 2022 |




### Dynamic

| Title                                                                                                                                      |         Venue         |                                       Code                                        | Year | Cite |
| :----------------------------------------------------------------------------------------------------------------------------------------- | :-------------------: | :-------------------------------------------------------------------------------: | :--: | :--: |
| [Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes](http://arxiv.org/abs/2011.13084)                                | arXiv:2011.13084 [cs] | [Neural-Scene-Flow-Fields](https://github.com/zl548/Neural-Scene-Flow-Fields.git) | 2021 |
| [D-NeRF: Neural Radiance Fields for Dynamic Scenes](http://arxiv.org/abs/2011.13961)                                                       | arXiv:2011.13961 [cs] |                [D-NeRF](https://github.com/albertpumarola/D-NeRF)                 | 2020 |
| [Dynamic View Synthesis from Dynamic Monocular Video](http://arxiv.org/abs/2105.06468)                                                     | arXiv:2105.06468 [cs] |             [DynamicNeRF](https://github.com/gaochen315/DynamicNeRF)              | 2021 |
| :heart: [HyperNeRF: A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields](http://arxiv.org/abs/2106.13228) | arXiv:2106.13228 [cs] |                 [hypernerf](https://github.com/google/hypernerf)                  | 2021 |
| [Neural Radiance Flow for 4D View Synthesis and Video Processing](https://arxiv.org/abs/2012.09790v2)                                      |         2020          |
| :heart: [Animatable Neural Implicit Surfaces for Creating Avatars from Videos](http://arxiv.org/abs/2203.08133)                            | arXiv:2203.08133 [cs] |                                                                                   | 2022 |

### Body

- https://github.com/3DFaceBody/awesome-3dbody-papers
- https://github.com/openMVG/awesome_3DReconstruction_list
- https://github.com/ytrock/THuman2.0-Dataset

| Title                                                                                                                    |                   Venue                   |                             Code                             | Year |
| :----------------------------------------------------------------------------------------------------------------------- | :---------------------------------------: | :----------------------------------------------------------: | :--: |
| [SMPL: A Skinned Multi-Person Linear Model]()                                                                            | ACM Trans. Graphics (Proc. SIGGRAPH Asia) |                                                              | 2015 |
| [Expressive Body Capture: 3D Hands, Face, and Body from a Single Image]()                                                |                   CVPR                    |                           [SMPL-X]                           | 2019 |
| [AMASS: Archive of Motion Capture as Surface Shapes]()                                                                   |                   ICCV                    |       [AMASS](https://amass.is.tue.mpg.de/index.html)        | 2019 |
| :heavy_check_mark: [SNARF: Differentiable Forward Skinning for Animating Non-Rigid Neural Implicit Shapes](http://arxiv.org/abs/2104.03953) |                   ICCV                    |                                                              | 2021 |
| :heavy_check_mark: [Animatable Neural Radiance Fields for Modeling Dynamic Human Bodies](http://arxiv.org/abs/2105.02872) | ICCV | [animatable_nerf](https://github.com/zju3dv/animatable_nerf) | 2021 |
| [Neural Actor: Neural Free-View Synthesis of Human Actors with Pose Control](http://arxiv.org/abs/2106.02019) | SIGGRAPH Asia |  | 2021 |
| :heavy_check_mark: [Animatable Neural Radiance Fields from Monocular RGB Videos](http://arxiv.org/abs/2106.13629) | ArXiv:2106.13629 [Cs] | [Anim-NeRF](https://github.com/JanaldoChen/Anim-NeRF) | arXiv. 2021 |
| [VIBE: Video Inference for Human Body Pose and Shape Estimation](http://arxiv.org/abs/1912.05656) | CVPR | [VIBE](https://github.com/mkocabas/VIBE) | arXiv. 2020 |
| :heavy_check_mark: [A-NeRF: Articulated Neural Radiance Fields for Learning Human Shape, Appearance, and Pose](http://arxiv.org/abs/2102.06199) | NeurIPS | [A-NeRF](https://github.com/LemonATsu/A-NeRF)  | arXiv. 2021 |
| [HumanNeRF: Free-Viewpoint Rendering of Moving People from Monocular Video](http://arxiv.org/abs/2201.04127) | CVPR | [humannerf](https://github.com/chungyiweng/humannerf) | 2022 |
| :heart: [The Power of Points for Modeling Humans in Clothing]()                                                          |                   ICCV                    |                                                              | 2021 |
| [StylePeople: A Generative Model of Fullbody Human Avatars](http://arxiv.org/abs/2104.08363)                             |           arXiv:2104.08363 [cs]           |                                                              | 2021 |
| [NPMs: Neural Parametric Models for 3D Deformable Shapes](http://arxiv.org/abs/2104.00702)                               |           arXiv:2104.00702 [cs]           |                                                              | 2021 |
| :heart: [ICON: Implicit Clothed Humans Obtained from Normals](http://arxiv.org/abs/2112.09127)                           |           arXiv:2112.09127 [cs]           |          [ICON](https://github.com/YuliangXiu/ICON)          | 2022 |
| :heart: [GDNA: Towards Generative Detailed Neural Avatars]()                                                             |                   CVPR                    |                                                              | 2022 |
| [SCANimate: Weakly Supervised Learning of Skinned Clothed Avatar Networks]()                                             |                   CVPR                    |                                                              | 2021 |
| [NeuralAnnot: Neural Annotator for 3D Human Mesh Training Sets](http://arxiv.org/abs/2011.11232)                         |           arXiv:2011.11232 [cs]           |                                                              | 2022 |
| :heart: [PyMAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop](http://arxiv.org/abs/2103.16507) | ICCV |  | 2021 |
| :heart: [Structured Local Radiance Fields for Human Avatar Modeling](http://arxiv.org/abs/2203.14478) | CVPR |  | arXiv. 2022 |
| :heart: [KeypointNeRF: Generalizing Image-Based Volumetric Avatars Using Relative Spatial Encoding of Keypoints](http://arxiv.org/abs/2205.04992) | arXiv:2205.04992 [cs] |  | 2022 |



### Body Generation

| Title                                                                                                               |         Venue         |                         Code                         | Year |
| :------------------------------------------------------------------------------------------------------------------ | :-------------------: | :--------------------------------------------------: | :--: |
| [DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations]() |  CVPR | [DeepFashion](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)  | 2016 |
| [Text2Human: Text-Driven Controllable Human Image Generation]() | ACM Transactions on Graphics (TOG) | [Text2Human](https://github.com/yumingj/Text2Human) | 2022 |
| [StyleGAN-Human: A Data-Centric Odyssey of Human Generation](http://arxiv.org/abs/2204.11823)                       | arXiv:2204.11823 [cs] |                                                      | 2022 |
| :heavy_check_mark: [3D-Aware Semantic-Guided Generative Model for Human Synthesis](http://arxiv.org/abs/2112.01422) | arXiv:2112.01422 [cs] |                                                      | 2021 |
| :heart: [InsetGAN for Full-Body Image Generation](http://arxiv.org/abs/2203.07293)                                  | arXiv:2203.07293 [cs] |                                                      | 2022 |
| [AvatarCLIP: Zero-Shot Text-Driven Generation and Animation of 3D Avatars](http://arxiv.org/abs/2205.08535)         |       SIGGRAPH        | [AvatarCLIP](https://github.com/hongfz16/AvatarCLIP) | 2022 |
| [Liquid Warping GAN: A Unified Framework for Human Motion Imitation, Appearance Transfer and Novel View Synthesis](http://arxiv.org/abs/1909.12224) | ICCV | [impersonator](https://github.com/svip-lab/impersonator)  | 2019 |
| :heavy_check_mark: [SMPLpix: Neural Avatars from 3D Human Models](http://arxiv.org/abs/2008.06872) | Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision | [smplpix](https://github.com/sergeyprokudin/smplpix) | arXiv. 2021 |
| :heavy_check_mark: [Unsupervised Learning of Efficient Geometry-Aware Neural Articulated Representations](http://arxiv.org/abs/2204.08839) | arXiv. | | 2022 |


### Body from video

| Title                                                                                                                             |         Venue         | Code | Year |
| :-------------------------------------------------------------------------------------------------------------------------------- | :-------------------: | :--: | :--: |
| [SelfRecon: Self Reconstruction Your Digital Avatar from Monocular Video](http://arxiv.org/abs/2201.12792)                        | arXiv:2201.12792 [cs] |      | 2022 |
| [Generalizable Neural Performer: Learning Robust Radiance Fields for Human Novel View Synthesis](http://arxiv.org/abs/2204.11798) | arXiv:2204.11798 [cs] |      | 2022 |


### 3D FACE Avatars

- https://github.com/TimoBolkart/BFM_to_FLAME

| Title                                                                                                                                     |            Venue             |                                  Code                                  | Year |
| :---------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------: | :--------------------------------------------------------------------: | :--: |
| [A Morphable Model for the Synthesis of 3D Faces](https://doi.org/10.1145/311535.311556) | Proceedings of the 26th Annual Conference on Computer Graphics and Interactive Techniques |  | SIGGRAPH ’99, USA: ACM Press/Addison-Wesley Publishing Co. 1999 |
| [Learning a Model of Facial Shape and Expression from 4D Scans]() | ACM Transactions on Graphics | [FLAME] | 2017 |
| :heart: [FLAME-in-NeRF : Neural Control of Radiance Fields for Free View Face Animation](http://arxiv.org/abs/2108.04913) | arXiv:2108.04913 [cs] |                                                                              | 2021 |
| [Learning a Model of Facial Shape and Expression from 4D Scans]()                                                                         | ACM Transactions on Graphics |                                                                        | 2017 |
| :heart: [EMOCA: Emotion Driven Monocular Face Capture and Animation]()                                                                    |             CVPR             |               [emoca](https://github.com/radekd91/emoca)               | 2022 |
| [FaceVerse: A Fine-Grained and Detail-Controllable 3D Face Morphable Model from a Hybrid Dataset]()                                       |             CVPR             |                                                                        | 2022 |
| [I M Avatar: Implicit Morphable Head Avatars from Videos](http://arxiv.org/abs/2112.07471)                                                |             CVPR             |              [IMavatar](https://github.com/zhengyuf/IMavatar)                                                          | 2022 |
| :heavy_check_mark: [Neural Head Avatars from Monocular RGB Videos](http://arxiv.org/abs/2112.01554)                                       |    arXiv:2112.01554 [cs]     | [neural-head-avatars](https://github.com/philgras/neural-head-avatars) | 2022 |
| [PVA: Pixel-Aligned Volumetric Avatars](http://arxiv.org/abs/2101.02697)                                                                  |    arXiv:2101.02697 [cs]     |                                                                        | 2021 |
| [AD-NeRF: Audio Driven Neural Radiance Fields for Talking Head Synthesis](http://arxiv.org/abs/2103.11078)                                |    arXiv:2103.11078 [cs]     |                                                                        | 2021 |
| [Semantic-Aware Implicit Neural Audio-Driven Video Portrait Generation](http://arxiv.org/abs/2201.07786)                                  | arXiv:2201.07786 [cs, eess]  |                                                                        | 2022 |
| [HeadGAN: One-Shot Neural Head Synthesis and Editing](http://arxiv.org/abs/2012.08261)                                                    |    arXiv:2012.08261 [cs]     |                                                                        | 2021 |
| [KeypointNeRF: Generalizing Image-Based Volumetric Avatars Using Relative Spatial Encoding of Keypoints](http://arxiv.org/abs/2205.04992) |    arXiv:2205.04992 [cs]     |                                                                        | 2022 |
| [Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set](http://arxiv.org/abs/1903.08527) | ArXiv:1903.08527 [Cs] | [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch)  | arXiv. 2020 |


### Face Style

| Title                                                                                                      |         Venue         |                              Code                               | Year |
| :--------------------------------------------------------------------------------------------------------- | :-------------------: | :-------------------------------------------------------------: | ---- |
| [Pastiche Master: Exemplar-Based High-Resolution Portrait Style Transfer](http://arxiv.org/abs/2203.13248) | arXiv:2203.13248 [cs] | [DualStyleGAN](https://github.com/williamyang1991/DualStyleGAN) | 2022 |
| [Stitch It in Time: GAN-Based Facial Editing of Real Videos](http://arxiv.org/abs/2201.08361)              |        arXiv.         |           [STIT](https://github.com/rotemtzaban/STIT)           | 2022 |
| [Fix the Noise: Disentangling Source Feature for Transfer Learning of StyleGAN](http://arxiv.org/abs/2204.14079) | ArXiv:2204.14079 [Cs] | [FixNoise](https://github.com/LeeDongYeun/FixNoise)  | arXiv. 2022 |


### Face Animation

| Title                                                                                 | Venue | Code | Year |
| :------------------------------------------------------------------------------------ | :---: | :--: | ---- |
| [Thin-Plate Spline Motion Model for Image Animation](http://arxiv.org/abs/2203.14367) | CVPR  |      | 2022 |
| [Depth-Aware Generative Adversarial Network for Talking Head Video Generation](http://arxiv.org/abs/2203.06605) | CVPR | [DaGAN](https://github.com/harlanhong/CVPR2022-DaGAN)  | arXiv. 2022 |


### Renderer & Regularization

| Title                                                                                                                        |                       Venue                        |                              Code                               | Year |
| :--------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------: | :-------------------------------------------------------------: | ---- |
| [Implicit Geometric Regularization for Learning Shapes](http://arxiv.org/abs/2002.10099)                                     |                        ICML                        |                            [Eikonal]                            | 2020 |
| [Neural 3D Scene Reconstruction with the Manhattan-World Assumption](http://arxiv.org/abs/2205.02836)                        |                        CVPR                        |    [manhattan_sdf](https://github.com/zju3dv/manhattan_sdf)     | 2022 |
| [Differentiable Signed Distance Function Rendering]()                                                                        | Transactions on Graphics (Proceedings of SIGGRAPH) | [sdf](https://github.com/lucidrains/differentiable-SDF-pytorch) | 2022 |
| [NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-View Reconstruction](http://arxiv.org/abs/2106.10689) |                                                    |            [NeuS](https://github.com/Totoro97/NeuS)             | 2021 |
| :heart: [Volume Rendering of Neural Implicit Surfaces](http://arxiv.org/abs/2106.12052)                                                |  arXiv:2106.12052 [cs]   |                     [volsdf](https://github.com/lioryariv/volsdf)                                                                                          | 2021 |
| [Multiview Neural Surface Reconstruction by Disentangling Geometry and Appearance](https://arxiv.org/abs/2003.09852v3) | NeurIPS | [idr](https://github.com/lioryariv/idr)  | 2020 |
| [UNISURF: Unifying Neural Implicit Surfaces and Radiance Fields for Multi-View Reconstruction]() | ICCV | [unisurf](https://github.com/autonomousvision/unisurf) | 2021 |
| [MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction](https://arxiv.org/abs/2206.00665v1) | ArXiv:2206.00665 |  | 2022 |
| [IRON: Inverse Rendering by Optimizing Neural SDFs and Materials from Photometric Images](http://arxiv.org/abs/2204.02232) | CVPR |  | arXiv. 2022 |
| [Direct Voxel Grid Optimization: Super-Fast Convergence for Radiance Fields Reconstruction](http://arxiv.org/abs/2111.11215) | CVPR | [DirectVoxGO](https://github.com/sunset1995/DirectVoxGO) | arXiv. 2022 |
| [Improved Direct Voxel Grid Optimization for Radiance Fields Reconstruction](http://arxiv.org/abs/2206.05085) | ArXiv:2206.05085 [Cs] |  | arXiv. 2022 |
| [Improved Surface Reconstruction Using High-Frequency Details](http://arxiv.org/abs/2206.07850) | ArXiv:2206.07850 [Cs] |  | arXiv. 2022 |
| [InfoNeRF: Ray Entropy Minimization for Few-Shot Neural Volume Rendering](http://arxiv.org/abs/2112.15399) | CVPR | [InfoNeRF](https://github.com/mjmjeong/InfoNeRF) | arXiv. 2022 |
| [Improving Neural Implicit Surfaces Geometry with Patch Warping](http://arxiv.org/abs/2112.09648) | CVPR | [NeuralWarp](https://github.com/fdarmon/NeuralWarp) | arXiv. 2022 |



### Motion

- https://github.com/xianfei/SysMocap

| Title                                                                                                                        |                       Venue                        |                              Code                               | Year |
| :--------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------: | :-------------------------------------------------------------: | ---- |
| [GANimator: Neural Motion Synthesis from a Single Sequence]()  | ACM Transactions on Graphics (TOG)  | [ganimator](https://github.com/PeizhuoLi/ganimator) | 2022 |
| [Watch It Move: Unsupervised Discovery of 3D Joints for Re-Posing of Articulated Objects] | CVPR |  [watch-it-move](https://github.com/NVlabs/watch-it-move) | 2022 |
| [Learn to Dance with AIST++: Music Conditioned 3D Dance Generation]() | ICCV |  | 2021 |
| [Talking Head(?) Anime from a Single Image 3: Now the Body Too](http://pkhungurn.github.io/talking-head-anime-3/) | | [talking-head-anime](https://github.com/pkhungurn/talking-head-anime-3-demo) | 2022 |
| [PhysCap: Physically Plausible Monocular 3D Motion Capture in Real Time]() | ACM Transactions on Graphics |  | 2020 |



### Shape generation

| Title                                                                                                                        |                       Venue                        |                              Code                               | Year |
| :--------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------: | :-------------------------------------------------------------: | ---- |
| [Learning Implicit Fields for Generative Shape Modeling](http://arxiv.org/abs/1812.02822)                           | arXiv:1812.02822 [cs] |      | 2019 |


### SMPL estimation
| Title                                                                                                                        |                       Venue                        |                              Code                               | Year |
| :--------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------: | :-------------------------------------------------------------: | ---- |
| [VIBE: Video Inference for Human Body Pose and Shape Estimation](http://arxiv.org/abs/1912.05656) | CVPR | [VIBE](https://github.com/mkocabas/VIBE)  | arXiv. 2020 |
| [TransPose: Real-Time 3D Human Translation and Pose Estimation with Six Inertial Sensors]() | ACM Transactions on Graphics |  [TransPose](https://github.com/Xinyu-Yi/TransPose) | 2021 |
| [Monocular Expressive Body Regression through Body-Driven Attention](https://expose.is.tue.mpg.de) | European Conference on Computer Vision (ECCV) | [expose](https://github.com/vchoutas/expose)  | 2020 |
| [Human Mesh Recovery from Multiple Shots](http://arxiv.org/abs/2012.09843) | CVPR | [multishot](https://github.com/geopavlakos/multishot)  | arXiv. 2022 |



### Segmentation
| Title                                                                                                                        |                       Venue                        |                              Code                               | Year |
| :--------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------: | :-------------------------------------------------------------: | ---- |
| [Robust High-Resolution Video Matting with Temporal Guidance](http://arxiv.org/abs/2108.11515) | ArXiv:2108.11515 [Cs] | [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) | arXiv. 2021 |
| [Robust High-Resolution Video Matting with Temporal Guidance](http://arxiv.org/abs/2108.11515) | ArXiv:2108.11515 [Cs] | [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) | arXiv. 2021 |


### Datasets
| Title                                                                                                                        |                       Venue                        |                              Code                               | Year |
| :--------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------: | :-------------------------------------------------------------: | ---- |
| [Structured Local Radiance Fields for Human Avatar Modeling](http://arxiv.org/abs/2203.14478) | CVPR | [THUman4.0-Dataset](https://github.com/ZhengZerong/THUman4.0-Dataset)  | 2022 |


## SDF

- https://github.com/facebookresearch/pifuhd

| Title                                                                                                                                 |         Venue         |                          Code                          | Year |
| :------------------------------------------------------------------------------------------------------------------------------------ | :-------------------: | :----------------------------------------------------: | :--: |
| :white_check_mark: [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](http://arxiv.org/abs/1901.05103) | arXiv:1901.05103 [cs] | [DeepSDF](https://github.com/facebookresearch/DeepSDF) | 2019 |
| [Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling](http://arxiv.org/abs/1610.07584)      |        NeurIPS        |                                                        | 2016 |
| [Occupancy Networks: Learning 3D Reconstruction in Function Space](http://arxiv.org/abs/1812.03828)                                   | arXiv:1812.03828 [cs] |                                                        | 2019 |
| [PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization](http://arxiv.org/abs/1905.05172)               | arXiv:1905.05172 [cs] |                                                        | 2019 |
| [Deep Meta Functionals for Shape Representation](http://arxiv.org/abs/1908.06277)                                                     | arXiv:1908.06277 [cs] |                                                        | 2019 |

### 3D

| Title                                                                                                                                      |         Venue         |                             Code                             | Year |
| :----------------------------------------------------------------------------------------------------------------------------------------- | :-------------------: | :----------------------------------------------------------: | :--: |
| [Escaping Plato’s Cave: 3D Shape From Adversarial Rendering](http://arxiv.org/abs/1811.11606)                                              |         ICCV          |                                                              | 2019 |
| [StyleRig: Rigging StyleGAN for 3D Control over Portrait Images](http://arxiv.org/abs/2004.00121)                                          | arXiv:2004.00121 [cs] |                                                              | 2020 |
| [Exemplar-Based 3D Portrait Stylization](http://arxiv.org/abs/2104.14559)                                                                  | arXiv:2104.14559 [cs] | [github](https://github.com/halfjoe/3D-Portrait-Stylization) | 2021 |
| :heart: [Landmark Detection and 3D Face Reconstruction for Caricature Using a Nonlinear Parametric Model](http://arxiv.org/abs/2004.09190) | arXiv:2004.09190 [cs] |  [CaricatureFace](https://github.com/Juyong/CaricatureFace)  | 2021 |
| [SofGAN: A Portrait Image Generator with Dynamic Styling](http://arxiv.org/abs/2007.03780)                                                 | arXiv:2007.03780 [cs] |        [sofgan](https://github.com/apchenstu/sofgan)         | 2021 |
| :heart: [FreeStyleGAN: Free-View Editable Portrait Rendering with the Camera Manifold](http://arxiv.org/abs/2109.09378)                    | arXiv:2109.09378 [cs] |                                                              | 2021 |
| [PIRenderer: Controllable Portrait Image Generation via Semantic Neural Rendering](http://arxiv.org/abs/2109.08379)                        |         ICCV          |       [PIRender](https://github.com/RenYurui/PIRender)       | 2021 |

### 3DMM Face

- https://github.com/tencent-ailab/hifi3dface
- https://github.com/ascust/3DMM-Fitting-Pytorch

| Title                                                                                                                     |         Venue         |                                     Code                                     | Year |
| :------------------------------------------------------------------------------------------------------------------------ | :-------------------: | :--------------------------------------------------------------------------: | :--: |
| [Neural Head Reenactment with Latent Pose Descriptors](http://arxiv.org/abs/2004.12000)                                   |         CVPR          | [latent-pose-reenactment](https://github.com/shrubb/latent-pose-reenactment) | 2020 |
| [Synergy between 3DMM and 3D Landmarks for Accurate 3D Facial Geometry](http://arxiv.org/abs/2110.09772)                  | arXiv:2110.09772 [cs] |                                                                              | 2021 |

### Point Cloud

| Title                                                                                                                                                               |         Venue         | Code | Year |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :-------------------: | :--: | :--: |
| [Point-Based Modeling of Human Clothing](https://openaccess.thecvf.com/content/ICCV2021/html/Zakharkin_Point-Based_Modeling_of_Human_Clothing_ICCV_2021_paper.html) |         ICCV          |      | 2021 |
| [ADOP: Approximate Differentiable One-Pixel Point Rendering](http://arxiv.org/abs/2110.06635)                                                                       | arXiv:2110.06635 [cs] |      | 2021 |

### Stylization

| Title                                                              |         Venue         |                         Code                          | Year |
| :----------------------------------------------------------------- | :-------------------: | :---------------------------------------------------: | :--: |
| [Learning to Stylize Novel Views](http://arxiv.org/abs/2105.13509) | arXiv:2105.13509 [cs] | [stylescene](https://github.com/hhsinping/stylescene) | 2021 |

### Datasets

- https://github.com/ofirkris/Faces-datasets

| Title                                                                                                                                |                                     Venue                                     |                                 Code                                 | Year |
| :----------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------: | :------------------------------------------------------------------: | :--: |
| [Common Objects in 3D: Large-Scale Learning and Evaluation of Real-Life 3D Category Reconstruction](http://arxiv.org/abs/2109.00512) |                                     ICCV                                      |                                                                      | 2021 |
| [A 3D Face Model for Pose and Illumination Invariant Face Recognition]()                                                             | IEEE International Conference on Advanced Video and Signal Based Surveillance | [BFM](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads) | 2009 |
| [SfSNet: Learning Shape, Reflectance and Illuminance of Faces in the Wild](http://arxiv.org/abs/1712.01261)                          |                             arXiv:1712.01261 [cs]                             |                                                                      | 2018 |

### 3D-aware image synthesis (ref)

| Title                                                                                                           |            Venue            | Code | Year |
| :-------------------------------------------------------------------------------------------------------------- | :-------------------------: | :--: | :--: |
| [Visual Object Networks: Image Generation with Disentangled 3D Representation](http://arxiv.org/abs/1812.02725) | arXiv:1812.02725 [cs, stat] |      | 2018 |
| [Escaping Plato’s Cave: 3D Shape From Adversarial Rendering](http://arxiv.org/abs/1811.11606)                   |            ICCV             |      | 2019 |
| [HoloGAN: Unsupervised Learning of 3D Representations from Natural Images](http://arxiv.org/abs/1904.01326)     |            ICCV             |      | 2019 |

## Face

### Tools

- https://github.com/wuhuikai/FaceSwap
- https://github.com/hysts/anime-face-detector
- https://github.com/qq775193759/3D-CariGAN
- https://github.com/yeemachine/kalidokit
- https://github.com/sicxu/Deep3DFaceRecon_pytorch
- https://github.com/happy-jihye/face-vid2vid-demo

### Edit

| Title                                                                                                                                     |         Venue         | Code | Year |
| :---------------------------------------------------------------------------------------------------------------------------------------- | :-------------------: | :--: | :--: |
| [FaceEraser: Removing Facial Parts for Augmented Reality](http://arxiv.org/abs/2109.10760)                                                | arXiv:2109.10760 [cs] |      | 2021 |
| [DyStyle: Dynamic Neural Network for Multi-Attribute-Conditioned Style Editing](http://arxiv.org/abs/2109.10737)                          | arXiv:2109.10737 [cs] |      | 2021 |
| :heart: [StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators](http://arxiv.org/abs/2108.00946)                               | arXiv:2108.00946 [cs] |      | 2021 |
| [Beholder-GAN: Generation and Beautification of Facial Images with Conditioning on Their Beauty Level](http://arxiv.org/abs/1902.02593)   | arXiv:1902.02593 [cs] |      | 2019 |
| [Mind the Gap: Domain Gap Control for Single Shot Domain Adaptation for Generative Adversarial Networks](http://arxiv.org/abs/2110.08398) | arXiv:2110.08398 [cs] |      | 2021 |
| [Fine-Grained Control of Artistic Styles in Image Generation](http://arxiv.org/abs/2110.10278)                                            | arXiv:2110.10278 [cs] |      | 2021 |

### Anime Face

- https://github.com/Sxela/ArcaneGAN
- https://github.com/mchong6/GANsNRoses
- https://github.com/FilipAndersson245/cartoon-gan
- https://github.com/venture-anime/cartoongan-pytorch

| Title                                                                                                                          |            Venue            |                                          Code                                           | Year |
| :----------------------------------------------------------------------------------------------------------------------------- | :-------------------------: | :-------------------------------------------------------------------------------------: | :--: |
| [AniGAN: Style-Guided Generative Adversarial Networks for Unsupervised Anime Face Generation](http://arxiv.org/abs/2102.12593) |    arXiv:2102.12593 [cs]    |                                                                                         | 2021 |
| [AnimeGAN: A Novel Lightweight GAN for Photo Animation]                                                                        |                             |              [AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2)               | 2020 |
| :heart: [Learning to Cartoonize Using White-Box Cartoon Representations](https://ieeexplore.ieee.org/document/9157493/)        |            CVPR             | [White-box-Cartoonization](https://github.com/SystemErrorWang/White-box-Cartoonization) | 2020 |
| [Generative Adversarial Networks for Photo to Hayao Miyazaki Style Cartoons](http://arxiv.org/abs/2005.07702)                  | arXiv:2005.07702 [cs, eess] |                                                                                         | 2020 |

### 3DMM

- https://github.com/lattas/AvatarMe

| Title                                                                                    |                                           Venue                                           |  Code  |                              Year                               |
| :--------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------: | :----: | :-------------------------------------------------------------: |
| [A Morphable Model for the Synthesis of 3D Faces](https://doi.org/10.1145/311535.311556) | Proceedings of the 26th Annual Conference on Computer Graphics and Interactive Techniques | [3DMM] | SIGGRAPH ’99, USA: ACM Press/Addison-Wesley Publishing Co. 1999 |

### Face

| Title                                                                                      |         Venue         | Code | Year |
| :----------------------------------------------------------------------------------------- | :-------------------: | :--: | :--: |
| [SketchHairSalon: Deep Sketch-Based Hair Image Synthesis](http://arxiv.org/abs/2109.07874) | arXiv:2109.07874 [cs] |      | 2021 |

### Face Alignment

| Title                                                |                             Venue                              | Code | Year |
| :--------------------------------------------------- | :------------------------------------------------------------: | :--: | :--: |
| [Face Alignment Across Large Poses: A 3D Solution]() | IEEE Transactions on Pattern Analysis and Machine Intelligence |      | 2019 |

### Face Recognition

| Title                                                                                |                                 Venue                                  | Code | Year |
| :----------------------------------------------------------------------------------- | :--------------------------------------------------------------------: | :--: | :--: |
| [High-Fidelity Pose and Expression Normalization for Face Recognition in the Wild]() | 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) |      | 2015 |

### Face swapping

- https://github.com/mindslab-ai/hififace

## 3D

| Title                                                                                                                                                                   |               Venue               |                             Code                             | Year |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------: | :----------------------------------------------------------: | :--: |
| [Unsupervised Learning of Probably Symmetric Deformable 3D Objects from Images in the Wild](http://arxiv.org/abs/1911.11130)                                            |       arXiv:1911.11130 [cs]       |       [unsup3d](https://github.com/elliottwu/unsup3d)        | 2020 |
| [Do 2D GANs Know 3D Shape? Unsupervised 3D Shape Reconstruction from 2D Image GANs](http://arxiv.org/abs/2011.00844)                                                    |       arXiv:2011.00844 [cs]       |     [GAN2Shape](https://github.com/XingangPan/GAN2Shape)     | 2021 |
| [A Geometric Analysis of Deep Generative Image Models and Its Applications](https://openreview.net/forum?id=GH7QRzUDdXG)                                                |               ICLR                |                                                              | 2021 |
| [Lifting 2D StyleGAN for 3D-Aware Face Generation](http://arxiv.org/abs/2011.13126)                                                                                     |               CVPR                |      [LiftedGAN](https://github.com/seasonSH/LiftedGAN)      | 2021 |
| [Image GANs Meet Differentiable Rendering for Inverse Graphics and Interpretable 3D Neural Rendering](http://arxiv.org/abs/2010.09125)                                  |       arXiv:2010.09125 [cs]       |                                                              | 2021 |
| [Neural 3D Mesh Renderer](http://arxiv.org/abs/1711.07566)                                                                                                              |               CVPR                |                                                              | 2018 |
| [Fast-GANFIT: Generative Adversarial Network for High Fidelity 3D Face Reconstruction](http://arxiv.org/abs/2105.07474)                                                 |       arXiv:2105.07474 [cs]       |                                                              | 2021 |
| [Inverting Generative Adversarial Renderer for Face Reconstruction](http://arxiv.org/abs/2105.02431)                                                                    |               CVPR                | [StyleRenderer](https://github.com/WestlyPark/StyleRenderer) | 2021 |
| [Learning to Aggregate and Personalize 3D Face from In-the-Wild Photo Collection](http://arxiv.org/abs/2106.07852)                                                      |       arXiv:2106.07852 [cs]       |                                                              | 2021 |
| [Subdivision-Based Mesh Convolution Networks](http://arxiv.org/abs/2106.02285)                                                                                          |       arXiv:2106.02285 [cs]       |                                                              | 2021 |
| [Learning to Aggregate and Personalize 3D Face from In-the-Wild Photo Collection](http://arxiv.org/abs/2106.07852)                                                      |               CVPR                |                                                              | 2021 |
| [To Fit or Not to Fit: Model-Based Face Reconstruction and Occlusion Segmentation from Weak Supervision](http://arxiv.org/abs/2106.09614)                               |       arXiv:2106.09614 [cs]       |                                                              | 2021 |
| [Unsupervised Learning of Depth and Depth-of-Field Effect from Natural Images with Aperture Rendering Generative Adversarial Networks](http://arxiv.org/abs/2106.13041) | arXiv:2106.13041 [cs, eess, stat] |                                                              | 2021 |
| [DOVE: Learning Deformable 3D Objects by Watching Videos](http://arxiv.org/abs/2107.10844)                                                                              |       arXiv:2107.10844 [cs]       |                                                              | 2021 |
| [De-Rendering the World’s Revolutionary Artefacts](http://arxiv.org/abs/2104.03954)                                                                                     |               CVPR                |                                                              | 2021 |
| [Learning Generative Models of Textured 3D Meshes from Real-World Images](http://arxiv.org/abs/2103.15627)                                                              |               ICCV                |                                                              | 2021 |
| [Toward Realistic Single-View 3D Object Reconstruction with Unsupervised Learning from Multiple Images](http://arxiv.org/abs/2109.02288)                                |               ICCV                |                                                              | 2021 |

## DA

| Title                                                                                                                                          |         Venue         | Code | Year |
| :--------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------: | :--: | :--: |
| [Semi-Supervised Domain Adaptation via Adaptive and Progressive Feature Alignment](http://arxiv.org/abs/2106.02845)                            | arXiv:2106.02845 [cs] |      | 2021 |
| [Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Semantic Segmentation](http://arxiv.org/abs/2101.10979) | arXiv:2101.10979 [cs] |      | 2021 |

## Data

- https://github.com/koaning/doubtlab

| Title                                                                                                                                                              |            Venue            | Code | Year |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------: | :--: | :--: |
| :white_check_mark: [Semi-Supervised Active Learning with Temporal Output Discrepancy](http://arxiv.org/abs/2107.14153)                                             |            ICCV             |      | 2021 |
| :heart: [Mean Teachers Are Better Role Models: Weight-Averaged Consistency Targets Improve Semi-Supervised Deep Learning Results](http://arxiv.org/abs/1703.01780) |           NeurIPS           |      | 2017 |
| [When Deep Learners Change Their Mind: Learning Dynamics for Active Learning](http://arxiv.org/abs/2107.14707)                                                     |    arXiv:2107.14707 [cs]    |      | 2021 |
| [On The State of Data In Computer Vision: Human Annotations Remain Indispensable for Developing Deep Learning Models](http://arxiv.org/abs/2108.00114)             |    arXiv:2108.00114 [cs]    |      | 2021 |
| [StyleAugment: Learning Texture De-Biased Representations by Style Augmentation without Pre-Defined Textures](http://arxiv.org/abs/2108.10549)                     |    arXiv:2108.10549 [cs]    |      | 2021 |
| [Multi-Task Self-Training for Learning General Representations](http://arxiv.org/abs/2108.11353)                                                                   |            ICCV             |      | 2021 |
| [OOWL500: Overcoming Dataset Collection Bias in the Wild](http://arxiv.org/abs/2108.10992)                                                                         |    arXiv:2108.10992 [cs]    |      | 2021 |
| [Ghost Loss to Question the Reliability of Training Data]()                                                                                                        |         IEEE Access         |      | 2020 |
| [Revisiting 3D ResNets for Video Recognition](http://arxiv.org/abs/2109.01696)                                                                                     | arXiv:2109.01696 [cs, eess] |      | 2021 |
| :heart: [Revisiting ResNets: Improved Training and Scaling Strategies](http://arxiv.org/abs/2103.07579)                                                            |    arXiv:2103.07579 [cs]    |      | 2021 |
| [Learning Fast Sample Re-Weighting Without Reward Data](http://arxiv.org/abs/2109.03216)                                                                           |            ICCV             |      | 2021 |
| [How Important Is Importance Sampling for Deep Budgeted Training?](http://arxiv.org/abs/2110.14283)                                                                |    arXiv:2110.14283 [cs]    |      | 2021 |

## CNN & BN

### Light architecture

- https://github.com/yoshitomo-matsubara/torchdistill

| Title                                                                                                                                |         Venue         | Code | Year |
| :----------------------------------------------------------------------------------------------------------------------------------- | :-------------------: | :--: | :--: |
| [Network Augmentation for Tiny Deep Learning](http://arxiv.org/abs/2110.08890)                                                       | arXiv:2110.08890 [cs] |      | 2021 |
| [Non-Deep Networks](http://arxiv.org/abs/2110.07641)                                                                                 | arXiv:2110.07641 [cs] |      | 2021 |
| [When to Prune? A Policy towards Early Structural Pruning](http://arxiv.org/abs/2110.12007)                                          | arXiv:2110.12007 [cs] |      | 2021 |
| :heart: [ConformalLayers: A Non-Linear Sequential Neural Network with Associative Layers](http://arxiv.org/abs/2110.12108)           | arXiv:2110.12108 [cs] |      | 2021 |
| [CHIP: CHannel Independence-Based Pruning for Compact Neural Networks](http://arxiv.org/abs/2110.13981)                              | arXiv:2110.13981 [cs] |      | 2021 |
| [Do We Actually Need Dense Over-Parameterization? In-Time Over-Parameterization in Sparse Training](http://arxiv.org/abs/2102.02887) | arXiv:2102.02887 [cs] |      | 2021 |

### Antialiased CNNs

| Title                                                                                            |         Venue         | Code |    Year     |
| :----------------------------------------------------------------------------------------------- | :-------------------: | :--: | :---------: |
| [Making Convolutional Networks Shift-Invariant Again](http://arxiv.org/abs/1904.11486)           | arXiv:1904.11486 [cs] |      |    2019     |
| [Group Equivariant Convolutional Networks](http://arxiv.org/abs/1602.07576)                      |         ICML          |      | arXiv. 2016 |
| [Harmonic Networks: Deep Translation and Rotation Equivariance](http://arxiv.org/abs/1612.04642) |         CVPR          |      | arXiv. 2017 |
| [Learning Steerable Filters for Rotation Equivariant CNNs](http://arxiv.org/abs/1711.07289)      |         CVPR          |      | arXiv. 2018 |

### Architecture

| Title                                                                                                                                                           |            Venue            |                           Code                           | Year |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------: | :------------------------------------------------------: | :--: |
| [Beyond BatchNorm: Towards a General Understanding of Normalization in Deep Learning](http://arxiv.org/abs/2106.05956)                                          |    arXiv:2106.05956 [cs]    |                                                          | 2021 |
| [R-Drop: Regularized Dropout for Neural Networks](http://arxiv.org/abs/2106.14448)                                                                              |    arXiv:2106.14448 [cs]    |                                                          | 2021 |
| [Switchable Whitening for Deep Representation Learning](http://arxiv.org/abs/1904.09739)                                                                        |            ICCV             |                                                          | 2019 |
| [Positional Normalization](http://arxiv.org/abs/1907.04312)                                                                                                     |    arXiv:1907.04312 [cs]    |                                                          | 2019 |
| [On Feature Normalization and Data Augmentation](http://arxiv.org/abs/2002.11102)                                                                               | arXiv:2002.11102 [cs, stat] |                                                          | 2021 |
| [Channel Equilibrium Networks for Learning Deep Representation](http://arxiv.org/abs/2003.00214)                                                                |    arXiv:2003.00214 [cs]    |                                                          | 2020 |
| [Representative Batch Normalization with Feature Calibration]()                                                                                                 |            CVPR             |                                                          | 2021 |
| [EPSANet: An Efficient Pyramid Squeeze Attention Block on Convolutional Neural Network](http://arxiv.org/abs/2105.14447)                                        |    arXiv:2105.14447 [cs]    |                                                          | 2021 |
| [Bias Loss for Mobile Neural Networks](http://arxiv.org/abs/2107.11170)                                                                                         |    arXiv:2107.11170 [cs]    |                                                          | 2021 |
| [Compositional Models: Multi-Task Learning and Knowledge Transfer with Modular Networks](http://arxiv.org/abs/2107.10963)                                       |    arXiv:2107.10963 [cs]    |                                                          | 2021 |
| [Log-Polar Space Convolution for Convolutional Neural Networks](http://arxiv.org/abs/2107.11943)                                                                |    arXiv:2107.11943 [cs]    |                                                          | 2021 |
| [Decoupled Dynamic Filter Networks](http://arxiv.org/abs/2104.14107)                                                                                            |    arXiv:2104.14107 [cs]    |                                                          | 2021 |
| [Spectral Leakage and Rethinking the Kernel Size in CNNs](http://arxiv.org/abs/2101.10143)                                                                      |    arXiv:2101.10143 [cs]    |                                                          | 2021 |
| [Learning with Noisy Labels via Sparse Regularization](http://arxiv.org/abs/2108.00192)                                                                         |            ICCV             |                                                          | 2021 |
| :heart: [Impact of Aliasing on Generalization in Deep Convolutional Networks](http://arxiv.org/abs/2108.03489)                                                  |            ICCV             |                                                          | 2021 |
| [Orthogonal Over-Parameterized Training](http://arxiv.org/abs/2004.04690)                                                                                       |            CVPR             |                                                          | 2021 |
| [Multiplying Matrices Without Multiplying](http://arxiv.org/abs/2106.10860)                                                                                     |            ICML             |                                                          | 2021 |
| [AASeg: Attention Aware Network for Real Time Semantic Segmentation](http://arxiv.org/abs/2108.04349)                                                           | arXiv:2108.04349 [cs, eess] |                                                          | 2021 |
| [MicroNet: Improving Image Recognition with Extremely Low FLOPs](http://arxiv.org/abs/2108.05894)                                                               |            ICCV             |                                                          | 2021 |
| [Contextual Convolutional Neural Networks](http://arxiv.org/abs/2108.07387)                                                                                     |    arXiv:2108.07387 [cs]    |                                                          | 2021 |
| [Torch.Manual_seed(3407) Is All You Need: On the Influence of Random Seeds in Deep Learning Architectures for Computer Vision](http://arxiv.org/abs/2109.08203) |    arXiv:2109.08203 [cs]    |                                                          | 2021 |
| [KATANA: Simple Post-Training Robustness Using Test Time Augmentations](http://arxiv.org/abs/2109.08191)                                                        |    arXiv:2109.08191 [cs]    |                                                          | 2021 |
| [Global Pooling, More than Meets the Eye: Position Information Is Encoded Channel-Wise in CNNs](http://arxiv.org/abs/2108.07884)                                |            ICCV             |                                                          | 2021 |
| :white_check_mark: [A ConvNet for the 2020s](http://arxiv.org/abs/2201.03545)                                                                                   |    arXiv:2201.03545 [cs]    | [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) | 2022 |

### Compression

| Title                                                                                                    |         Venue         | Code | Year |
| :------------------------------------------------------------------------------------------------------- | :-------------------: | :--: | :--: |
| [AdaPruner: Adaptive Channel Pruning and Effective Weights Inheritance](http://arxiv.org/abs/2109.06397) | arXiv:2109.06397 [cs] |      | 2021 |

### Detection

| Title                                                                                                      |         Venue         | Code | Year |
| :--------------------------------------------------------------------------------------------------------- | :-------------------: | :--: | :--: |
| [Anchor DETR: Query Design for Transformer-Based Detector](http://arxiv.org/abs/2109.07107)                | arXiv:2109.07107 [cs] |      | 2021 |
| :heart: [Detecting Twenty-Thousand Classes Using Image-Level Supervision](http://arxiv.org/abs/2201.02605) | arXiv:2201.02605 [cs] |      | 2022 |

### Segmentation

- https://github.com/xuebinqin/U-2-Net#usage-for-portrait-generation

| Title                                                           |          Venue           | Code | Year |
| :-------------------------------------------------------------- | :----------------------: | :--: | :--: |
| [Robust High-Resolution Video Matting with Temporal Guidance]() | arXiv:2108.11515 [cs.CV] |      | 2021 |

### MLP

| Title                                                                                                                 |          Venue           | Code | Year |
| :-------------------------------------------------------------------------------------------------------------------- | :----------------------: | :--: | :--: |
| [ResMLP: Feedforward Networks for Image Classification with Data-Efficient Training](http://arxiv.org/abs/2105.03404) |  arXiv:2105.03404 [cs]   |      | 2021 |
| [ConvMLP: Hierarchical Convolutional MLPs for Vision](http://arxiv.org/abs/2109.04454)                                |  arXiv:2109.04454 [cs]   |      | 2021 |
| [A Battle of Network Structures: An Empirical Study of CNN, Transformer, and MLP]()                                   | arXiv:2108.13002 [cs.CV] |      | 2021 |
| [Sparse-MLP: A Fully-MLP Architecture with Conditional Computation](http://arxiv.org/abs/2109.02008)                  |  arXiv:2109.02008 [cs]   |      | 2021 |
| [MLP-Mixer: An All-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601v1)                                   |           2021           |
| [CycleMLP: A MLP-like Architecture for Dense Prediction](http://arxiv.org/abs/2107.10224)                             |           ICLR           |      | 2022 |

### Transformer

- https://github.com/xxxnell/how-do-vits-work

| Title                                                                                                                                                  |            Venue            |                       Code                       | Year |
| :----------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------: | :----------------------------------------------: | :--: |
| [Training Data-Efficient Image Transformers & Distillation through Attention](http://arxiv.org/abs/2012.12877)                                         |    arXiv:2012.12877 [cs]    | [deit](https://github.com/facebookresearch/deit) | 2020 |
| [Intriguing Properties of Vision Transformers](http://arxiv.org/abs/2105.10497)                                                                        |    arXiv:2105.10497 [cs]    |                                                  | 2021 |
| [CogView: Mastering Text-to-Image Generation via Transformers](http://arxiv.org/abs/2105.13290)                                                        |    arXiv:2105.13290 [cs]    |                                                  | 2021 |
| [An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale](http://arxiv.org/abs/2010.11929)                                          |    arXiv:2010.11929 [cs]    |                                                  | 2021 |
| [Scaling Vision Transformers](http://arxiv.org/abs/2106.04560)                                                                                         |    arXiv:2106.04560 [cs]    |                                                  | 2021 |
| [IA-RED$^2$: Interpretability-Aware Redundancy Reduction for Vision Transformers](http://arxiv.org/abs/2106.12620)                                     |    arXiv:2106.12620 [cs]    |                                                  | 2021 |
| [Rethinking and Improving Relative Position Encoding for Vision Transformer]()                                                                         |            ICCV             |                                                  | 2021 |
| [Go Wider Instead of Deeper](http://arxiv.org/abs/2107.11817)                                                                                          |    arXiv:2107.11817 [cs]    |                                                  | 2021 |
| [A Unified Efficient Pyramid Transformer for Semantic Segmentation](http://arxiv.org/abs/2107.14209)                                                   |    arXiv:2107.14209 [cs]    |                                                  | 2021 |
| :heart: [Conditional DETR for Fast Training Convergence](http://arxiv.org/abs/2108.06152)                                                              |            ICCV             |                                                  | 2021 |
| :heart: [Sketch Your Own GAN](http://arxiv.org/abs/2108.02774)                                                                                         |            ICCV             |                                                  | 2021 |
| [CrossFormer: A Versatile Vision Transformer Based on Cross-Scale Attention](http://arxiv.org/abs/2108.00154)                                          |    arXiv:2108.00154 [cs]    |                                                  | 2021 |
| [Uformer: A General U-Shaped Transformer for Image Restoration](http://arxiv.org/abs/2106.03106)                                                       |    arXiv:2106.03106 [cs]    |                                                  | 2021 |
| [ConvNets vs. Transformers: Whose Visual Representations Are More Transferable?](http://arxiv.org/abs/2108.05305)                                      |    arXiv:2108.05305 [cs]    |                                                  | 2021 |
| [Mobile-Former: Bridging MobileNet and Transformer](http://arxiv.org/abs/2108.05895)                                                                   |    arXiv:2108.05895 [cs]    |                                                  | 2021 |
| [SOTR: Segmenting Objects with Transformers](http://arxiv.org/abs/2108.06747)                                                                          |            ICCV             |                                                  | 2021 |
| [Video Transformer Network](http://arxiv.org/abs/2102.00719)                                                                                           |    arXiv:2102.00719 [cs]    |                                                  | 2021 |
| [Do Vision Transformers See Like Convolutional Neural Networks?](http://arxiv.org/abs/2108.08810)                                                      | arXiv:2108.08810 [cs, stat] |                                                  | 2021 |
| [UCTransNet: Rethinking the Skip Connections in U-Net from a Channel-Wise Perspective with Transformer](http://arxiv.org/abs/2109.04335)               | arXiv:2109.04335 [cs, eess] |                                                  | 2021 |
| [$\infty$-Former: Infinite Memory Transformer](http://arxiv.org/abs/2109.00301)                                                                        |    arXiv:2109.00301 [cs]    |                                                  | 2021 |
| [PnP-DETR: Towards Efficient Visual Analysis with Transformers](http://arxiv.org/abs/2109.07036)                                                       |            ICCV             |                                                  | 2021 |
| [MobileViT: Light-Weight, General-Purpose, and Mobile-Friendly Vision Transformer](http://arxiv.org/abs/2110.02178)                                    |    arXiv:2110.02178 [cs]    |                                                  | 2021 |
| [MetaFormer Is Actually What You Need for Vision](http://arxiv.org/abs/2111.11418)                                                                     |    arXiv:2111.11418 [cs]    |                                                  | 2021 |
| [Restormer: Efficient Transformer for High-Resolution Image Restoration](http://arxiv.org/abs/2111.09881)                                              |    arXiv:2111.09881 [cs]    | [Restormer](https://github.com/swz30/Restormer)  | 2021 |
| :white_check_mark: [An Empirical Study of Training Self-Supervised Vision Transformers](http://arxiv.org/abs/2104.02057)                               |    arXiv:2104.02057 [cs]    |                                                  | 2021 |
| :white_check_mark: [When Vision Transformers Outperform ResNets without Pre-Training or Strong Data Augmentations](https://arxiv.org/abs/2106.01548v2) |  arXiv:2106.01548 [cs.CV]   |                                                  | 2021 |
| [Visual Attention Network](http://arxiv.org/abs/2202.09741)                                                                                            |    arXiv:2202.09741 [cs]    |                                                  | 2022 |

### ssl

- https://github.com/ucasligang/awesome-MIM

| Title                                                                                                                       |         Venue         |                            Code                            | Year |
| :-------------------------------------------------------------------------------------------------------------------------- | :-------------------: | :--------------------------------------------------------: | :--: |
| [Emerging Properties in Self-Supervised Vision Transformers](http://arxiv.org/abs/2104.14294)                               | arXiv:2104.14294 [cs] |      [dino](https://github.com/facebookresearch/dino)      | 2021 |
| [What Is Considered Complete for Visual Recognition?](http://arxiv.org/abs/2105.13978)                                      | arXiv:2105.13978 [cs] |                                                            | 2021 |
| [On the Efficacy of Small Self-Supervised Contrastive Models without Distillation Signals](http://arxiv.org/abs/2107.14762) | arXiv:2107.14762 [cs] |                                                            | 2021 |
| :heart: [Improving Contrastive Learning by Visualizing Feature Transformation](http://arxiv.org/abs/2108.02982)             |         ICCV          |                                                            | 2021 |
| [Scale Efficiently: Insights from Pre-Training and Fine-Tuning Transformers](http://arxiv.org/abs/2109.10686)               | arXiv:2109.10686 [cs] |                                                            | 2021 |
| [FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling](http://arxiv.org/abs/2110.08263)             | arXiv:2110.08263 [cs] |                                                            | 2021 |
| [BEiT: BERT Pre-Training of Image Transformers](http://arxiv.org/abs/2106.08254)                                            | arXiv:2106.08254 [cs] |                                                            | 2021 |
| :heart: [Parametric Contrastive Learning](http://arxiv.org/abs/2107.12028)                                                  |         ICCV          |                                                            | 2021 |
| :heart: [ImageNet-21K Pretraining for the Masses](http://arxiv.org/abs/2104.10972)                                          |        NeurIPS        | [ImageNet21K](https://github.com/Alibaba-MIIL/ImageNet21K) | 2021 |
| :heart: [ML-Decoder: Scalable and Versatile Classification Head](http://arxiv.org/abs/2111.12933)                           | arXiv:2111.12933 [cs] |  [ML_Decoder](https://github.com/Alibaba-MIIL/ML_Decoder)  | 2021 |
| [Asymmetric Loss For Multi-Label Classification](http://arxiv.org/abs/2009.14119)                                           |         ICCV          |         [ASL](https://github.com/Alibaba-MIIL/ASL)         | 2021 |
| [Grounded Language-Image Pre-Training](http://arxiv.org/abs/2112.03857)                                                     | arXiv:2112.03857 [cs] |                                                            | 2021 |

## Finetune

| Title                                                                                            |        Venue         | Code | Year |
| :----------------------------------------------------------------------------------------------- | :------------------: | :--: | :--: |
| :heart: [How Transferable Are Features in Deep Neural Networks?](http://arxiv.org/abs/1411.1792) | arXiv:1411.1792 [cs] |      | 2014 |

## Positional Encoding

| Title                                                                                                                            |            Venue            | Code | Year |
| :------------------------------------------------------------------------------------------------------------------------------- | :-------------------------: | :--: | :--: |
| [Positional Encoding as Spatial Inductive Bias in GANs](http://arxiv.org/abs/2012.05217)                                         |    arXiv:2012.05217 [cs]    |      | 2020 |
| [Mind the Pad -- CNNs Can Develop Blind Spots](http://arxiv.org/abs/2010.02178)                                                  | arXiv:2010.02178 [cs, stat] |      | 2020 |
| :heart: [How Much Position Information Do Convolutional Neural Networks Encode?](http://arxiv.org/abs/2001.08248)                |            ICLR             |      | 2020 |
| [On Translation Invariance in CNNs: Convolutional Layers Can Exploit Absolute Spatial Location](http://arxiv.org/abs/2003.07064) |            CVPR             |      | 2020 |
| [Rethinking and Improving Relative Position Encoding for Vision Transformer]()                                                   |            ICCV             |      | 2021 |
| [A Structured Dictionary Perspective on Implicit Neural Representations](http://arxiv.org/abs/2112.01917)                        |    arXiv:2112.01917 [cs]    |      | 2021 |

## NAS

### NAS cls

| Title                                                                                                 | Venue | Code | Year |
| :---------------------------------------------------------------------------------------------------- | :---: | :--: | :--: |
| [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578v2)          | ICLR  |      | 2017 |
| [Learning Transferable Architectures for Scalable Image Recognition](http://arxiv.org/abs/1707.07012) | CVPR  |      | 2018 |
| [Progressive Neural Architecture Search](http://arxiv.org/abs/1712.00559)                             | ECCV  |      | 2018 |
| [Efficient Neural Architecture Search via Parameter Sharing](http://arxiv.org/abs/1802.03268)         | ICML  |      | 2018 |
| [MnasNet: Platform-Aware Neural Architecture Search for Mobile](http://arxiv.org/abs/1807.11626)      | CVPR  |      | 2019 |
| [DARTS: Differentiable Architecture Search](http://arxiv.org/abs/1806.09055)                          | ICLR  |      | 2019 |

### NAS GAN

| Title                                                                                                                                                                                                                             |                             Venue                              | Code | Year |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------: | :--: | :--: |
| [AlphaGAN: Fully Differentiable Architecture Search for Generative Adversarial Networks]()                                                                                                                                        | IEEE Transactions on Pattern Analysis and Machine Intelligence |      | 2021 |
| [GAN Compression: Efficient Architectures for Interactive Conditional GANs](http://openaccess.thecvf.com/content_CVPR_2020/html/Li_GAN_Compression_Efficient_Architectures_for_Interactive_Conditional_GANs_CVPR_2020_paper.html) |                              CVPR                              |      | 2020 |
| [Off-Policy Reinforcement Learning for Efficient and Effective GAN Architecture Search](http://arxiv.org/abs/2007.09180)                                                                                                          |                              ECCV                              |      | 2020 |
| [AutoGAN-Distiller: Searching to Compress Generative Adversarial Networks](http://arxiv.org/abs/2006.08198)                                                                                                                       |                              ICML                              |      | 2020 |
| [A Multi-Objective Architecture Search for Generative Adversarial Networks](https://doi.org/10.1145/3377929.3390004)                                                                                                              |                                                                |      | 2020 |
| [AutoGAN: Neural Architecture Search for Generative Adversarial Networks](http://arxiv.org/abs/1908.03835)                                                                                                                        |                              ICCV                              |      | 2019 |

## Low-level

### Super-resolution

- https://github.com/nihui/realsr-ncnn-vulkan

### Frame Interpolation

| Title                                                                         |         Venue         | Code | Year |
| :---------------------------------------------------------------------------- | :-------------------: | :--: | :--: |
| [FILM: Frame Interpolation for Large Motion](http://arxiv.org/abs/2202.04901) | arXiv:2202.04901 [cs] |      | 2022 |

### Denoising

| Title                                                                            |                 Venue                 |                     Code                      | Year |
| :------------------------------------------------------------------------------- | :-----------------------------------: | :-------------------------------------------: | :--: |
| [Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering]()       | IEEE Transactions on Image Processing |                                               | 2007 |
| [Towards Flexible Blind JPEG Artifacts Removal](http://arxiv.org/abs/2109.14573) |      arXiv:2109.14573 [cs, eess]      | [FBCNN](https://github.com/jiaxi-jiang/FBCNN) | 2021 |
