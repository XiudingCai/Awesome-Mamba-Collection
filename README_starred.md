#  Awesome-Mamba-Collection
![Awesome](https://awesome.re/badge.svg) ![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) ![Stars](https://img.shields.io/github/stars/XiudingCai/Awesome-Mamba-Collection)

Welcome to Awesome Mamba Resources! This repository is a curated collection of papers, tutorials, videos, and other valuable resources related to Mamba. Whether you're a beginner or an experienced user, this collection aims to provide a comprehensive reference for all things Mamba. Explore the latest research papers, dive into helpful tutorials, and discover insightful videos to enhance your understanding and proficiency in Mamba. Join us in this open collaboration to foster knowledge sharing and empower the Mamba community. Let's embark on an exciting journey with Mamba!

Feel free to modify and customize this introduction according to your preferences. Good luck with your repository! If you have any other questions, feel free to ask. -by ChatGPT

If you want to see the star count of each paper's code, switch to [this](https://github.com/XiudingCai/Awesome-Mamba-Collection/blob/main/README_starred.md).

Enjoy it below!

- [ Papers](#head2)
  - [ Architecture](#head3)
  - [Theoretical Analysis](#head4)
  - [ Vision](#head5)
  - [ Language](#head6)
  - [ Multi-Modal](#head7)
  - [ Medical](#head8)
  - [Tabular Data](#head9)
  - [ Graph](#head10)
  - [Point Cloud](#head11)
  - [Time Series](#head12)
  - [ Speech](#head13)
  - [Recommendation ](#head14)
  - [Reinforcement Learning](#head15)
  - [ Survey](#head16)
- [ Tutorials](#head17)
  - [ Blogs](#head18)
  - [ Videos](#head19)
  - [ Books](#head20)
  - [ Codes](#head21)
  - [Other Awesome Mamba List](#head22)
- [ Contributions](#head23)
  - [Contribute in 3 Steps](#head24)
  - [ Guidelines](#head25)
- [ Acknowledgement](#head26)

<span id="head2"></span>
##  Papers
<span id="head3"></span>
###  Architecture
* Mamba: Linear-Time Sequence Modeling with Selective State Spaces [[paper](https://arxiv.org/abs/2312.00752)] [[code](https://github.com/state-spaces/mamba)] (2023.12.01) ![Stars](https://img.shields.io/github/stars/state-spaces/mamba) 
* MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts [[paper](https://arxiv.org/abs/2401.04081)] [[code](https://github.com/llm-random/llm-random)] (2024.01.08) ![Stars](https://img.shields.io/github/stars/llm-random/llm-random) 
* BlackMamba: Mixture of Experts for State-Space Models [[paper](https://arxiv.org/abs/2402.01771)] [[code](https://github.com/zyphra/blackmamba)] (2024.02.01) ![Stars](https://img.shields.io/github/stars/zyphra/blackmamba) 
* Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling [[paper](https://arxiv.org/abs/2402.10211)] [[homepage](https://hiss-csp.github.io/)] (2024.02.15)
* DenseMamba: State Space Models with Dense Hidden Connection for Efficient Large Language Models [[paper](https://arxiv.org/abs/2403.00818)] [[code](https://github.com/WailordHe/DenseSSM)] (2024.02.26) ![Stars](https://img.shields.io/github/stars/WailordHe/DenseSSM) 
* Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models [[paper](https://arxiv.org/abs/2402.19427)] [[code](https://github.com/kyegomez/Griffin)] (2024.02.29) ![Stars](https://img.shields.io/github/stars/kyegomez/Griffin) 
* Jamba: A Hybrid Transformer-Mamba Language Model [[paper](https://arxiv.org/abs/2403.19887)] [[code](https://github.com/kyegomez/Jamba)] (2024.03.28) ![Stars](https://img.shields.io/github/stars/kyegomez/Jamba) 
* Transformers are SSMs: Generalized Models and Efficient Algorithms with Structured State Space Duality [[poster](https://icml.cc/virtual/2024/poster/32613)]

<span id="head4"></span>
### Theoretical Analysis
* StableSSM: Alleviating the Curse of Memory in State-space Models through Stable Reparameterization [[paper](https://arxiv.org/abs/2311.14495)] (2023.11.24)
* From Generalization Analysis to Optimization Designs for State Space Models [[paper](https://arxiv.org/abs/2405.02670)] (2024.05.04)

<span id="head5"></span>
###  Vision
- Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model [[paper](https://arxiv.org/abs/2401.09417)] [[code](https://github.com/hustvl/Vim)] (2024.01.17) ![Stars](https://img.shields.io/github/stars/hustvl/Vim) 
- VMamba: Visual State Space Model [[paper](https://arxiv.org/abs/2401.10166)] [[code](https://github.com/MzeroMiko/VMamba)] (2024.01.18) ![Stars](https://img.shields.io/github/stars/MzeroMiko/VMamba) 
- U-shaped Vision Mamba for Single Image Dehazing [[paper](https://arxiv.org/abs/2402.04139)] (2024.02.06)
- Scalable Diffusion Models with State Space Backbone [[paper](https://arxiv.org/abs/2402.05608)] [[code](https://github.com/feizc/dis)] (2024.02.08) ![Stars](https://img.shields.io/github/stars/feizc/dis) 
- Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data [[paper](https://arxiv.org/abs/2402.05892)] (2024.02.08)
- Pan-Mamba: Effective pan-sharpening with State Space Model [[paper](https://arxiv.org/abs/2402.12192)] (2024.02.19)
- MambaIR: A Simple Baseline for Image Restoration with State-Space Model [[paper](https://arxiv.org/abs/2402.15648)] [[code](https://github.com/csguoh/mambair)] (2024.02.23) ![Stars](https://img.shields.io/github/stars/csguoh/mambair) 
- Res-VMamba: Fine-Grained Food Category Visual Classification Using Selective State Space Models with Deep Residual Learning [[paper](https://arxiv.org/abs/2402.15761)] [[code](https://github.com/chishengchen/resvmamba)] (2024.02.24) ![Stars](https://img.shields.io/github/stars/chishengchen/resvmamba) 
- MiM-ISTD: Mamba-in-Mamba for Efficient Infrared Small Target Detection [[paper](https://arxiv.org/abs/2403.02148)] [[code](https://github.com/txchen-USTC/MiM-ISTD)] (2024.03.04) ![Stars](https://img.shields.io/github/stars/txchen-USTC/MiM-ISTD) 
- Motion Mamba: Efficient and Long Sequence Motion Generation with Hierarchical and Bidirectional Selective SSM [[paper](https://arxiv.org/abs/2403.07487)] [[code](https://github.com/steve-zeyu-zhang/MotionMamba)] (2024.03.12) ![Stars](https://img.shields.io/github/stars/steve-zeyu-zhang/MotionMamba) 
- Video Mamba Suite: State Space Model as a Versatile Alternative for Video Understanding [[paper](https://arxiv.org/abs/2403.09626)] [[code](https://github.com/opengvlab/video-mamba-suite)] (2024.03.12) ![Stars](https://img.shields.io/github/stars/opengvlab/video-mamba-suite) 
- LocalMamba: Visual State Space Model with Windowed Selective Scan [[paper](https://arxiv.org/abs/2403.09338)] [[code](https://github.com/hunto/LocalMamba)] (2024.03.14) ![Stars](https://img.shields.io/github/stars/hunto/LocalMamba) 
- EfficientVMamba: Atrous Selective Scan for Light Weight Visual Mamba [[paper](https://arxiv.org/abs/2403.09977)] (2024.03.15)
- VmambaIR: Visual State Space Model for Image Restoration [[paper](https://arxiv.org/abs/2403.11423)] [[code](https://github.com/alphacatplus/vmambair)] (2024.03.18) ![Stars](https://img.shields.io/github/stars/alphacatplus/vmambair) 
- ZigMa: Zigzag Mamba Diffusion Model [[paper](https://arxiv.org/abs/2403.13802)] [[code](https://taohu.me/zigma/)] (2024.03.20)
- SiMBA: Simplified Mamba-Based Architecture for Vision and Multivariate Time series [[paper](https://arxiv.org/abs/2403.15360)] (2024.03.22)
- VMRNN: Integrating Vision Mamba and LSTM for Efficient and Accurate Spatiotemporal Forecasting [[paper](https://arxiv.org/abs/2403.16536)] [[code](https://github.com/yyyujintang/VMRNN-PyTorch)] (2024.03.25) (CVPR24 Precognition Workshop) ![Stars](https://img.shields.io/github/stars/yyyujintang/VMRNN-PyTorch) 
- PlainMamba: Improving Non-Hierarchical Mamba in Visual Recognition [[paper](https://arxiv.org/abs/2403.17695)] [[code](https://github.com/ChenhongyiYang/PlainMamba)] (2024.03.26) ![Stars](https://img.shields.io/github/stars/ChenhongyiYang/PlainMamba) 
- ReMamber: Referring Image Segmentation with Mamba Twister [[paper](https://arxiv.org/abs/2403.17839)] (2024.03.26)
- Gamba: Marry Gaussian Splatting with Mamba for single view 3D reconstruction [[paper](https://arxiv.org/abs/2403.18795)] (2024.03.27)
- RSMamba: Remote Sensing Image Classification with State Space Model [[paper](https://arxiv.org/abs/2403.19654)] [[code](https://github.com/KyanChen/RSMamba)] (2024.03.28) ![Stars](https://img.shields.io/github/stars/KyanChen/RSMamba) 
- MambaMixer: Efficient Selective State Space Models with Dual Token and Channel Selection [[paper](https://arxiv.org/abs/2403.19888)] (2024.03.29)
- HSIMamba: Hyperpsectral Imaging Efficient Feature Learning with Bidirectional State Space for Classification [[paper](https://arxiv.org/abs/2404.00272)] (2024.03.30)
- Samba: Semantic Segmentation of Remotely Sensed Images with State Space Model [[paper](https://arxiv.org/abs/2404.01705)] [[code](https://github.com/zhuqinfeng1999/Samba)] (2024.04.02) ![Stars](https://img.shields.io/github/stars/zhuqinfeng1999/Samba) 
- RS-Mamba for Large Remote Sensing Image Dense Prediction [[paper](https://arxiv.org/abs/2404.02668)] [[code](https://github.com/walking-shadow/Official_Remote_Sensing_Mamba)] (2024.04.03) ![Stars](https://img.shields.io/github/stars/walking-shadow/Official_Remote_Sensing_Mamba) 
- RS3Mamba: Visual State Space Model for Remote Sensing Images Semantic Segmentation [[paper](https://arxiv.org/abs/2404.02457)] (2024.04.03)
- InsectMamba: Insect Pest Classification with State Space Model [[paper](https://arxiv.org/abs/2404.03611)] (2024.04.04)
- ChangeMamba: Remote Sensing Change Detection with Spatio-Temporal State Space Model [[paper](https://arxiv.org/abs/2404.04256)] [[code](https://github.com/zifuwan/Sigma)] (2024.04.05) ![Stars](https://img.shields.io/github/stars/zifuwan/Sigma) 
- RhythmMamba: Fast Remote Physiological Measurement with Arbitrary Length Videos [[paper](https://arxiv.org/abs/2404.06483)] [[code](https://github.com/zizheng-guo/RhythmMamba)] (2024.04.09) ![Stars](https://img.shields.io/github/stars/zizheng-guo/RhythmMamba) 
- Simba: Mamba augmented U-ShiftGCN for Skeletal Action Recognition in Videos [[paper](https://arxiv.org/abs/2404.07645)] (2024.04.11)
- FusionMamba: Efficient Image Fusion with State Space Model [[paper](https://arxiv.org/abs/2404.07932)] (2024.04.11)
- DGMamba: Domain Generalization via Generalized State Space Model [[paper](https://arxiv.org/abs/2404.07794)] [[code](https://github.com/longshaocong/DGMamba)] (2024.04.11) ![Stars](https://img.shields.io/github/stars/longshaocong/DGMamba) 
- SpectralMamba: Efficient Mamba for Hyperspectral Image Classification [[paper](https://arxiv.org/abs/2404.08489)] (2024.04.12)
- FreqMamba: Viewing Mamba from a Frequency Perspective for Image Deraining [[paper](https://arxiv.org/abs/2404.09476)] (2024.04.15)
- HSIDMamba: Exploring Bidirectional State-Space Models for Hyperspectral Denoising [[paper](https://arxiv.org/abs/2404.09697)] (2024.04.15)
- A Novel State Space Model with Local Enhancement and State Sharing for Image Fusion [[paper](https://arxiv.org/abs/2404.09293)] (2024.04.15)
- FusionMamba: Dynamic Feature Enhancement for Multimodal Image Fusion with Mamba [[paper](https://arxiv.org/abs/2404.09498)] [[code](https://github.com/millieXie/FusionMamba)] (2024.04.15) ![Stars](https://img.shields.io/github/stars/millieXie/FusionMamba) 
- MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection [[paper](https://arxiv.org/abs/2404.06564)] [[project](https://lewandofskee.github.io/projects/MambaAD/)] [[code](https://github.com/lewandofskee/MambaAD)] (2024.04.15) ![Stars](https://img.shields.io/github/stars/lewandofskee/MambaAD) 
- Text-controlled Motion Mamba: Text-Instructed Temporal Grounding of Human Motion [[paper](https://arxiv.org/abs/2404.11375)] (2024.04.17)
- CU-Mamba: Selective State Space Models with Channel Learning for Image Restoration [[paper](https://arxiv.org/abs/2404.11778)] (2024.04.17)
- MambaPupil: Bidirectional Selective Recurrent model for Event-based Eye tracking [[paper](https://arxiv.org/abs/2404.12083)] (2024.04.18)
- MambaMOS: LiDAR-based 3D Moving Object Segmentation with Motion-aware State Space Model [[paper](https://arxiv.org/abs/2404.12794)] [[code](https://github.com/Terminal-K/MambaMOS)] (2024.04.19) ![Stars](https://img.shields.io/github/stars/Terminal-K/MambaMOS) 
- ST-SSMs: Spatial-Temporal Selective State of Space Model for Traffic Forecasting [[paper](https://arxiv.org/abs/2404.13257)] (2024.04.20)
- MambaUIE&SR: Unraveling the Ocean's Secrets with Only 2.8 FLOPs [[paper](https://arxiv.org/abs/2404.13884)] [[code](https://github.com/1024AILab/MambaUIE)] (2024.04.22) ![Stars](https://img.shields.io/github/stars/1024AILab/MambaUIE) 
- ST-MambaSync: The Confluence of Mamba Structure and Spatio-Temporal Transformers for Precipitous Traffic Prediction [[paper](https://arxiv.org/abs/2404.15899)] (2024.04.24)
- Spectral-Spatial Mamba for Hyperspectral Image Classification [[paper](https://arxiv.org/abs/2404.18401)] (2024.04.29)
- CLIP-Mamba: CLIP Pretrained Mamba Models with OOD and Hessian Evaluation [[paper](https://arxiv.org/abs/2404.19394)] [[code](https://github.com/raytrun/mamba-clip)] (2024.04.30) ![Stars](https://img.shields.io/github/stars/raytrun/mamba-clip) 
- TRAMBA: A Hybrid Transformer and Mamba Architecture for Practical Audio and Bone Conduction Speech Super Resolution and Enhancement on Mobile and Wearable Platforms [[paper](https://arxiv.org/abs/2405.01242)] (2024.05.02)
- SSUMamba: Spatial-Spectral Selective State Space Model for Hyperspectral Image Denoising [[paper](https://arxiv.org/abs/2405.01726)] [[code](https://github.com/lronkitty/SSUMamba)] (2024.05.02) ![Stars](https://img.shields.io/github/stars/lronkitty/SSUMamba) 
- FER-YOLO-Mamba: Facial Expression Detection and Classification Based on Selective State Space [[paper](https://arxiv.org/abs/2405.01828)] [[code](https://github.com/SwjtuMa/FER-YOLO-Mamba)] (2024.05.03) ![Stars](https://img.shields.io/github/stars/SwjtuMa/FER-YOLO-Mamba) 
- DVMSR: Distillated Vision Mamba for Efficient Super-Resolution [[paper](https://arxiv.org/abs/2405.03008)] [[code](https://github.com/nathan66666/DVMSR)] (2024.05.05) ![Stars](https://img.shields.io/github/stars/nathan66666/DVMSR) 
- Matten: Video Generation with Mamba-Attention [[paper](https://arxiv.org/abs/2405.03025)] (2024.05.05)
- MemoryMamba: Memory-Augmented State Space Model for Defect Recognition [[paper](https://arxiv.org/abs/2405.03673)] (2024.05.06)
- MambaJSCC: Deep Joint Source-Channel Coding with Visual State Space Model [[paper](https://arxiv.org/abs/2405.03125)] (2024.05.06)
- Retinexmamba: Retinex-based Mamba for Low-light Image Enhancement [[paper](https://arxiv.org/abs/2405.03349)] [[code](https://github.com/YhuoyuH/RetinexMamba)] (2024.05.06) ![Stars](https://img.shields.io/github/stars/YhuoyuH/RetinexMamba) 
- SMCD: High Realism Motion Style Transfer via Mamba-based Diffusion [[paper](https://arxiv.org/abs/2405.02844)] (2024.05.06)
- VMambaCC: A Visual State Space Model for Crowd Counting [[paper](https://arxiv.org/abs/2405.03978)] (2024.05.07)
- Frequency-Assisted Mamba for Remote Sensing Image Super-Resolution [[paper](https://arxiv.org/abs/2405.04964)] (2024.05.08)
- StyleMamba : State Space Model for Efficient Text-driven Image Style Transfer [[paper](https://arxiv.org/abs/2405.05027)] (2024.05.08)
- MambaOut: Do We Really Need Mamba for Vision? [[paper](https://arxiv.org/abs/2405.07992)] [[code](https://github.com/yuweihao/MambaOut)] (2024.05.13) ![Stars](https://img.shields.io/github/stars/yuweihao/MambaOut) 
- OverlapMamba: Novel Shift State Space Model for LiDAR-based Place Recognition [[paper](https://arxiv.org/abs/2405.07966)] [[code](https://github.com/SCNU-RISLAB/OverlapMamba)] (2024.05.13) ![Stars](https://img.shields.io/github/stars/SCNU-RISLAB/OverlapMamba) 
- GMSR:Gradient-Guided Mamba for Spectral Reconstruction from RGB Images [[paper](https://arxiv.org/abs/2405.07777)] [[code](https://github.com/wxy11-27/GMSR)] (2024.05.13) ![Stars](https://img.shields.io/github/stars/wxy11-27/GMSR) 
- WaterMamba: Visual State Space Model for Underwater Image Enhancement [[paper](https://arxiv.org/abs/2405.08419)] (2024.05.14)
- Rethinking Scanning Strategies with Vision Mamba in Semantic Segmentation of Remote Sensing Imagery: An Experimental Study [[paper](https://arxiv.org/abs/2405.08493)] (2024.05.14)
- Multiscale Global Attention for Abnormal Geological Hazard Segmentation [[paper](https://ieeexplore.ieee.org/abstract/document/10492495)] (2024.05.15)
- IRSRMamba: Infrared Image Super-Resolution via Mamba-based Wavelet Transform Feature Modulation Model [[paper](https://arxiv.org/abs/2405.09873)] (2024.05.16)
- CM-UNet: Hybrid CNN-Mamba UNet for Remote Sensing Image Semantic Segmentation [[paper](https://arxiv.org/abs/2405.10530)] [[code](https://github.com/XiaoBuL/CM-UNet)] (2024.05.17) ![Stars](https://img.shields.io/github/stars/XiaoBuL/CM-UNet) 
- NetMamba: Efficient Network Traffic Classification via Pre-training Unidirectional Mamba [[paper](https://arxiv.org/abs/2405.11449)] (2024.05.19)
- Mamba-in-Mamba: Centralized Mamba-Cross-Scan in Tokenized Mamba Model for Hyperspectral Image Classification [[paper](https://arxiv.org/abs/2405.12003)] (2024.05.20)
- 3DSS-Mamba: 3D-Spectral-Spatial Mamba for Hyperspectral Image Classification [[paper](https://arxiv.org/abs/2405.12487)] (2024.05.21)

<span id="head6"></span>

###  Language
* MambaByte: Token-free Selective State Space Model [[paper](https://arxiv.org/abs/2401.13660)] [[code](https://github.com/lucidrains/MEGABYTE-pytorch)] (2024.01.24) ![Stars](https://img.shields.io/github/stars/lucidrains/MEGABYTE-pytorch) 
* Is Mamba Capable of In-Context Learning? [[paper](https://arxiv.org/abs/2402.03170)] (2024.02.05)
* Can Mamba Learn How to Learn? A Comparative Study on In-Context Learning Tasks  [[paper](https://arxiv.org/abs/2402.04248)] (2024.02.06)
* SpaceByte: Towards Deleting Tokenization from Large Language Modeling [[paper](https://arxiv.org/abs/2404.14408)] (2024.04.22)
* State-Free Inference of State-Space Models: The Transfer Function Approach [[paper](https://arxiv.org/abs/2405.06147)] (2024.05.10)

<span id="head7"></span>
###  Multi-Modal
* VL-Mamba: Exploring State Space Models for Multimodal Learning [[paper](https://arxiv.org/abs/2403.13600)] [[code](https://github.com/ZhengYu518/VL-Mamba)] (2024.03.20) ![Stars](https://img.shields.io/github/stars/ZhengYu518/VL-Mamba) 
* Cobra: Extending Mamba to Multi-Modal Large Language Model for Efficient Inference [[paper](https://arxiv.org/abs/2403.14520)] [[code](https://sites.google.com/view/cobravlm)] (2024.03.21)
* SpikeMba: Multi-Modal Spiking Saliency Mamba for Temporal Video Grounding [[paper](https://arxiv.org/abs/2404.01174)] (2024.04.01)
* Sigma: Siamese Mamba Network for Multi-Modal Semantic Segmentation [[paper](https://arxiv.org/abs/2404.03425)] [[code](https://github.com/ChenHongruixuan/MambaCD)] (2024.04.04) ![Stars](https://img.shields.io/github/stars/ChenHongruixuan/MambaCD) 
* SurvMamba: State Space Model with Multi-grained Multi-modal Interaction for Survival Prediction [[paper](https://arxiv.org/abs/2404.08027)] (2024.04.11)
* MambaDFuse: A Mamba-based Dual-phase Model for Multi-modality Image Fusion [[paper](https://arxiv.org/abs/2404.08406)] (2024.04.12)
* Fusion-Mamba for Cross-modality Object Detection [[paper](https://arxiv.org/abs/2404.09146)] (2024.04.14)
* CFMW: Cross-modality Fusion Mamba for Multispectral Object Detection under Adverse Weather Conditions [[paper](https://arxiv.org/abs/2404.16302)] [[code](https://github.com/lhy-zjut/CFMW)] (2024.04.25) ![Stars](https://img.shields.io/github/stars/lhy-zjut/CFMW) 

<span id="head8"></span>
###  Medical
- U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation [[paper](https://arxiv.org/abs/2401.04722)] [[code](https://github.com/bowang-lab/U-Mamba)] [[dataset](https://drive.google.com/drive/folders/1DmyIye4Gc9wwaA7MVKFVi-bWD2qQb-qN?usp=sharing)] [[homepage](https://wanglab.ai/u-mamba.html)] (2024.01.09) ![Stars](https://img.shields.io/github/stars/bowang-lab/U-Mamba) 

- SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation [[paper](https://arxiv.org/abs/2401.13560)] [[code](https://github.com/ge-xing/SegMamba)] (2024.01.24) ![Stars](https://img.shields.io/github/stars/ge-xing/SegMamba) 

- MambaMorph: a Mamba-based Backbone with Contrastive Feature Learning for Deformable MR-CT Registration [[paper](https://arxiv.org/abs/2401.13934)] [[code](https://github.com/guo-stone/mambamorph)] (2024.01.24) ![Stars](https://img.shields.io/github/stars/guo-stone/mambamorph) 

- Vivim: a Video Vision Mamba for Medical Video Object Segmentation [[paper](https://arxiv.org/abs/2401.14168)] [[code](https://github.com/scott-yjyang/Vivim)] (2024.01.25) ![Stars](https://img.shields.io/github/stars/scott-yjyang/Vivim) 

- VM-UNet: Vision Mamba UNet for Medical Image Segmentation [[paper](https://arxiv.org/abs/2402.02491)] [[code](https://github.com/jcruan519/vm-unet)] (2024.02.04) ![Stars](https://img.shields.io/github/stars/jcruan519/vm-unet) 

- Swin-UMamba: Mamba-based UNet with ImageNet-based pretraining [[paper](https://arxiv.org/abs/2402.03302)] [[code](https://github.com/jiarunliu/swin-umamba)] (2024.02.05) ![Stars](https://img.shields.io/github/stars/jiarunliu/swin-umamba) 

- nnMamba: 3D Biomedical Image Segmentation, Classification and Landmark Detection with State Space Model [[paper](https://arxiv.org/abs/2402.03526)] [[code](https://github.com/lhaof/nnmamba)] (2024.02.05) ![Stars](https://img.shields.io/github/stars/lhaof/nnmamba) 

- Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation [[paper](https://arxiv.org/abs/2402.05079)] [[code](https://github.com/ziyangwang007/mamba-unet)] (2024.02.07) ![Stars](https://img.shields.io/github/stars/ziyangwang007/mamba-unet) 

- FD-Vision Mamba for Endoscopic Exposure Correction [[paper](https://arxiv.org/abs/2402.06378)] (2024.02.09)

- Semi-Mamba-UNet: Pixel-Level Contrastive Cross-Supervised Visual Mamba-based UNet for Semi-Supervised Medical Image Segmentation [[paper](https://arxiv.org/abs/2402.07245)] [[code](https://github.com/ziyangwang007/mamba-unet)] (2024.02.11) ![Stars](https://img.shields.io/github/stars/ziyangwang007/mamba-unet) 

- P-Mamba: Marrying Perona Malik Diffusion with Mamba for Efficient Pediatric Echocardiographic Left Ventricular Segmentation [[paper](https://arxiv.org/abs/2402.08506)] (2024.02.13)

- Weak-Mamba-UNet: Visual Mamba Makes CNN and ViT Work Better for Scribble-based Medical Image Segmentation [[paper](https://arxiv.org/abs/2402.10887)] [[code](https://github.com/ziyangwang007/mamba-unet)] (2024.02.16) ![Stars](https://img.shields.io/github/stars/ziyangwang007/mamba-unet) 

- MambaMIR: An Arbitrary-Masked Mamba for Joint Medical Image Reconstruction and Uncertainty Estimation [[paper](https://arxiv.org/abs/2402.18451)] (2024.02.28)

- A PTM-Aware Protein Language Model with Bidirectional Gated Mamba Blocks

  [[Paper](https://www.biorxiv.org/content/10.1101/2024.02.28.581983v1)] [[Huggingface](https://huggingface.co/ChatterjeeLab/PTM-Mamba)] [[code](https://github.com/programmablebio/ptm-mamba)] (2024.02.28) ![Stars](https://img.shields.io/github/stars/programmablebio/ptm-mamba) 

- MedMamba: Vision Mamba for Medical Image Classification [[paper](https://arxiv.org/abs/2403.03849)] [[code](https://github.com/YubiaoYue/MedMamba)] (2024.03.06) ![Stars](https://img.shields.io/github/stars/YubiaoYue/MedMamba) 

- Motion-Guided Dual-Camera Tracker for Low-Cost Skill Evaluation of Gastric Endoscopy [[paper](https://arxiv.org/abs/2403.05146)] (2024.03.08)

- MamMIL: Multiple Instance Learning for Whole Slide Images with State Space Models [[paper](https://arxiv.org/abs/2403.05160)] (2024.03.08)

- LightM-UNet: Mamba Assists in Lightweight UNet for Medical Image Segmentation [[paper](https://arxiv.org/abs/2403.05246)] [[code](https://github.com/MrBlankness/LightM-UNet)] (2024.03.08) ![Stars](https://img.shields.io/github/stars/MrBlankness/LightM-UNet) 

- ClinicalMamba: A Generative Clinical Language Model on Longitudinal Clinical Notes [[paper](https://arxiv.org/abs/2403.05795)] (2024.03.09)

- Large Window-based Mamba UNet for Medical Image Segmentation: Beyond Convolution and Self-attention [[paper](https://arxiv.org/abs/2403.07332)] [[code](https://github.com/wjh892521292/lma-unet)] (2024.03.12) ![Stars](https://img.shields.io/github/stars/wjh892521292/lma-unet) 

- MD-Dose: A Diffusion Model based on the Mamba for Radiotherapy Dose Prediction [[paper](https://arxiv.org/abs/2403.08479)] [[code](https://github.com/flj19951219/mamba_dose)] (2024.03.13) ![Stars](https://img.shields.io/github/stars/flj19951219/mamba_dose) 

- VM-UNET-V2 Rethinking Vision Mamba UNet for Medical Image Segmentation [[paper](https://arxiv.org/abs/2403.09157)] [[code](https://github.com/nobodyplayer1/vm-unetv2)] (2024.03.14) ![Stars](https://img.shields.io/github/stars/nobodyplayer1/vm-unetv2) 

- H-vmunet: High-order Vision Mamba UNet for Medical Image Segmentation [[paper](https://arxiv.org/abs/2403.13642)] [[code](https://github.com/wurenkai/H-vmunet)] (2024.03.20) ![Stars](https://img.shields.io/github/stars/wurenkai/H-vmunet) 

- ProMamba: Prompt-Mamba for polyp segmentation [[paper](https://arxiv.org/abs/2403.13660)] (2024.03.20)

- UltraLight VM-UNet: Parallel Vision Mamba Significantly Reduces Parameters for Skin Lesion Segmentation [[paper](https://arxiv.org/abs/2403.20035)] [[code](https://github.com/wurenkai/UltraLight-VM-UNet)] (2024.03.29) ![Stars](https://img.shields.io/github/stars/wurenkai/UltraLight-VM-UNet) 

- VMambaMorph: a Visual Mamba-based Framework with Cross-Scan Module for Deformable 3D Image Registration [[paper](https://arxiv.org/abs/2404.05105)] (2024.04.07)

- Vim4Path: Self-Supervised Vision Mamba for Histopathology Images [[paper](https://arxiv.org/abs/2404.13222)] [[code](https://github.com/AtlasAnalyticsLab/Vim4Path)] (2024.04.20) ![Stars](https://img.shields.io/github/stars/AtlasAnalyticsLab/Vim4Path) 

- Sparse Reconstruction of Optical Doppler Tomography Based on State Space Model [[paper](https://arxiv.org/abs/2404.17484)] (2024.04.26)

- AC-MAMBASEG: An adaptive convolution and Mamba-based architecture for enhanced skin lesion segmentation [[paper](https://arxiv.org/abs/2405.03011)] [[code](https://github.com/vietthanh2710/AC-MambaSeg)] (2024.05.05) ![Stars](https://img.shields.io/github/stars/vietthanh2710/AC-MambaSeg) 

- HC-Mamba: Vision MAMBA with Hybrid Convolutional Techniques for Medical Image Segmentation [[paper](https://arxiv.org/abs/2405.05007)] (2024.05.08)

- VM-DDPM: Vision Mamba Diffusion for Medical Image Synthesis [[paper](https://arxiv.org/abs/2405.05667)] (2024.05.09)

<span id="head9"></span>
### Tabular Data
* MambaTab: A Simple Yet Effective Approach for Handling Tabular Data [[paper](https://arxiv.org/abs/2401.08867)] (2024.01.16)

<span id="head10"></span>
###  Graph
* Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces [[paper](https://arxiv.org/abs/2402.00789)] [[code](https://github.com/bowang-lab/Graph-Mamba)] (2024.02.01) ![Stars](https://img.shields.io/github/stars/bowang-lab/Graph-Mamba) 
* Graph Mamba: Towards Learning on Graphs with State Space Models [[paper](https://arxiv.org/abs/2402.08678)] [[code](https://github.com/graphmamba/gmn)] (2024.02.13) ![Stars](https://img.shields.io/github/stars/graphmamba/gmn) 
* STG-Mamba: Spatial-Temporal Graph Learning via Selective State Space Model [[paper](https://arxiv.org/abs/2403.12418)] (2024.03.19)

<span id="head11"></span>
### Point Cloud
* PointMamba: A Simple State Space Model for Point Cloud Analysis [[paper](https://arxiv.org/abs/2402.10739)] (2024.02.16)
* Point Could Mamba: Point Cloud Learning via State Space Model [[paper](https://arxiv.org/abs/2403.00762)] [[code](https://github.com/zhang-tao-whu/pcm)] (2024.03.01) ![Stars](https://img.shields.io/github/stars/zhang-tao-whu/pcm) 
* 3DMambaIPF: A State Space Model for Iterative Point Cloud Filtering via Differentiable Rendering [[paper](https://arxiv.org/abs/2404.05522)] (2024.04.08)
* 3DMambaComplete: Exploring Structured State Space Model for Point Cloud Completion [[paper](https://arxiv.org/abs/2404.07106)] (2024.04.10)
* Mamba3D: Enhancing Local Features for 3D Point Cloud Analysis via State Space Model [[paper](https://arxiv.org/abs/2404.14966)] [[code](https://github.com/xhanxu/Mamba3D)] (2024.04.23) ![Stars](https://img.shields.io/github/stars/xhanxu/Mamba3D) 

<span id="head12"></span>
### Time Series
* Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling [[paper](https://arxiv.org/abs/2402.10211)] [[code](https://github.com/raunaqbhirangi/hiss)] [[homepage](https://hiss-csp.github.io/)] (2024.02.15) ![Stars](https://img.shields.io/github/stars/raunaqbhirangi/hiss) 
* MambaStock: Selective state space model for stock prediction [[paper](https://arxiv.org/abs/2402.18959)] [[code](https://github.com/zshicode/MambaStock)] (2024.02.29) ![Stars](https://img.shields.io/github/stars/zshicode/MambaStock) 
* MambaLithium: Selective state space model for remaining-useful-life, state-of-health, and state-of-charge estimation of lithium-ion batteries [[paper](https://arxiv.org/abs/2403.05430)] [[code](https://github.com/zshicode/MambaLithium)] (2024.03.08) ![Stars](https://img.shields.io/github/stars/zshicode/MambaLithium) 
* TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting [[paper](https://arxiv.org/abs/2403.09898)] [[code](https://github.com/atik-ahamed/timemachine)] (2024.03.14) ![Stars](https://img.shields.io/github/stars/atik-ahamed/timemachine) 
* Is Mamba Effective for Time Series Forecasting? [[paper](https://arxiv.org/abs/2403.11144)] [[code](https://github.com/wzhwzhwzh0921/S-D-Mamba)] (2024.03.17) ![Stars](https://img.shields.io/github/stars/wzhwzhwzh0921/S-D-Mamba) 
* SiMBA: Simplified Mamba-Based Architecture for Vision and Multivariate Time series [[paper](https://arxiv.org/abs/2403.15360)] (2024.03.22)
* MambaMixer: Efficient Selective State Space Models with Dual Token and Channel Selection [[paper](https://arxiv.org/abs/2403.19888)] [[project](https://mambamixer.github.io/)] (2024.03.29)
* HARMamba: Efficient Wearable Sensor Human Activity Recognition Based on Bidirectional Selective SSM [[paper](https://arxiv.org/abs/2403.20183)] (2024.03.29)
* Integrating Mamba and Transformer for Long-Short Range Time Series Forecasting [[paper](https://arxiv.org/abs/2404.14757)] [[code](https://github.com/XiongxiaoXu/Mambaformerin-Time-Series)] (2024.04.23) ![Stars](https://img.shields.io/github/stars/XiongxiaoXu/Mambaformerin-Time-Series) 
* Bi-Mamba4TS: Bidirectional Mamba for Time Series Forecasting [[paper](https://arxiv.org/abs/2404.15772)] [[code](https://github.com/davidwynter/Bi-Mamba4TS)] (2024.04.24) ![Stars](https://img.shields.io/github/stars/davidwynter/Bi-Mamba4TS) 
* MAMCA -- Optimal on Accuracy and Efficiency for Automatic Modulation Classification with Extended Signal Length [[paper](https://arxiv.org/abs/2405.11263)] [[code](https://github.com/ZhangYezhuo/MAMCA)] (2024.05.18) ![Stars](https://img.shields.io/github/stars/ZhangYezhuo/MAMCA) 

<span id="head13"></span>

###  Speech
* Multichannel Long-Term Streaming Neural Speech Enhancement for Static and Moving Speakers [[paper](https://arxiv.org/abs/2403.07675)] [[code](https://github.com/Audio-WestlakeU/NBSS)] (2024.03.12) ![Stars](https://img.shields.io/github/stars/Audio-WestlakeU/NBSS) 
* Dual-path Mamba: Short and Long-term Bidirectional Selective Structured State Space Models for Speech Separation [[paper](https://arxiv.org/abs/2403.18257)] (2024.03.27)
* Multichannel Long-Term Streaming Neural Speech Enhancement for Static and Moving Speakers [[paper](https://arxiv.org/abs/2403.18276)] [[code](https://github.com/zhichaoxu-shufe/RankMamba)] (2024.03.27) ![Stars](https://img.shields.io/github/stars/zhichaoxu-shufe/RankMamba) 
* SPMamba: State-space model is all you need in speech separation [[paper](https://arxiv.org/abs/2404.02063)] [[code](https://github.com/JusperLee/SPMamba)] (2024.04.02) ![Stars](https://img.shields.io/github/stars/JusperLee/SPMamba) 
* An Investigation of Incorporating Mamba for Speech Enhancement [[paper](https://arxiv.org/abs/2405.06573)] (2024.05.10)
* SSAMBA: Self-Supervised Audio Representation Learning with Mamba State Space Model [[paper](https://arxiv.org/abs/2405.11831)] (2024.05.20)
* Mamba in Speech: Towards an Alternative to Self-Attention [[paper](https://arxiv.org/abs/2405.12609)] (2024.05.21)

<span id="head14"></span>
### Recommendation 
* Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models [[paper](https://arxiv.org/abs/2403.05430)] [[code](https://github.com/chengkai-liu/mamba4rec)] (2024.03.06) ![Stars](https://img.shields.io/github/stars/chengkai-liu/mamba4rec) 
* Uncovering Selective State Space Model's Capabilities in Lifelong Sequential Recommendation [[paper](https://arxiv.org/abs/2403.16371)] [[code](https://github.com/nancheng58/Rec-Mamba)] (2024.03.25) ![Stars](https://img.shields.io/github/stars/nancheng58/Rec-Mamba) 

<span id="head15"></span>
### Reinforcement Learning
* Decision Mamba: Reinforcement Learning via Sequence Modeling with Selective State Spaces [[paper](https://arxiv.org/abs/2403.19925)] [[code](https://github.com/toshihiro-ota/decision-mamba)] (2024.03.29) ![Stars](https://img.shields.io/github/stars/toshihiro-ota/decision-mamba) 
* Hierarchical Decision Mamba [[paper](https://arxiv.org/abs/2405.07943)] [[code](https://github.com/meowatthemoon/HierarchicalDecisionMamba)] (2024.05.13) ![Stars](https://img.shields.io/github/stars/meowatthemoon/HierarchicalDecisionMamba) 
* Is Mamba Compatible with Trajectory Optimization in Offline Reinforcement Learning? [[paper](https://arxiv.org/abs/2405.12094)] (2024.05.20)

<span id="head16"></span>
###  Survey
* State Space Model for New-Generation Network Alternative to Transformers: A Survey [[paper](https://arxiv.org/abs/2404.09516)] [[project](https://github.com/Event-AHU/Mamba_State_Space_Model_Paper_List)] (2024.04.15)
* Mamba-360: Survey of State Space Models as Transformer Alternative for Long Sequence Modelling: Methods, Applications, and Challenges [[paper](https://arxiv.org/abs/2404.16112)] [[project](https://github.com/badripatro/mamba360)] (2024.04.24)
* A Survey on Visual Mamba [[paper](https://arxiv.org/abs/2404.15956)] (2024.04.24)

<span id="head17"></span>
##  Tutorials
<span id="head18"></span>
###  Blogs
* The Annotated S4 [[URL](https://srush.github.io/annotated-s4/#part-1b-addressing-long-range-dependencies-with-hippo)]
* The Annotated Mamba [[URL](https://srush.github.io/annotated-mamba/hard.html#part-1-cumulative-sums)]
* A Visual Guide to Mamba and State Space Models [[URL](https://maartengrootendorst.substack.com/p/a-visual-guide-to-mamba-and-state)]
* Mamba No. 5 (A Little Bit Of...) [[URL](https://jameschen.io/jekyll/update/2024/02/12/mamba.html)]

<span id="head19"></span>
###  Videos
* S4: Efficiently Modeling Long Sequences with Structured State Spaces | Albert Gu [[URL](https://www.youtube.com/watch?v=luCBXCErkCs)]
* Mamba and S4 Explained: Architecture, Parallel Scan, Kernel Fusion, Recurrent, Convolution, Math [[URL](https://www.youtube.com/watch?v=8Q_tqwpTpVU)]
* MAMBA from Scratch: Neural Nets Better and Faster than Transformers [[URL](https://www.youtube.com/watch?v=N6Piou4oYx8)]

<span id="head20"></span>
###  Books
* Linear State‐Space Control Systems [[URL](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470117873)]
* Modeling sequences with structured state spaces [[URL](https://searchworks.stanford.edu/view/14784021)]

<span id="head21"></span>
###  Codes
* The official Mamba Repository is currently only available for Linux. [[URL](https://github.com/state-spaces/mamba)]
* If you are searching for a runnable implementation not focused on speed,
  * mamba-minimal: Simple, minimal implementation of the Mamba SSM in one file of PyTorch. [[URL](https://github.com/johnma2006/mamba-minimal/tree/master)] 
  * mamba.py: An efficient Mamba implementation in PyTorch and MLX. [[URL](https://github.com/alxndrTL/mamba.py)]

<span id="head22"></span>
### Other Awesome Mamba List
* awesome-ssm-ml [[URL](https://github.com/AvivBick/awesome-ssm-ml)]
* Awesome-Mamba: Collect papers about Mamba [[URL](https://github.com/caojiaolong/Awesome-Mamba)]
* Awesome-Mamba-Papers [[URL](https://github.com/yyyujintang/Awesome-Mamba-Papers)]

<span id="head23"></span>
##  Contributions
🎉 Thank you for considering contributing to our Awesome Mamba Collection repository! 🚀

<span id="head24"></span>
### Contribute in 3 Steps
1. **Fork the Repo:** Fork this repo to your GitHub account.
2. **Edit Content:** Contribute by adding new resources or improving existing content in the `README.md` file.
3. **Create a Pull Request:** Open a pull request (PR) from your branch to the main repository.

<span id="head25"></span>
###  Guidelines
- Follow the existing structure and formatting.
- Ensure added resources are relevant to State Space Models in Machine Learning.
- Verify that links work correctly.

<span id="head26"></span>
##  Acknowledgement
Thanks the template from [Awesome-Visual-Transformer](https://github.com/dk-liang/Awesome-Visual-Transformer) and [Awesome State-Space Resources for ML](https://github.com/AvivBick/awesome-ssm-ml/tree/main)

