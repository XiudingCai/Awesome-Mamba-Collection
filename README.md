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
  - [ Diffusion](#head27)
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
* Mamba: Linear-Time Sequence Modeling with Selective State Spaces [[paper](https://arxiv.org/abs/2312.00752)] [[code](https://github.com/state-spaces/mamba)] (2023.12.01)
* MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts [[paper](https://arxiv.org/abs/2401.04081)] [[code](https://github.com/llm-random/llm-random)] (2024.01.08)
* BlackMamba: Mixture of Experts for State-Space Models [[paper](https://arxiv.org/abs/2402.01771)] [[code](https://github.com/zyphra/blackmamba)] (2024.02.01)
* Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling [[paper](https://arxiv.org/abs/2402.10211)] [[homepage](https://hiss-csp.github.io/)] (2024.02.15)
* DenseMamba: State Space Models with Dense Hidden Connection for Efficient Large Language Models [[paper](https://arxiv.org/abs/2403.00818)] [[code](https://github.com/WailordHe/DenseSSM)] (2024.02.26)
* Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models [[paper](https://arxiv.org/abs/2402.19427)] [[code](https://github.com/kyegomez/Griffin)] (2024.02.29)
* Jamba: A Hybrid Transformer-Mamba Language Model [[paper](https://arxiv.org/abs/2403.19887)] [[code](https://github.com/kyegomez/Jamba)] (2024.03.28)
* (ICML 2024, Mamba-2) Transformers are SSMs: Generalized Models and Efficient Algorithms with Structured State Space Duality [[paper](https://arxiv.org/abs/2405.21060)] [[poster](https://icml.cc/virtual/2024/poster/32613)] [[code](https://github.com/state-spaces/mamba)] (2024.05.31)

<span id="head4"></span>
### Theoretical Analysis
* StableSSM: Alleviating the Curse of Memory in State-space Models through Stable Reparameterization [[paper](https://arxiv.org/abs/2311.14495)] (2023.11.24)
* The Hidden Attention of Mamba Models [[paper](https://arxiv.org/abs/2403.01590)] [[code](https://github.com/AmeenAli/HiddenMambaAttn)] (2024.03.03)
* Understanding Robustness of Visual State Space Models for Image Classification [[paper](https://arxiv.org/abs/2403.10935)] (2024.03.16)
* From Generalization Analysis to Optimization Designs for State Space Models [[paper](https://arxiv.org/abs/2405.02670)] (2024.05.04)
* Demystify Mamba in Vision: A Linear Attention Perspective [[paper](https://arxiv.org/abs/2405.16605)] [[code](https://github.com/LeapLabTHU/MLLA)] (2024.05.26)
* A Unified Implicit Attention Formulation for Gated-Linear Recurrent Sequence Models [[paper](https://arxiv.org/abs/2405.16504)] (2024.05.26)
* The Expressive Capacity of State Space Models: A Formal Language Perspective [[paper](https://arxiv.org/abs/2405.17394)] (2024.05.27)
* Unlocking the Secrets of Linear Complexity Sequence Model from A Unified Perspective [[paper](https://arxiv.org/abs/2405.17383)] (2024.05.27)
* (ICML 2024) Transformers are SSMs: Generalized Models and Efficient Algorithms with Structured State Space Duality [[paper](https://arxiv.org/abs/2405.21060)] [[poster](https://icml.cc/virtual/2024/poster/32613)] [[code](https://github.com/state-spaces/mamba)] (2024.05.31)
* MambaLRP: Explaining Selective State Space Sequence Models [[paper](https://arxiv.org/abs/2406.07592)] (2024.06.11)
* Towards a theory of learning dynamics in deep state space models [[paper](https://arxiv.org/abs/2407.07279)] (2024.07.10)

<span id="head5"></span>
###  Vision
- (ICML 2024) Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model [[paper](https://arxiv.org/abs/2401.09417)] [[code](https://github.com/hustvl/Vim)] (2024.01.17)
- VMamba: Visual State Space Model [[paper](https://arxiv.org/abs/2401.10166)] [[code](https://github.com/MzeroMiko/VMamba)] (2024.01.18)
- U-shaped Vision Mamba for Single Image Dehazing [[paper](https://arxiv.org/abs/2402.04139)] (2024.02.06)
- Scalable Diffusion Models with State Space Backbone [[paper](https://arxiv.org/abs/2402.05608)] [[code](https://github.com/feizc/dis)] (2024.02.08)
- Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data [[paper](https://arxiv.org/abs/2402.05892)] (2024.02.08)
- Pan-Mamba: Effective pan-sharpening with State Space Model [[paper](https://arxiv.org/abs/2402.12192)] (2024.02.19)
- (ECCV 2024) MambaIR: A Simple Baseline for Image Restoration with State-Space Model [[paper](https://arxiv.org/abs/2402.15648)] [[code](https://github.com/csguoh/mambair)] (2024.02.23)
- Res-VMamba: Fine-Grained Food Category Visual Classification Using Selective State Space Models with Deep Residual Learning [[paper](https://arxiv.org/abs/2402.15761)] [[code](https://github.com/chishengchen/resvmamba)] (2024.02.24)
- MiM-ISTD: Mamba-in-Mamba for Efficient Infrared Small Target Detection [[paper](https://arxiv.org/abs/2403.02148)] [[code](https://github.com/txchen-USTC/MiM-ISTD)] (2024.03.04)
- Motion Mamba: Efficient and Long Sequence Motion Generation with Hierarchical and Bidirectional Selective SSM [[paper](https://arxiv.org/abs/2403.07487)] [[code](https://github.com/steve-zeyu-zhang/MotionMamba)] (2024.03.12)
- Video Mamba Suite: State Space Model as a Versatile Alternative for Video Understanding [[paper](https://arxiv.org/abs/2403.09626)] [[code](https://github.com/opengvlab/video-mamba-suite)] (2024.03.12)
- LocalMamba: Visual State Space Model with Windowed Selective Scan [[paper](https://arxiv.org/abs/2403.09338)] [[code](https://github.com/hunto/LocalMamba)] (2024.03.14)
- EfficientVMamba: Atrous Selective Scan for Light Weight Visual Mamba [[paper](https://arxiv.org/abs/2403.09977)] (2024.03.15)
- VmambaIR: Visual State Space Model for Image Restoration [[paper](https://arxiv.org/abs/2403.11423)] [[code](https://github.com/alphacatplus/vmambair)] (2024.03.18)
- (ECCV 2024) ZigMa: Zigzag Mamba Diffusion Model [[paper](https://arxiv.org/abs/2403.13802)] [[code](https://github.com/CompVis/zigma)] [[project](https://taohu.me/zigma/)] (2024.03.20)
- SiMBA: Simplified Mamba-Based Architecture for Vision and Multivariate Time series [[paper](https://arxiv.org/abs/2403.15360)] (2024.03.22)
- VMRNN: Integrating Vision Mamba and LSTM for Efficient and Accurate Spatiotemporal Forecasting [[paper](https://arxiv.org/abs/2403.16536)] [[code](https://github.com/yyyujintang/VMRNN-PyTorch)] (2024.03.25) (CVPR24 Precognition Workshop)
- PlainMamba: Improving Non-Hierarchical Mamba in Visual Recognition [[paper](https://arxiv.org/abs/2403.17695)] [[code](https://github.com/ChenhongyiYang/PlainMamba)] (2024.03.26)
- ReMamber: Referring Image Segmentation with Mamba Twister [[paper](https://arxiv.org/abs/2403.17839)] (2024.03.26)
- Gamba: Marry Gaussian Splatting with Mamba for single view 3D reconstruction [[paper](https://arxiv.org/abs/2403.18795)] (2024.03.27)
- RSMamba: Remote Sensing Image Classification with State Space Model [[paper](https://arxiv.org/abs/2403.19654)] [[code](https://github.com/KyanChen/RSMamba)] (2024.03.28)
- MambaMixer: Efficient Selective State Space Models with Dual Token and Channel Selection [[paper](https://arxiv.org/abs/2403.19888)] (2024.03.29)
- HSIMamba: Hyperpsectral Imaging Efficient Feature Learning with Bidirectional State Space for Classification [[paper](https://arxiv.org/abs/2404.00272)] (2024.03.30)
- Samba: Semantic Segmentation of Remotely Sensed Images with State Space Model [[paper](https://arxiv.org/abs/2404.01705)] [[code](https://github.com/zhuqinfeng1999/Samba)] (2024.04.02)
- RS-Mamba for Large Remote Sensing Image Dense Prediction [[paper](https://arxiv.org/abs/2404.02668)] [[code](https://github.com/walking-shadow/Official_Remote_Sensing_Mamba)] (2024.04.03)
- RS3Mamba: Visual State Space Model for Remote Sensing Images Semantic Segmentation [[paper](https://arxiv.org/abs/2404.02457)] (2024.04.03)
- InsectMamba: Insect Pest Classification with State Space Model [[paper](https://arxiv.org/abs/2404.03611)] (2024.04.04)
- (IEEE TGRS 2024) ChangeMamba: Remote Sensing Change Detection with Spatio-Temporal State Space Model [[paper](https://arxiv.org/abs/2404.03425)] [[code](https://github.com/ChenHongruixuan/MambaCD)] (2024.04.04)
- RhythmMamba: Fast Remote Physiological Measurement with Arbitrary Length Videos [[paper](https://arxiv.org/abs/2404.06483)] [[code](https://github.com/zizheng-guo/RhythmMamba)] (2024.04.09)
- Simba: Mamba augmented U-ShiftGCN for Skeletal Action Recognition in Videos [[paper](https://arxiv.org/abs/2404.07645)] (2024.04.11)
- FusionMamba: Efficient Image Fusion with State Space Model [[paper](https://arxiv.org/abs/2404.07932)] (2024.04.11)
- DGMamba: Domain Generalization via Generalized State Space Model [[paper](https://arxiv.org/abs/2404.07794)] [[code](https://github.com/longshaocong/DGMamba)] (2024.04.11)
- SpectralMamba: Efficient Mamba for Hyperspectral Image Classification [[paper](https://arxiv.org/abs/2404.08489)] (2024.04.12)
- FreqMamba: Viewing Mamba from a Frequency Perspective for Image Deraining [[paper](https://arxiv.org/abs/2404.09476)] (2024.04.15)
- HSIDMamba: Exploring Bidirectional State-Space Models for Hyperspectral Denoising [[paper](https://arxiv.org/abs/2404.09697)] (2024.04.15)
- A Novel State Space Model with Local Enhancement and State Sharing for Image Fusion [[paper](https://arxiv.org/abs/2404.09293)] (2024.04.15)
- FusionMamba: Dynamic Feature Enhancement for Multimodal Image Fusion with Mamba [[paper](https://arxiv.org/abs/2404.09498)] [[code](https://github.com/millieXie/FusionMamba)] (2024.04.15)
- MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection [[paper](https://arxiv.org/abs/2404.06564)] [[project](https://lewandofskee.github.io/projects/MambaAD/)] [[code](https://github.com/lewandofskee/MambaAD)] (2024.04.15)
- Text-controlled Motion Mamba: Text-Instructed Temporal Grounding of Human Motion [[paper](https://arxiv.org/abs/2404.11375)] (2024.04.17)
- CU-Mamba: Selective State Space Models with Channel Learning for Image Restoration [[paper](https://arxiv.org/abs/2404.11778)] (2024.04.17)
- MambaPupil: Bidirectional Selective Recurrent model for Event-based Eye tracking [[paper](https://arxiv.org/abs/2404.12083)] (2024.04.18)
- MambaMOS: LiDAR-based 3D Moving Object Segmentation with Motion-aware State Space Model [[paper](https://arxiv.org/abs/2404.12794)] [[code](https://github.com/Terminal-K/MambaMOS)] (2024.04.19)
- ST-SSMs: Spatial-Temporal Selective State of Space Model for Traffic Forecasting [[paper](https://arxiv.org/abs/2404.13257)] (2024.04.20)
- MambaUIE&SR: Unraveling the Ocean's Secrets with Only 2.8 FLOPs [[paper](https://arxiv.org/abs/2404.13884)] [[code](https://github.com/1024AILab/MambaUIE)] (2024.04.22)
- ST-MambaSync: The Confluence of Mamba Structure and Spatio-Temporal Transformers for Precipitous Traffic Prediction [[paper](https://arxiv.org/abs/2404.15899)] (2024.04.24)
- Spectral-Spatial Mamba for Hyperspectral Image Classification [[paper](https://arxiv.org/abs/2404.18401)] (2024.04.29)
- CLIP-Mamba: CLIP Pretrained Mamba Models with OOD and Hessian Evaluation [[paper](https://arxiv.org/abs/2404.19394)] [[code](https://github.com/raytrun/mamba-clip)] (2024.04.30)
- SSUMamba: Spatial-Spectral Selective State Space Model for Hyperspectral Image Denoising [[paper](https://arxiv.org/abs/2405.01726)] [[code](https://github.com/lronkitty/SSUMamba)] (2024.05.02)
- FER-YOLO-Mamba: Facial Expression Detection and Classification Based on Selective State Space [[paper](https://arxiv.org/abs/2405.01828)] [[code](https://github.com/SwjtuMa/FER-YOLO-Mamba)] (2024.05.03)
- DVMSR: Distillated Vision Mamba for Efficient Super-Resolution [[paper](https://arxiv.org/abs/2405.03008)] [[code](https://github.com/nathan66666/DVMSR)] (2024.05.05)
- Matten: Video Generation with Mamba-Attention [[paper](https://arxiv.org/abs/2405.03025)] (2024.05.05)
- MemoryMamba: Memory-Augmented State Space Model for Defect Recognition [[paper](https://arxiv.org/abs/2405.03673)] (2024.05.06)
- MambaJSCC: Deep Joint Source-Channel Coding with Visual State Space Model [[paper](https://arxiv.org/abs/2405.03125)] (2024.05.06)
- Retinexmamba: Retinex-based Mamba for Low-light Image Enhancement [[paper](https://arxiv.org/abs/2405.03349)] [[code](https://github.com/YhuoyuH/RetinexMamba)] (2024.05.06)
- SMCD: High Realism Motion Style Transfer via Mamba-based Diffusion [[paper](https://arxiv.org/abs/2405.02844)] (2024.05.06)
- VMambaCC: A Visual State Space Model for Crowd Counting [[paper](https://arxiv.org/abs/2405.03978)] (2024.05.07)
- Frequency-Assisted Mamba for Remote Sensing Image Super-Resolution [[paper](https://arxiv.org/abs/2405.04964)] (2024.05.08)
- StyleMamba : State Space Model for Efficient Text-driven Image Style Transfer [[paper](https://arxiv.org/abs/2405.05027)] (2024.05.08)
- MambaOut: Do We Really Need Mamba for Vision? [[paper](https://arxiv.org/abs/2405.07992)] [[code](https://github.com/yuweihao/MambaOut)] (2024.05.13)
- OverlapMamba: Novel Shift State Space Model for LiDAR-based Place Recognition [[paper](https://arxiv.org/abs/2405.07966)] [[code](https://github.com/SCNU-RISLAB/OverlapMamba)] (2024.05.13)
- GMSR:Gradient-Guided Mamba for Spectral Reconstruction from RGB Images [[paper](https://arxiv.org/abs/2405.07777)] [[code](https://github.com/wxy11-27/GMSR)] (2024.05.13)
- WaterMamba: Visual State Space Model for Underwater Image Enhancement [[paper](https://arxiv.org/abs/2405.08419)] (2024.05.14)
- Rethinking Scanning Strategies with Vision Mamba in Semantic Segmentation of Remote Sensing Imagery: An Experimental Study [[paper](https://arxiv.org/abs/2405.08493)] (2024.05.14)
- Multiscale Global Attention for Abnormal Geological Hazard Segmentation [[paper](https://ieeexplore.ieee.org/abstract/document/10492495)] (2024.05.15)
- IRSRMamba: Infrared Image Super-Resolution via Mamba-based Wavelet Transform Feature Modulation Model [[paper](https://arxiv.org/abs/2405.09873)] (2024.05.16)
- RSDehamba: Lightweight Vision Mamba for Remote Sensing Satellite Image Dehazing [[paper](https://arxiv.org/abs/2405.10030)] (2024.05.16)
- CM-UNet: Hybrid CNN-Mamba UNet for Remote Sensing Image Semantic Segmentation [[paper](https://arxiv.org/abs/2405.10530)] [[code](https://github.com/XiaoBuL/CM-UNet)] (2024.05.17)
- NetMamba: Efficient Network Traffic Classification via Pre-training Unidirectional Mamba [[paper](https://arxiv.org/abs/2405.11449)] (2024.05.19)
- Mamba-in-Mamba: Centralized Mamba-Cross-Scan in Tokenized Mamba Model for Hyperspectral Image Classification [[paper](https://arxiv.org/abs/2405.12003)] (2024.05.20)
- 3DSS-Mamba: 3D-Spectral-Spatial Mamba for Hyperspectral Image Classification [[paper](https://arxiv.org/abs/2405.12487)] (2024.05.21)
- Multi-Scale VMamba: Hierarchy in Hierarchy Visual State Space Model [[paper](https://arxiv.org/abs/2405.14174)] [[code](https://github.com/YuHengsss/MSVMamba)] (2024.05.23)
- Efficient Visual State Space Model for Image Deblurring [[paper](https://arxiv.org/abs/2405.14343)] (2024.05.23)
- Scalable Visual State Space Model with Fractal Scanning [[paper](https://arxiv.org/abs/2405.14480)] (2024.05.23)
- DiM: Diffusion Mamba for Efficient High-Resolution Image Synthesis [[paper](https://arxiv.org/abs/2405.14224)] (2024.05.23)
- Mamba-R: Vision Mamba ALSO Needs Registers [[paper](https://arxiv.org/abs/2405.14858)] [[code](https://github.com/wangf3014/Mamba-Reg)] [[project](https://wangf3014.github.io/mambar-page/)] (2024.05.23)
- MambaVC: Learned Visual Compression with Selective State Spaces [[paper](https://arxiv.org/abs/2405.15413)] [[code](https://github.com/QinSY123/2024-MambaVC)] (2024.05.24)
- Scaling Diffusion Mamba with Bidirectional SSMs for Efficient Image and Video Generation [[paper](https://arxiv.org/abs/2405.15881)] (2024.05.24)
- MambaLLIE: Implicit Retinex-Aware Low Light Enhancement with Global-then-Local State Space [[paper](https://arxiv.org/abs/2405.16105)] (2024.05.25)
- Image Deraining with Frequency-Enhanced State Space Model [[paper](https://arxiv.org/abs/2405.16470)] (2024.05.26)
- Demystify Mamba in Vision: A Linear Attention Perspective [[paper](https://arxiv.org/abs/2405.16605)] [[code](https://github.com/LeapLabTHU/MLLA)] (2024.05.26)
- Vim-F: Visual State Space Model Benefiting from Learning in the Frequency Domain [[paper](https://arxiv.org/abs/2405.18679)] [[code](https://github.com/yws-wxs/Vim-F)] (2024.05.29)
- FourierMamba: Fourier Learning Integration with State Space Models for Image Deraining [[paper](https://arxiv.org/abs/2405.19450)] (2024.05.29)
- DeMamba: AI-Generated Video Detection on Million-Scale GenVideo Benchmark [[paper](https://arxiv.org/abs/2405.19707)] [[code](https://github.com/chenhaoxing/DeMamba)] (2024.05.30)
- Dual Hyperspectral Mamba for Efficient Spectral Compressive Imaging [[paper](https://arxiv.org/abs/2406.00449)] (2024.06.01)
- LLEMamba: Low-Light Enhancement via Relighting-Guided Mamba with Deep Unfolding Network [[paper](https://arxiv.org/abs/2406.01028)] (2024.06.03)
- GrootVL: Tree Topology is All You Need in State Space Model [[paper](https://arxiv.org/abs/2406.02395)] [[code](https://github.com/EasonXiao-888/GrootVL)] (2024.06.04)
- Feasibility of State Space Models for Network Traffic Generation [[paper](https://arxiv.org/abs/2406.02784)] (2024.06.04)
- Rethinking Spiking Neural Networks as State Space Models [[paper](https://arxiv.org/abs/2406.02923)] (2024.06.05)
- Learning 1D Causal Visual Representation with De-focus Attention Networks [[paper](https://arxiv.org/abs/2406.04342)] (2024.06.06)
- MambaDepth: Enhancing Long-range Dependency for Self-Supervised Fine-Structured Monocular Depth Estimation [[paper](https://arxiv.org/abs/2406.04532)] (2024.06.06)
- Efficient 3D Shape Generation via Diffusion Mamba with Bidirectional SSMs [[paper](https://arxiv.org/abs/2406.05038)] (2024.06.07)
- Mamba YOLO: SSMs-Based YOLO For Object Detection [[paper](https://arxiv.org/abs/2406.05835)] [[code](https://github.com/HZAI-ZJNU/Mamba-YOLO)] (2024.06.09)
- HDMba: Hyperspectral Remote Sensing Imagery Dehazing with State Space Model [[paper](https://arxiv.org/abs/2406.05700)] [[code](https://github.com/RsAI-lab/HDMba)] (2024.06.09)
- MVGamba: Unify 3D Content Generation as State Space Sequence Modeling [[paper](https://arxiv.org/abs/2406.06367)] (2024.06.10)
- MHS-VM: Multi-Head Scanning in Parallel Subspaces for Vision Mamba [[paper](https://arxiv.org/abs/2406.05992)] (2024.06.10)
- DualMamba: A Lightweight Spectral-Spatial Mamba-Convolution Network for Hyperspectral Image Classification [[paper](https://arxiv.org/abs/2406.07050)] (2024.06.11)
- Autoregressive Pretraining with Mamba in Vision [[paper](https://arxiv.org/abs/2406.07537)] [[code](https://github.com/OliverRensu/ARM)] (2024.06.11)
- PixMamba: Leveraging State Space Models in a Dual-Level Architecture for Underwater Image Enhancement [[paper](https://arxiv.org/abs/2406.08444)] [[code](https://github.com/weitunglin/pixmamba)] (2024.06.12)
- PyramidMamba: Rethinking Pyramid Feature Fusion with Selective Space State Model for Semantic Segmentation of Remote Sensing Imagery [[paper](https://arxiv.org/abs/2406.10828)] (2024.06.16)
- LFMamba: Light Field Image Super-Resolution with State Space Model [[paper](https://arxiv.org/abs/2406.12463)] (2024.06.18)
- Slot State Space Models [[paper](https://arxiv.org/abs/2406.12272)] (2024.06.18)
- SEDMamba: Enhancing Selective State Space Modelling with Bottleneck Mechanism and Fine-to-Coarse Temporal Fusion for Efficient Error Detection in Robot-Assisted Surgery [[paper](https://arxiv.org/abs/2406.15920)] (2024.06.22)
- Soft Masked Mamba Diffusion Model for CT to MRI Conversion [[paper](https://arxiv.org/abs/2406.17815)] [[code](https://github.com/wongzbb/DiffMa-Diffusion-Mamba)] (2024.06.22)
- SUM: Saliency Unification through Mamba for Visual Attention Modeling [[paper](https://arxiv.org/abs/2406.17815)] [[code](https://github.com/Arhosseini77/SUM)] (2024.06.25)
- VideoMambaPro: A Leap Forward for Mamba in Video Understanding [[paper](https://arxiv.org/abs/2406.19006)] (2024.06.27)
- MTMamba: Enhancing Multi-Task Dense Scene Understanding by Mamba-Based Decoders [[paper](https://arxiv.org/abs/2407.02228)] [[code](https://github.com/EnVision-Research/MTMamba)] (2024.07.02)
- VFIMamba: Video Frame Interpolation with State Space Models [[paper](https://arxiv.org/abs/2407.02315)] [[code](https://github.com/MCG-NJU/VFIMamba)] (2024.07.02)
- QueryMamba: A Mamba-Based Encoder-Decoder Architecture with a Statistical Verb-Noun Interaction Module for Video Action Forecasting @ Ego4D Long-Term Action Anticipation Challenge 2024 [[paper](https://arxiv.org/abs/2407.04184)] [[code](https://github.com/zeyun-zhong/querymamba)] (2024.07.04)
- A Mamba-based Siamese Network for Remote Sensing Change Detection [[paper](https://arxiv.org/abs/2407.06839)] [[code](https://github.com/JayParanjape/M-CD)] (2024.07.08)
- Mamba-FSCIL: Dynamic Adaptation with Selective State Space Model for Few-Shot Class-Incremental Learning [[paper](https://arxiv.org/abs/2407.06136)] (2024.07.08)
- HTD-Mamba: Efficient Hyperspectral Target Detection with Pyramid State Space Model [[paper](https://arxiv.org/abs/2407.06841)] [[code](https://github.com/shendb2022/HTD-Mamba)] (2024.07.09)
- MambaVision: A Hybrid Mamba-Transformer Vision Backbone [[paper](https://arxiv.org/abs/2407.08083)] [[code](https://github.com/NVlabs/MambaVision)] (2024.07.10)
- Parallelizing Autoregressive Generation with Variational State Space Models [[paper](https://arxiv.org/abs/2407.08415)] (2024.07.11)
- GraphMamba: An Efficient Graph Structure Learning Vision Mamba for Hyperspectral Image Classification [[paper](https://arxiv.org/abs/2407.08255)] (2024.07.11)
- ST-Mamba: Spatial-Temporal Mamba for Traffic Flow Estimation Recovery using Limited Data [[paper](https://arxiv.org/abs/2407.08558)] (2024.07.11)
- SR-Mamba: Effective Surgical Phase Recognition with State Space Model [[paper](https://arxiv.org/abs/2407.08333)] [[code](https://github.com/rcao-hk/SR-Mamba)] (2024.07.11)
- VideoMamba: Spatio-Temporal Selective State Space Model [[paper](https://arxiv.org/abs/2407.08476)] [[code](https://github.com/jinyjelly/videomamba)] (2024.07.11)
- DMM: Disparity-guided Multispectral Mamba for Oriented Object Detection in Remote Sensing [[paper](https://arxiv.org/abs/2407.08132)] [[code](https://github.com/Another-0/DMM)] (2024.07.11)
- Parallelizing Autoregressive Generation with Variational State Space Models [[paper](https://arxiv.org/abs/2407.08415)] (2024.07.11)
- InfiniMotion: Mamba Boosts Memory in Transformer for Arbitrary Long Motion Generation [[paper](https://arxiv.org/abs/2407.10061)] [[code](https://steve-zeyu-zhang.github.io/InfiniMotion/)] (2024.07.14)
- Hamba: Single-view 3D Hand Reconstruction with Graph-guided Bi-Scanning Mamba [[paper](https://arxiv.org/abs/2407.09646)] [[code](https://humansensinglab.github.io/Hamba/)] (2024.07.12)
- OPa-Ma: Text Guided Mamba for 360-degree Image Out-painting [[paper](https://arxiv.org/abs/2407.10923)] (2024.07.15)
- Enhancing Temporal Action Localization: Advanced S6 Modeling with Recurrent Mechanism [[paper](https://arxiv.org/abs/2407.13078)] (2024.07.18)
- GroupMamba: Parameter-Efficient and Accurate Group Visual State Space Model [[paper](https://arxiv.org/abs/2407.13772)] [[code](https://github.com/Amshaker/GroupMamba)] (2024.07.18)
- (ICML 2024 Workshop) Investigating the Indirect Object Identification circuit in Mamba [[paper](https://arxiv.org/abs/2407.14008)] [[openreview](https://openreview.net/forum?id=lq7ZaYuwub)] [[code](https://github.com/Phylliida/investigating-mamba-ioi)] (2024.07.19)

<span id="head6"></span>

###  Language
* MambaByte: Token-free Selective State Space Model [[paper](https://arxiv.org/abs/2401.13660)] [[code](https://github.com/lucidrains/MEGABYTE-pytorch)] (2024.01.24)
* Is Mamba Capable of In-Context Learning? [[paper](https://arxiv.org/abs/2402.03170)] (2024.02.05)
* (ICML 2024) Can Mamba Learn How to Learn? A Comparative Study on In-Context Learning Tasks  [[paper](https://arxiv.org/abs/2402.04248)] (2024.02.06)
* SpaceByte: Towards Deleting Tokenization from Large Language Modeling [[paper](https://arxiv.org/abs/2404.14408)] (2024.04.22)
* State-Free Inference of State-Space Models: The Transfer Function Approach [[paper](https://arxiv.org/abs/2405.06147)] (2024.05.10)
* Mamba4KT:An Efficient and Effective Mamba-based Knowledge Tracing Model [[paper](https://arxiv.org/abs/2405.16542)] (2024.05.26)
* Zamba: A Compact 7B SSM Hybrid Model [[paper](https://arxiv.org/abs/2405.16712)] (2024.05.26)
* State Space Models are Comparable to Transformers in Estimating Functions with Dynamic Smoothness [[paper](https://arxiv.org/abs/2405.19036)] (2024.05.29)
* Mamba State-Space Models Can Be Strong Downstream Learners [[paper](https://arxiv.org/abs/2406.00209)] (2024.05.30)
* Learning to Estimate System Specifications in Linear Temporal Logic using Transformers and Mamba [[paper](https://arxiv.org/pdf/2405.20917)] (2024.05.31)
* Pretrained Hybrids with MAD Skills [[paper](https://arxiv.org/abs/2406.00894)] (2024.06.02)
* LongSSM: On the Length Extension of State-space Models in Language Modelling [[paper](https://arxiv.org/abs/2406.02080)] (2024.06.04)
* Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling [[paper](https://arxiv.org/abs/2406.07522)] [[code](https://github.com/microsoft/Samba)] (2024.06.04)
* How Effective are State Space Models for Machine Translation? [[paper](https://arxiv.org/abs/2407.05489)] (2024.06.07)
* State Soup: In-Context Skill Learning, Retrieval and Mixing [[paper](https://arxiv.org/abs/2406.08423)] (2024.06.12)
* An Empirical Study of Mamba-based Language Models [[paper](https://arxiv.org/abs/2406.07887)] (2024.06.12)
* DeciMamba: Exploring the Length Extrapolation Potential of Mamba [[paper](https://arxiv.org/abs/2406.14528)] [[code](https://github.com/assafbk/DeciMamba)] (2024.06.20)
* Hydra: Bidirectional State Space Models Through Generalized Matrix Mixers [[paper](https://arxiv.org/abs/2407.09941)] [[code](https://github.com/goombalab/hydra)] (2024.07.13)
* MambaForGCN: Enhancing Long-Range Dependency with State Space Model and Kolmogorov-Arnold Networks for Aspect-Based Sentiment Analysis [[paper](https://arxiv.org/abs/2407.10347)] (2024.07.14)
* Longhorn: State Space Models are Amortized Online Learners [[paper](https://arxiv.org/abs/2407.14207)] [[code](https://github.com/Cranial-XIX/longhorn)] (2024.07.13)

<span id="head7"></span>

###  Multi-Modal
* VL-Mamba: Exploring State Space Models for Multimodal Learning [[paper](https://arxiv.org/abs/2403.13600)] [[code](https://github.com/ZhengYu518/VL-Mamba)] (2024.03.20)
* Cobra: Extending Mamba to Multi-Modal Large Language Model for Efficient Inference [[paper](https://arxiv.org/abs/2403.14520)] [[code](https://sites.google.com/view/cobravlm)] (2024.03.21)
* SpikeMba: Multi-Modal Spiking Saliency Mamba for Temporal Video Grounding [[paper](https://arxiv.org/abs/2404.01174)] (2024.04.01)
* Sigma: Siamese Mamba Network for Multi-Modal Semantic Segmentation [[paper](https://arxiv.org/abs/2404.04256)] [[code](https://github.com/zifuwan/Sigma)] (2024.04.04)
* SurvMamba: State Space Model with Multi-grained Multi-modal Interaction for Survival Prediction [[paper](https://arxiv.org/abs/2404.08027)] (2024.04.11)
* MambaDFuse: A Mamba-based Dual-phase Model for Multi-modality Image Fusion [[paper](https://arxiv.org/abs/2404.08406)] (2024.04.12)
* Fusion-Mamba for Cross-modality Object Detection [[paper](https://arxiv.org/abs/2404.09146)] (2024.04.14)
* CFMW: Cross-modality Fusion Mamba for Multispectral Object Detection under Adverse Weather Conditions [[paper](https://arxiv.org/abs/2404.16302)] [[code](https://github.com/lhy-zjut/CFMW)] (2024.04.25)
* Meteor: Mamba-based Traversal of Rationale for Large Language and Vision Models [[paper](https://arxiv.org/abs/2405.15574)] [[code](https://github.com/ByungKwanLee/Meteor)] (2024.05.24)
* Coupled Mamba: Enhanced Multi-modal Fusion with Coupled State Space Model [[paper](https://arxiv.org/abs/2405.18014)] (2024.05.28)
* S4Fusion: Saliency-aware Selective State Space Model for Infrared Visible Image Fusion [[paper](https://arxiv.org/abs/2405.20881)] (2024.05.31)
* SHMamba: Structured Hyperbolic State Space Model for Audio-Visual Question Answering [[paper](https://arxiv.org/abs/2406.09833)] (2024.06.14)

<span id="head27"></span>

### Diffusion 

* Scalable Diffusion Models with State Space Backbone [[paper](https://arxiv.org/abs/2402.05608)] [[code](https://github.com/feizc/dis)] (2024.02.08)
* P-Mamba: Marrying Perona Malik Diffusion with Mamba for Efficient Pediatric Echocardiographic Left Ventricular Segmentation [[paper](https://arxiv.org/abs/2402.08506)] (2024.02.13)
* MD-Dose: A Diffusion Model based on the Mamba for Radiotherapy Dose Prediction [[paper](https://arxiv.org/abs/2403.08479)] [[code](https://github.com/flj19951219/mamba_dose)] (2024.03.13)
* (ECCV 2024) ZigMa: Zigzag Mamba Diffusion Model [[paper](https://arxiv.org/abs/2403.13802)] [[code](https://github.com/CompVis/zigma)] [[project](https://taohu.me/zigma/)] (2024.03.20)
* SMCD: High Realism Motion Style Transfer via Mamba-based Diffusion [[paper](https://arxiv.org/abs/2405.02844)] (2024.05.06)
* VM-DDPM: Vision Mamba Diffusion for Medical Image Synthesis [[paper](https://arxiv.org/abs/2405.05667)] (2024.05.09)
* DiM: Diffusion Mamba for Efficient High-Resolution Image Synthesis [[paper](https://arxiv.org/abs/2405.14224)] (2024.05.23)
* UniAnimate: Taming Unified Video Diffusion Models for Consistent Human Image Animation [[paper](https://arxiv.org/abs/2406.01188)] [[code](https://github.com/ali-vilab/UniAnimate)] (2024.06.03)
* Dimba: Transformer-Mamba Diffusion Models [[paper](https://arxiv.org/abs/2406.01159)] (2024.06.03)

<span id="head8"></span>

###  Medical
- U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation [[paper](https://arxiv.org/abs/2401.04722)] [[code](https://github.com/bowang-lab/U-Mamba)] [[dataset](https://drive.google.com/drive/folders/1DmyIye4Gc9wwaA7MVKFVi-bWD2qQb-qN?usp=sharing)] [[homepage](https://wanglab.ai/u-mamba.html)] (2024.01.09)

- SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation [[paper](https://arxiv.org/abs/2401.13560)] [[code](https://github.com/ge-xing/SegMamba)] (2024.01.24)

- MambaMorph: a Mamba-based Backbone with Contrastive Feature Learning for Deformable MR-CT Registration [[paper](https://arxiv.org/abs/2401.13934)] [[code](https://github.com/guo-stone/mambamorph)] (2024.01.24)

- Vivim: a Video Vision Mamba for Medical Video Object Segmentation [[paper](https://arxiv.org/abs/2401.14168)] [[code](https://github.com/scott-yjyang/Vivim)] (2024.01.25)

- VM-UNet: Vision Mamba UNet for Medical Image Segmentation [[paper](https://arxiv.org/abs/2402.02491)] [[code](https://github.com/jcruan519/vm-unet)] (2024.02.04)

- Swin-UMamba: Mamba-based UNet with ImageNet-based pretraining [[paper](https://arxiv.org/abs/2402.03302)] [[code](https://github.com/jiarunliu/swin-umamba)] (2024.02.05)

- nnMamba: 3D Biomedical Image Segmentation, Classification and Landmark Detection with State Space Model [[paper](https://arxiv.org/abs/2402.03526)] [[code](https://github.com/lhaof/nnmamba)] (2024.02.05)

- Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation [[paper](https://arxiv.org/abs/2402.05079)] [[code](https://github.com/ziyangwang007/mamba-unet)] (2024.02.07)

- FD-Vision Mamba for Endoscopic Exposure Correction [[paper](https://arxiv.org/abs/2402.06378)] (2024.02.09)

- Semi-Mamba-UNet: Pixel-Level Contrastive Cross-Supervised Visual Mamba-based UNet for Semi-Supervised Medical Image Segmentation [[paper](https://arxiv.org/abs/2402.07245)] [[code](https://github.com/ziyangwang007/mamba-unet)] (2024.02.11)

- P-Mamba: Marrying Perona Malik Diffusion with Mamba for Efficient Pediatric Echocardiographic Left Ventricular Segmentation [[paper](https://arxiv.org/abs/2402.08506)] (2024.02.13)

- Weak-Mamba-UNet: Visual Mamba Makes CNN and ViT Work Better for Scribble-based Medical Image Segmentation [[paper](https://arxiv.org/abs/2402.10887)] [[code](https://github.com/ziyangwang007/mamba-unet)] (2024.02.16)

- MambaMIR: An Arbitrary-Masked Mamba for Joint Medical Image Reconstruction and Uncertainty Estimation [[paper](https://arxiv.org/abs/2402.18451)] (2024.02.28)

- A PTM-Aware Protein Language Model with Bidirectional Gated Mamba Blocks

  [[Paper](https://www.biorxiv.org/content/10.1101/2024.02.28.581983v1)] [[Huggingface](https://huggingface.co/ChatterjeeLab/PTM-Mamba)] [[code](https://github.com/programmablebio/ptm-mamba)] (2024.02.28)

- MedMamba: Vision Mamba for Medical Image Classification [[paper](https://arxiv.org/abs/2403.03849)] [[code](https://github.com/YubiaoYue/MedMamba)] (2024.03.06)

- Motion-Guided Dual-Camera Tracker for Low-Cost Skill Evaluation of Gastric Endoscopy [[paper](https://arxiv.org/abs/2403.05146)] (2024.03.08)

- MamMIL: Multiple Instance Learning for Whole Slide Images with State Space Models [[paper](https://arxiv.org/abs/2403.05160)] (2024.03.08)

- LightM-UNet: Mamba Assists in Lightweight UNet for Medical Image Segmentation [[paper](https://arxiv.org/abs/2403.05246)] [[code](https://github.com/MrBlankness/LightM-UNet)] (2024.03.08)

- ClinicalMamba: A Generative Clinical Language Model on Longitudinal Clinical Notes [[paper](https://arxiv.org/abs/2403.05795)] (2024.03.09)

- Large Window-based Mamba UNet for Medical Image Segmentation: Beyond Convolution and Self-attention [[paper](https://arxiv.org/abs/2403.07332)] [[code](https://github.com/wjh892521292/lma-unet)] (2024.03.12)

- MD-Dose: A Diffusion Model based on the Mamba for Radiotherapy Dose Prediction [[paper](https://arxiv.org/abs/2403.08479)] [[code](https://github.com/flj19951219/mamba_dose)] (2024.03.13)

- VM-UNET-V2 Rethinking Vision Mamba UNet for Medical Image Segmentation [[paper](https://arxiv.org/abs/2403.09157)] [[code](https://github.com/nobodyplayer1/vm-unetv2)] (2024.03.14)

- H-vmunet: High-order Vision Mamba UNet for Medical Image Segmentation [[paper](https://arxiv.org/abs/2403.13642)] [[code](https://github.com/wurenkai/H-vmunet)] (2024.03.20)

- ProMamba: Prompt-Mamba for polyp segmentation [[paper](https://arxiv.org/abs/2403.13660)] (2024.03.20)

- UltraLight VM-UNet: Parallel Vision Mamba Significantly Reduces Parameters for Skin Lesion Segmentation [[paper](https://arxiv.org/abs/2403.20035)] [[code](https://github.com/wurenkai/UltraLight-VM-UNet)] (2024.03.29)

- VMambaMorph: a Visual Mamba-based Framework with Cross-Scan Module for Deformable 3D Image Registration [[paper](https://arxiv.org/abs/2404.05105)] (2024.04.07)

- Vim4Path: Self-Supervised Vision Mamba for Histopathology Images [[paper](https://arxiv.org/abs/2404.13222)] [[code](https://github.com/AtlasAnalyticsLab/Vim4Path)] (2024.04.20)

- Sparse Reconstruction of Optical Doppler Tomography Based on State Space Model [[paper](https://arxiv.org/abs/2404.17484)] (2024.04.26)

- AC-MAMBASEG: An adaptive convolution and Mamba-based architecture for enhanced skin lesion segmentation [[paper](https://arxiv.org/abs/2405.03011)] [[code](https://github.com/vietthanh2710/AC-MambaSeg)] (2024.05.05)

- HC-Mamba: Vision MAMBA with Hybrid Convolutional Techniques for Medical Image Segmentation [[paper](https://arxiv.org/abs/2405.05007)] (2024.05.08)

- VM-DDPM: Vision Mamba Diffusion for Medical Image Synthesis [[paper](https://arxiv.org/abs/2405.05667)] (2024.05.09)

- I2I-Mamba: Multi-modal medical image synthesis via selective state space modeling [[paper](https://arxiv.org/abs/2405.14022)] [[code](https://github.com/icon-lab/I2I-Mamba)] (2024.05.22)

- EHRMamba: Towards Generalizable and Scalable Foundation Models for Electronic Health Records [[paper](https://arxiv.org/abs/2405.14567)] (2024.05.23)

- MUCM-Net: A Mamba Powered UCM-Net for Skin Lesion Segmentation [[paper](https://arxiv.org/abs/2405.15925)] [[code](https://github.com/chunyuyuan/MUCM-Net)] (2024.05.24)

- UU-Mamba: Uncertainty-aware U-Mamba for Cardiac Image Segmentation [[paper](https://arxiv.org/abs/2405.17496)] (2024.05.25)

- Enhancing Global Sensitivity and Uncertainty Quantification in Medical Image Reconstruction with Monte Carlo Arbitrary-Masked Mamba [[paper](https://arxiv.org/abs/2405.17659)] (2024.05.27)

- Cardiovascular Disease Detection from Multi-View Chest X-rays with BI-Mamba [[paper](https://arxiv.org/abs/2405.18533)] (2024.05.28)

- fMRI predictors based on language models of increasing complexity recover brain left lateralization [[paper](https://arxiv.org/abs/2405.17992)] [[code](https://github.com/l-bg/llms_brain_lateralization)] (2024.05.28)

- SAM-VMNet: Deep Neural Networks For Coronary Angiography Vessel Segmentation [[paper](https://arxiv.org/abs/2406.00492)] (2024.06.01)

- Combining Graph Neural Network and Mamba to Capture Local and Global Tissue Spatial Relationships in Whole Slide Images [[paper](https://arxiv.org/abs/2406.04377)] [[code](https://github.com/rina-ding/gat-mamba)] (2024.06.05)

- Convolution and Attention-Free Mamba-based Cardiac Image Segmentation [[paper](https://arxiv.org/abs/2406.05786)] (2024.06.09)

- MMR-Mamba: Multi-Contrast MRI Reconstruction with Mamba and Spatial-Frequency Information Fusion [[paper](https://arxiv.org/abs/2406.18950)] (2024.06.27)

- Vision Mamba for Classification of Breast Ultrasound Images [[paper](https://arxiv.org/abs/2407.03552)] (2024.07.04)

- Fine-grained Context and Multi-modal Alignment for Freehand 3D Ultrasound Reconstruction [[paper](https://arxiv.org/abs/2407.04242)] (2024.07.05)

- LGRNet: Local-Global Reciprocal Network for Uterine Fibroid Segmentation in Ultrasound Videos [[paper](https://arxiv.org/abs/2407.05703)] [[code](https://github.com/bio-mlhui/LGRNet)] (2024.07.08)

- SliceMamba for Medical Image Segmentation [[paper](https://arxiv.org/abs/2407.08481)] (2024.07.11)

<span id="head9"></span>

### Tabular Data
* MambaTab: A Simple Yet Effective Approach for Handling Tabular Data [[paper](https://arxiv.org/abs/2401.08867)] (2024.01.16)

<span id="head10"></span>
###  Graph
* Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces [[paper](https://arxiv.org/abs/2402.00789)] [[code](https://github.com/bowang-lab/Graph-Mamba)] (2024.02.01)
* Graph Mamba: Towards Learning on Graphs with State Space Models [[paper](https://arxiv.org/abs/2402.08678)] [[code](https://github.com/graphmamba/gmn)] (2024.02.13)
* STG-Mamba: Spatial-Temporal Graph Learning via Selective State Space Model [[paper](https://arxiv.org/abs/2403.12418)] (2024.03.19)
* HeteGraph-Mamba: Heterogeneous Graph Learning via Selective State Space Model [[paper](https://arxiv.org/abs/2405.13915)] (2024.05.22)
* State Space Models on Temporal Graphs: A First-Principles Study [[paper](https://arxiv.org/abs/2406.00943)] (2024.06.03)
* Learning Long Range Dependencies on Graphs via Random Walks [[paper](https://arxiv.org/abs/2406.03386)] [[code](https://github.com/BorgwardtLab/NeuralWalker)] (2024.06.05)
* What Can We Learn from State Space Models for Machine Learning on Graphs? [[paper](https://arxiv.org/abs/2406.05815)] [[code](https://github.com/Graph-COM/GSSC)] (2024.06.09)
* SpoT-Mamba: Learning Long-Range Dependency on Spatio-Temporal Graphs with Selective State Spaces [[paper](https://arxiv.org/abs/2406.11244)] [[code](https://github.com/bdi-lab/SpoT-Mamba)] (2024.06.17)
* GraphMamba: An Efficient Graph Structure Learning Vision Mamba for Hyperspectral Image Classification [[paper](https://arxiv.org/abs/2407.08255)] (2024.07.11)

<span id="head11"></span>
### Point Cloud
* PointMamba: A Simple State Space Model for Point Cloud Analysis [[paper](https://arxiv.org/abs/2402.10739)] (2024.02.16)
* Point Could Mamba: Point Cloud Learning via State Space Model [[paper](https://arxiv.org/abs/2403.00762)] [[code](https://github.com/zhang-tao-whu/pcm)] (2024.03.01)
* 3DMambaIPF: A State Space Model for Iterative Point Cloud Filtering via Differentiable Rendering [[paper](https://arxiv.org/abs/2404.05522)] (2024.04.08)
* 3DMambaComplete: Exploring Structured State Space Model for Point Cloud Completion [[paper](https://arxiv.org/abs/2404.07106)] (2024.04.10)
* Mamba3D: Enhancing Local Features for 3D Point Cloud Analysis via State Space Model [[paper](https://arxiv.org/abs/2404.14966)] [[code](https://github.com/xhanxu/Mamba3D)] (2024.04.23)
* MAMBA4D: Efficient Long-Sequence Point Cloud Video Understanding with Disentangled Spatial-Temporal State Space Models [[paper](https://arxiv.org/abs/2405.14338)] (2024.05.23)
* PoinTramba: A Hybrid Transformer-Mamba Framework for Point Cloud Analysis [[paper](https://arxiv.org/abs/2405.15463)] [[code](https://github.com/xiaoyao3302/PoinTramba)] (2024.05.24)
* LCM: Locally Constrained Compact Point Cloud Model for Masked Point Modeling [[paper](https://arxiv.org/abs/2405.17149)] (2024.05.27)
* PointABM:Integrating Bidirectional State Space Model with Multi-Head Self-Attention for Point Cloud Analysis [[paper](https://arxiv.org/abs/2406.06069)] (2024.06.10)
* Voxel Mamba: Group-Free State Space Models for Point Cloud based 3D Object Detection [[paper](https://arxiv.org/abs/2406.10700)] (2024.06.15)
* Mamba24/8D: Enhancing Global Interaction in Point Clouds via State Space Model [[paper](https://arxiv.org/abs/2406.17442)] (2024.06.25)

<span id="head12"></span>
### Time Series
* Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling [[paper](https://arxiv.org/abs/2402.10211)] [[code](https://github.com/raunaqbhirangi/hiss)] [[homepage](https://hiss-csp.github.io/)] (2024.02.15)
* MambaStock: Selective state space model for stock prediction [[paper](https://arxiv.org/abs/2402.18959)] [[code](https://github.com/zshicode/MambaStock)] (2024.02.29)
* MambaLithium: Selective state space model for remaining-useful-life, state-of-health, and state-of-charge estimation of lithium-ion batteries [[paper](https://arxiv.org/abs/2403.05430)] [[code](https://github.com/zshicode/MambaLithium)] (2024.03.08)
* TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting [[paper](https://arxiv.org/abs/2403.09898)] [[code](https://github.com/atik-ahamed/timemachine)] (2024.03.14)
* Is Mamba Effective for Time Series Forecasting? [[paper](https://arxiv.org/abs/2403.11144)] [[code](https://github.com/wzhwzhwzh0921/S-D-Mamba)] (2024.03.17)
* SiMBA: Simplified Mamba-Based Architecture for Vision and Multivariate Time series [[paper](https://arxiv.org/abs/2403.15360)] (2024.03.22)
* MambaMixer: Efficient Selective State Space Models with Dual Token and Channel Selection [[paper](https://arxiv.org/abs/2403.19888)] [[project](https://mambamixer.github.io/)] (2024.03.29)
* HARMamba: Efficient Wearable Sensor Human Activity Recognition Based on Bidirectional Selective SSM [[paper](https://arxiv.org/abs/2403.20183)] (2024.03.29)
* Integrating Mamba and Transformer for Long-Short Range Time Series Forecasting [[paper](https://arxiv.org/abs/2404.14757)] [[code](https://github.com/XiongxiaoXu/Mambaformer-in-Time-Series)] (2024.04.23)
* Bi-Mamba4TS: Bidirectional Mamba for Time Series Forecasting [[paper](https://arxiv.org/abs/2404.15772)] [[code](https://github.com/davidwynter/Bi-Mamba4TS)] (2024.04.24)
* MAMCA -- Optimal on Accuracy and Efficiency for Automatic Modulation Classification with Extended Signal Length [[paper](https://arxiv.org/abs/2405.11263)] [[code](https://github.com/ZhangYezhuo/MAMCA)] (2024.05.18)
* Time-SSM: Simplifying and Unifying State Space Models for Time Series Forecasting [[paper](https://arxiv.org/abs/2405.16312)] (2024.05.25)
* MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting [[paper](https://arxiv.org/abs/2405.16440)] [[code](https://github.com/XiudingCai/MambaTS-pytorch)] (2024.05.26)
* Efficient Time Series Processing for Transformers and State-Space Models through Token Merging [[paper](https://arxiv.org/abs/2405.17951)] (2024.05.28)
* Joint Selective State Space Model and Detrending for Robust Time Series Anomaly Detection [[paper](https://arxiv.org/abs/2405.19823)] (2024.05.30)
* MSSC-BiMamba: Multimodal Sleep Stage Classification and Early Diagnosis of Sleep Disorders with Bidirectional Mamba [[paper](https://arxiv.org/abs/2405.20142)] (2024.05.30)
* Chimera: Effectively Modeling Multivariate Time Series with 2-Dimensional State Space Models [[paper](https://arxiv.org/abs/2406.04320)] (2024.06.06)
* TSCMamba: Mamba Meets Multi-View Learning for Time Series Classification [[paper](https://arxiv.org/abs/2406.04419)] (2024.06.06)
* C-Mamba: Channel Correlation Enhanced State Space Models for Multivariate Time Series Forecasting [[paper](https://arxiv.org/abs/2406.05316)] (2024.06.08)
* ECGMamba: Towards Efficient ECG Classification with BiSSM [[paper](https://arxiv.org/abs/2406.10098)] (2024.06.14)
* Mamba Hawkes Process [[paper](https://arxiv.org/abs/2407.05302)] (2024.07.07)
* MSegRNN:Enhanced SegRNN Model with Mamba for Long-Term Time Series Forecasting [[paper](https://arxiv.org/abs/2407.10768)] (2024.07.15)
* FMamba: Mamba based on Fast-attention for Multivariate Time-series Forecasting [[paper](https://arxiv.org/abs/2407.10768)] (2024.07.20)

<span id="head13"></span>

###  Speech
* Multichannel Long-Term Streaming Neural Speech Enhancement for Static and Moving Speakers [[paper](https://arxiv.org/abs/2403.07675)] [[code](https://github.com/Audio-WestlakeU/NBSS)] (2024.03.12)
* Dual-path Mamba: Short and Long-term Bidirectional Selective Structured State Space Models for Speech Separation [[paper](https://arxiv.org/abs/2403.18257)] (2024.03.27)
* Multichannel Long-Term Streaming Neural Speech Enhancement for Static and Moving Speakers [[paper](https://arxiv.org/abs/2403.18276)] [[code](https://github.com/zhichaoxu-shufe/RankMamba)] (2024.03.27)
* SPMamba: State-space model is all you need in speech separation [[paper](https://arxiv.org/abs/2404.02063)] [[code](https://github.com/JusperLee/SPMamba)] (2024.04.02)
* TRAMBA: A Hybrid Transformer and Mamba Architecture for Practical Audio and Bone Conduction Speech Super Resolution and Enhancement on Mobile and Wearable Platforms [[paper](https://arxiv.org/abs/2405.01242)] (2024.05.02)
* An Investigation of Incorporating Mamba for Speech Enhancement [[paper](https://arxiv.org/abs/2405.06573)] (2024.05.10)
* SSAMBA: Self-Supervised Audio Representation Learning with Mamba State Space Model [[paper](https://arxiv.org/abs/2405.11831)] (2024.05.20)
* Mamba in Speech: Towards an Alternative to Self-Attention [[paper](https://arxiv.org/abs/2405.12609)] (2024.05.21)
* Audio Mamba: Pretrained Audio State Space Model For Audio Tagging [[paper](https://arxiv.org/abs/2405.13636)] (2024.05.22)
* Audio Mamba: Selective State Spaces for Self-Supervised Audio Representations [[paper](https://arxiv.org/abs/2406.02178)] [[code](https://github.com/SiavashShams/ssamba)] (2024.06.04)
* Audio Mamba: Bidirectional State Space Model for Audio Representation Learning [[paper](https://arxiv.org/abs/2406.03344)] [[code](https://github.com/kyegomez/AudioMamba)] (2024.06.05)
* RawBMamba: End-to-End Bidirectional State Space Model for Audio Deepfake Detection [[paper](https://arxiv.org/abs/2406.06086)] (2024.06.10)
* Exploring the Capability of Mamba in Speech Applications [[paper](https://arxiv.org/abs/2406.16808)] (2024.06.24)
* DASS: Distilled Audio State Space Models Are Stronger and More Duration-Scalable Learners [[paper](https://export.arxiv.org/abs/2407.04082)] (2024.07.04)
* Speech Slytherin: Examining the Performance and Efficiency of Mamba for Speech Separation, Recognition, and Synthesis [[paper](https://arxiv.org/abs/2407.09732)] [[code](https://github.com/xi-j/Mamba-ASR)] (2024.07.13)

<span id="head14"></span>

### Recommendation 
* Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models [[paper](https://arxiv.org/abs/2403.05430)] [[code](https://github.com/chengkai-liu/mamba4rec)] (2024.03.06)
* Uncovering Selective State Space Model's Capabilities in Lifelong Sequential Recommendation [[paper](https://arxiv.org/abs/2403.16371)] [[code](https://github.com/nancheng58/Rec-Mamba)] (2024.03.25)
* EchoMamba4Rec: Harmonizing Bidirectional State Space Models with Spectral Filtering for Advanced Sequential Recommendation [[paper](https://arxiv.org/abs/2403.16371)] (2024.06.04)
* (WSDM 2024) Behavior-Dependent Linear Recurrent Units for Efficient Sequential Recommendation [[paper](https://arxiv.org/abs/2406.12580)] [[code](https://github.com/yueqirex/LRURec)] (2024.06.18)
* MLSA4Rec: Mamba Combined with Low-Rank Decomposed Self-Attention for Sequential Recommendation [[paper](https://arxiv.org/abs/2407.13135)] (2024.07.18)

<span id="head15"></span>
### Reinforcement Learning
* Decision Mamba: Reinforcement Learning via Sequence Modeling with Selective State Spaces [[paper](https://arxiv.org/abs/2403.19925)] [[code](https://github.com/toshihiro-ota/decision-mamba)] (2024.03.29)
* Hierarchical Decision Mamba [[paper](https://arxiv.org/abs/2405.07943)] [[code](https://github.com/meowatthemoon/HierarchicalDecisionMamba)] (2024.05.13)
* Is Mamba Compatible with Trajectory Optimization in Offline Reinforcement Learning? [[paper](https://arxiv.org/abs/2405.12094)] (2024.05.20)
* Deciphering Movement: Unified Trajectory Generation Model for Multi-Agent [[paper](https://arxiv.org/abs/2405.17680)] [[code](https://github.com/colorfulfuture/UniTraj-pytorch)] (2024.05.27)
* Decision Mamba: Reinforcement Learning via Hybrid Selective Sequence Modeling [[paper](https://arxiv.org/abs/2406.00079)] (2024.05.31)
* Mamba as Decision Maker: Exploring Multi-scale Sequence Modeling in Offline Reinforcement Learning [[paper](https://arxiv.org/abs/2406.02013)] [[code](https://github.com/AndyCao1125/MambaDM)] (2024.06.04)
* RoboMamba: Multimodal State Space Model for Efficient Robot Reasoning and Manipulation [[paper](https://arxiv.org/abs/2406.04339)] [[code](https://github.com/lmzpai/roboMamba)] (2024.06.06)
* Decision Mamba: A Multi-Grained State Space Model with Self-Evolution Regularization for Offline RL [[paper](https://arxiv.org/abs/2406.05427)] (2024.06.08)
* MaIL: Improving Imitation Learning with Mamba [[paper](https://arxiv.org/abs/2406.08234)] (2024.06.12)
* KalMamba: Towards Efficient Probabilistic State Space Models for RL under Uncertainty [[paper](https://arxiv.org/abs/2406.15131)] (2024.06.21)

<span id="head16"></span>
###  Survey
* State Space Model for New-Generation Network Alternative to Transformers: A Survey [[paper](https://arxiv.org/abs/2404.09516)] [[project](https://github.com/Event-AHU/Mamba_State_Space_Model_Paper_List)] (2024.04.15)
* Mamba-360: Survey of State Space Models as Transformer Alternative for Long Sequence Modelling: Methods, Applications, and Challenges [[paper](https://arxiv.org/abs/2404.16112)] [[project](https://github.com/badripatro/mamba360)] (2024.04.24)
* A Survey on Visual Mamba [[paper](https://arxiv.org/abs/2404.15956)] (2024.04.24)
* A Survey on Vision Mamba: Models, Applications and Challenges [[paper](https://arxiv.org/abs/2404.18861)] [[project](https://github.com/Ruixxxx/Awesome-Vision-Mamba-Models)] (2024.04.29)
* Vision Mamba: A Comprehensive Survey and Taxonomy [[paper](https://arxiv.org/abs/2405.04404)] [[project](https://github.com/lx6c78/Vision-Mamba-A-Comprehensive-Survey-and-Taxonomy)] (2024.05.07)
* Surveying Image Segmentation Approaches in Astronomy [[paper](https://arxiv.org/abs/2405.14238)] (2024.05.23)
* Computation-Efficient Era: A Comprehensive Survey of State Space Models in Medical Image Analysis [[paper](https://arxiv.org/abs/2406.03430)] [[project](https://github.com/xmindflow/Awesome_Mamba)] (2024.05.07)

<span id="head17"></span>
##  Tutorials
<span id="head18"></span>
###  Blogs
* The Annotated S4 [[URL](https://srush.github.io/annotated-s4/#part-1b-addressing-long-range-dependencies-with-hippo)]
* The Annotated Mamba [[URL](https://srush.github.io/annotated-mamba/hard.html#part-1-cumulative-sums)]
* A Visual Guide to Mamba and State Space Models [[URL](https://maartengrootendorst.substack.com/p/a-visual-guide-to-mamba-and-state)]
* Mamba No. 5 (A Little Bit Of...) [[URL](https://jameschen.io/jekyll/update/2024/02/12/mamba.html)]
* State Space Duality (Mamba-2) [[PART1](https://tridao.me/blog/2024/mamba2-part1-model/)] [[PART2](https://tridao.me/blog/2024/mamba2-part2-theory/)] [[PART3](https://tridao.me/blog/2024/mamba2-part3-algorithm/)] [[PART4](https://tridao.me/blog/2024/mamba2-part4-systems/)]

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
  * mamba.c: Inference of Mamba models in pure C. [[URL](https://github.com/kroggen/mamba.c)]

<span id="head22"></span>
### Other Awesome Mamba List
* awesome-ssm-ml [[URL](https://github.com/AvivBick/awesome-ssm-ml)]
* Awesome-Mamba: Collect papers about Mamba [[URL](https://github.com/caojiaolong/Awesome-Mamba)]
* Awesome-Mamba-Papers [[URL](https://github.com/yyyujintang/Awesome-Mamba-Papers)]
* Awesome-Mamba [[URL](https://github.com/pengzhangzhi/Awesome-Mamba)]

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
