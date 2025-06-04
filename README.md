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
  - [ Spatio-Temporal](#head29)
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
- [ Citation](#head28)
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
* Jamba-1.5: Hybrid Transformer-Mamba Models at Scale [[paper](https://arxiv.org/abs/2408.12570)] (2024.08.22)
* (ICCAD 2024) MARCA: Mamba Accelerator with ReConfigurable Architecture [[paper](https://arxiv.org/abs/2409.11440)] (2024.09.16)
* S7: Selective and Simplified State Space Layers for Sequence Modeling [[paper](https://arxiv.org/abs/2410.03464)] (2024.10.04)
* TransMamba: Flexibly Switching between Transformer and Mamba [[paper](https://arxiv.org/abs/2503.24067)] (2025.03.31)

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
* Parallelizing Linear Transformers with the Delta Rule over Sequence Length [[paper](https://arxiv.org/abs/2406.06484)] [[code](https://github.com/sustcsonglin/flash-linear-attention)] (2024.06.10)
* MambaLRP: Explaining Selective State Space Sequence Models [[paper](https://arxiv.org/abs/2406.07592)] (2024.06.11)
* Towards a theory of learning dynamics in deep state space models [[paper](https://arxiv.org/abs/2407.07279)] (2024.07.10)
* Lambda-Skip Connections: the architectural component that prevents Rank Collapse [[paper](https://arxiv.org/abs/2410.10609)] (2024.10.14)
* Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention [[paper](https://arxiv.org/abs/2502.01473)] (2025.02.03)
* EVM-Fusion: An Explainable Vision Mamba Architecture with Neural Algorithmic Fusion [[paper](https://arxiv.org/abs/2505.17367)] (2025.05.23)
* (ACL 2025) Mamba Knockout for Unraveling Factual Information Flow [[paper](https://arxiv.org/abs/2505.24244)] (2025.05.30)

<span id="head5"></span>
###  Vision
- (ICML 2024) Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model [[paper](https://arxiv.org/abs/2401.09417)] [[code](https://github.com/hustvl/Vim)] (2024.01.17)
- VMamba: Visual State Space Model [[paper](https://arxiv.org/abs/2401.10166)] [[code](https://github.com/MzeroMiko/VMamba)] (2024.01.18)
- U-shaped Vision Mamba for Single Image Dehazing [[paper](https://arxiv.org/abs/2402.04139)] (2024.02.06)
- Scalable Diffusion Models with State Space Backbone [[paper](https://arxiv.org/abs/2402.05608)] [[code](https://github.com/feizc/dis)] (2024.02.08)
- Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data [[paper](https://arxiv.org/abs/2402.05892)] (2024.02.08)
- Pan-Mamba: Effective pan-sharpening with State Space Model [[paper](https://arxiv.org/abs/2402.12192)] (2024.02.19)
- (ECCV 2024) MambaIR: A Simple Baseline for Image Restoration with State-Space Model [[paper](https://arxiv.org/abs/2402.15648)] [[code](https://github.com/csguoh/mambair)] (2024.02.23)
- (CVPR 2024 spotlight) State Space Models for Event Cameras [[paper](https://arxiv.org/abs/2402.15584)] [[code](https://github.com/uzh-rpg/ssms_event_cameras)] (2024.02.23)
- Res-VMamba: Fine-Grained Food Category Visual Classification Using Selective State Space Models with Deep Residual Learning [[paper](https://arxiv.org/abs/2402.15761)] [[code](https://github.com/chishengchen/resvmamba)] (2024.02.24)
- MiM-ISTD: Mamba-in-Mamba for Efficient Infrared Small Target Detection [[paper](https://arxiv.org/abs/2403.02148)] [[code](https://github.com/txchen-USTC/MiM-ISTD)] (2024.03.04)
- Motion Mamba: Efficient and Long Sequence Motion Generation with Hierarchical and Bidirectional Selective SSM [[paper](https://arxiv.org/abs/2403.07487)] [[code](https://github.com/steve-zeyu-zhang/MotionMamba)] (2024.03.12)
- LocalMamba: Visual State Space Model with Windowed Selective Scan [[paper](https://arxiv.org/abs/2403.09338)] [[code](https://github.com/hunto/LocalMamba)] (2024.03.14)
- EfficientVMamba: Atrous Selective Scan for Light Weight Visual Mamba [[paper](https://arxiv.org/abs/2403.09977)] (2024.03.15)
- Exploring Learning-based Motion Models in Multi-Object Tracking [[paper](https://arxiv.org/abs/2403.10826)] (2024.03.16)
- VmambaIR: Visual State Space Model for Image Restoration [[paper](https://arxiv.org/abs/2403.11423)] [[code](https://github.com/alphacatplus/vmambair)] (2024.03.18)
- (ECCV 2024) ZigMa: Zigzag Mamba Diffusion Model [[paper](https://arxiv.org/abs/2403.13802)] [[code](https://github.com/CompVis/zigma)] [[project](https://taohu.me/zigma/)] (2024.03.20)
- SiMBA: Simplified Mamba-Based Architecture for Vision and Multivariate Time series [[paper](https://arxiv.org/abs/2403.15360)] (2024.03.22)
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
- (ICLR 2025) Rethinking Spiking Neural Networks as State Space Models [[paper](https://arxiv.org/abs/2406.02923)] (2024.06.05)
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
- (ECCV 2024) MTMamba: Enhancing Multi-Task Dense Scene Understanding by Mamba-Based Decoders [[paper](https://arxiv.org/abs/2407.02228)] [[code](https://github.com/EnVision-Research/MTMamba)] (2024.07.02)
- QueryMamba: A Mamba-Based Encoder-Decoder Architecture with a Statistical Verb-Noun Interaction Module for Video Action Forecasting @ Ego4D Long-Term Action Anticipation Challenge 2024 [[paper](https://arxiv.org/abs/2407.04184)] [[code](https://github.com/zeyun-zhong/querymamba)] (2024.07.04)
- A Mamba-based Siamese Network for Remote Sensing Change Detection [[paper](https://arxiv.org/abs/2407.06839)] [[code](https://github.com/JayParanjape/M-CD)] (2024.07.08)
- Mamba-FSCIL: Dynamic Adaptation with Selective State Space Model for Few-Shot Class-Incremental Learning [[paper](https://arxiv.org/abs/2407.06136)] (2024.07.08)
- HTD-Mamba: Efficient Hyperspectral Target Detection with Pyramid State Space Model [[paper](https://arxiv.org/abs/2407.06841)] [[code](https://github.com/shendb2022/HTD-Mamba)] (2024.07.09)
- MambaVision: A Hybrid Mamba-Transformer Vision Backbone [[paper](https://arxiv.org/abs/2407.08083)] [[code](https://github.com/NVlabs/MambaVision)] (2024.07.10)
- Parallelizing Autoregressive Generation with Variational State Space Models [[paper](https://arxiv.org/abs/2407.08415)] (2024.07.11)
- GraphMamba: An Efficient Graph Structure Learning Vision Mamba for Hyperspectral Image Classification [[paper](https://arxiv.org/abs/2407.08255)] (2024.07.11)
- ST-Mamba: Spatial-Temporal Mamba for Traffic Flow Estimation Recovery using Limited Data [[paper](https://arxiv.org/abs/2407.08558)] (2024.07.11)
- SR-Mamba: Effective Surgical Phase Recognition with State Space Model [[paper](https://arxiv.org/abs/2407.08333)] [[code](https://github.com/rcao-hk/SR-Mamba)] (2024.07.11)
- DMM: Disparity-guided Multispectral Mamba for Oriented Object Detection in Remote Sensing [[paper](https://arxiv.org/abs/2407.08132)] [[code](https://github.com/Another-0/DMM)] (2024.07.11)
- Parallelizing Autoregressive Generation with Variational State Space Models [[paper](https://arxiv.org/abs/2407.08415)] (2024.07.11)
- Hamba: Single-view 3D Hand Reconstruction with Graph-guided Bi-Scanning Mamba [[paper](https://arxiv.org/abs/2407.09646)] [[code](https://humansensinglab.github.io/Hamba/)] (2024.07.12)
- InfiniMotion: Mamba Boosts Memory in Transformer for Arbitrary Long Motion Generation [[paper](https://arxiv.org/abs/2407.10061)] [[code](https://steve-zeyu-zhang.github.io/InfiniMotion/)] (2024.07.14)
- OPa-Ma: Text Guided Mamba for 360-degree Image Out-painting [[paper](https://arxiv.org/abs/2407.10923)] (2024.07.15)
- Enhancing Temporal Action Localization: Advanced S6 Modeling with Recurrent Mechanism [[paper](https://arxiv.org/abs/2407.13078)] (2024.07.18)
- GroupMamba: Parameter-Efficient and Accurate Group Visual State Space Model [[paper](https://arxiv.org/abs/2407.13772)] [[code](https://github.com/Amshaker/GroupMamba)] (2024.07.18)
- (ICML 2024 Workshop) Investigating the Indirect Object Identification circuit in Mamba [[paper](https://arxiv.org/abs/2407.14008)] [[openreview](https://openreview.net/forum?id=lq7ZaYuwub)] [[code](https://github.com/Phylliida/investigating-mamba-ioi)] (2024.07.19)
- (TGRS 2024) MambaHSI: Spatial–Spectral Mamba for Hyperspectral Image Classification [[paper](https://ieeexplore.ieee.org/document/10604894)] [[code](https://github.com/li-yapeng/MambaHSI)] (2024.07.19)
- MxT: Mamba x Transformer for Image Inpainting [[paper](https://arxiv.org/abs/2407.16126)] (2024.07.23)
- ALMRR: Anomaly Localization Mamba on Industrial Textured Surface with Feature Reconstruction and Refinement [[paper](https://arxiv.org/abs/2407.17705)] (2024.07.25)
- VSSD: Vision Mamba with Non-Casual State Space Duality [[paper](https://arxiv.org/abs/2407.18559)] [[code](https://github.com/YuHengsss/VSSD)] (2024.07.26)
- Mamba? Catch The Hype Or Rethink What Really Helps for Image Registration [[paper](https://arxiv.org/abs/2407.19274)] [[code](https://github.com/BailiangJ/rethink-reg)] (2024.07.26)
- PhysMamba: Leveraging Dual-Stream Cross-Attention SSD for Remote Physiological Measurement [[paper](https://arxiv.org/abs/2408.01077)] (2024.08.02)
- WaveMamba: Spatial-Spectral Wavelet Mamba for Hyperspectral Image Classification [[paper](https://arxiv.org/abs/2408.01231)] (2024.08.02)
- Spatial-Spectral Morphological Mamba for Hyperspectral Image Classification [[paper](https://arxiv.org/abs/2408.01372)] (2024.08.02)
- MambaST: A Plug-and-Play Cross-Spectral Spatial-Temporal Fuser for Efficient Pedestrian Detection [[paper](https://arxiv.org/abs/2408.01037)] [[code](https://github.com/XiangboGaoBarry/MambaST)] (2024.08.02)
- JambaTalk: Speech-Driven 3D Talking Head Generation Based on Hybrid Transformer-Mamba Language Model [[paper](https://arxiv.org/abs/2408.01627)] (2024.08.03)
- DeMansia: Mamba Never Forgets Any Tokens [[paper](https://arxiv.org/abs/2408.01986)] [[code](https://github.com/catalpaaa/DeMansia)] (2024.08.04)
- Neural Architecture Search based Global-local Vision Mamba for Palm-Vein Recognition [[paper](https://arxiv.org/abs/2408.05743)] (2024.08.11)
- MetMamba: Regional Weather Forecasting with Spatial-Temporal Mamba Model [[paper](https://arxiv.org/abs/2408.06400)] (2024.08.12)
- Mamba Retriever: Utilizing Mamba for Effective and Efficient Dense Retrieval [[paper](https://arxiv.org/abs/2408.08066)] (2024.08.15)
- ColorMamba: Towards High-quality NIR-to-RGB Spectral Translation with Mamba [[paper](https://arxiv.org/abs/2408.08087)] (2024.08.15)
- OccMamba: Semantic Occupancy Prediction with State Space Models [[paper](https://arxiv.org/abs/2408.09859)] (2024.08.19)
- Multi-Scale Representation Learning for Image Restoration with State-Space Model [[paper](https://export.arxiv.org/abs/2408.10145)] (2024.08.19)
- (ICML ES-FoMo 2024) ExpoMamba: Exploiting Frequency SSM Blocks for Efficient and Effective Image Enhancement [[paper](https://arxiv.org/abs/2408.09650)] [[code](https://github.com/eashanadhikarla/ExpoMamba)] (2024.08.19)
- MambaLoc: Efficient Camera Localisation via State Space Model [[paper](https://arxiv.org/abs/2408.09680)] (2024.08.20)
- MambaDS: Near-Surface Meteorological Field Downscaling with Topography Constrained Selective State Space Modeling [[paper](https://arxiv.org/abs/2408.10854)] (2024.08.20)
- MambaEVT: Event Stream based Visual Object Tracking using State Space Model [[paper](https://arxiv.org/abs/2408.10487)] [[code](https://github.com/Event-AHU/MambaEVT)] (2024.08.20)
- OMEGA: Efficient Occlusion-Aware Navigation for Air-Ground Robot in Dynamic Environments via State Space Model [[paper](https://arxiv.org/abs/2408.10618)] [[code](https://jmwang0117.github.io/OMEGA)] (2024.08.20)
- UNetMamba: An Efficient UNet-Like Mamba for Semantic Segmentation of High-Resolution Remote Sensing Images [[paper](https://arxiv.org/abs/2408.11545)] [[code](https://github.com/EnzeZhu2001/UNetMamba)] (2024.08.21)
- Exploring Robustness of Visual State Space model against Backdoor Attacks [[paper](https://arxiv.org/abs/2408.11679)] (2024.08.21)
- MambaCSR: Dual-Interleaved Scanning for Compressed Image Super-Resolution With SSMs [[paper](https://arxiv.org/abs/2408.11758)] (2024.08.21)
- MambaOcc: Visual State Space Model for BEV-based Occupancy Prediction with Local Adaptive Reordering [[paper](https://arxiv.org/abs/2408.11464)] [[code](https://github.com/Hub-Tian/MambaOcc)] (2024.08.21)
- Modeling Time-Variant Responses of Optical Compressors with Selective State Space Models [[paper](https://arxiv.org/abs/2408.12549)] (2024.08.22)
- Scalable Autoregressive Image Generation with Mamba [[paper](https://arxiv.org/abs/2408.12245)] [[code](https://github.com/hp-l33/aim)] (2024.08.23)
- O-Mamba: O-shape State-Space Model for Underwater Image Enhancement [[paper](https://arxiv.org/abs/2408.12816)] [[code](https://github.com/chenydong/o-mamba)] (2024.08.23)
- MSFMamba: Multi-Scale Feature Fusion State Space Model for Multi-Source Remote Sensing Image Classification [[paper](https://arxiv.org/abs/2408.14255)] (2024.08.26)
- ShapeMamba-EM: Fine-Tuning Foundation Model with Local Shape Descriptors and Mamba Blocks for 3D EM Image Segmentation [[paper](https://arxiv.org/abs/2408.14114)] (2024.08.26)
- ZeroMamba: Exploring Visual State Space Model for Zero-Shot Learning [[paper](https://arxiv.org/abs/2408.14868)] [[code](https://anonymous.4open.science/r/ZeroMamba)] (2024.08.27)
- MTMamba++: Enhancing Multi-Task Dense Scene Understanding via Mamba-Based Decoders [[paper](https://arxiv.org/abs/2408.15101)] [[code](https://github.com/EnVision-Research/MTMamba)] (2024.08.27)
- DrowzEE-G-Mamba: Leveraging EEG and State Space Models for Driver Drowsiness Detection [[paper](https://arxiv.org/abs/2408.16145)] (2024.08.28)
- TrackSSM: A General Motion Predictor by State-Space Model [[paper](https://arxiv.org/abs/2409.00487)] (2024.08.31)
- A Hybrid Transformer-Mamba Network for Single Image Deraining [[paper](https://arxiv.org/abs/2409.00410)] (2024.08.31)
- Shuffle Mamba: State Space Models with Random Shuffle for Multi-Modal Image Fusion [[paper](https://arxiv.org/abs/2409.01728)] (2024.09.03)
- (IEEE MMSP 2024) Efficient Image Compression Using Advanced State Space Models [[paper](https://arxiv.org/abs/2409.02743)] (2024.09.04)
- UV-Mamba: A DCN-Enhanced State Space Model for Urban Village Boundary Identification in High-Resolution Remote Sensing Images [[paper](https://arxiv.org/abs/2409.03431)] (2024.09.05)
- DSDFormer: An Innovative Transformer-Mamba Framework for Robust High-Precision Driver Distraction Identification [[paper](https://arxiv.org/abs/2409.05587)] (2024.09.09)
- PPMamba: A Pyramid Pooling Local Auxiliary SSM-Based Model for Remote Sensing Image Semantic Segmentation [[paper](https://arxiv.org/abs/2409.06309)] (2024.09.10)
- Retinex-RAWMamba: Bridging Demosaicing and Denoising for Low-Light RAW Image Enhancement [[paper](https://arxiv.org/abs/2409.07040)] (2024.09.11)
- CollaMamba: Efficient Collaborative Perception with Cross-Agent Spatial-Temporal State Space Model [[paper](https://arxiv.org/abs/2409.07714)] (2024.09.12)
- SparX: A Sparse Cross-Layer Connection Mechanism for Hierarchical Vision Mamba and Transformer Networks [[paper](https://arxiv.org/abs/2409.09649)] [[code](https://github.com/LMMMEng/SparX)] (2024.09.15)
- (ECCV 2024) Famba-V: Fast Vision Mamba with Cross-Layer Token Fusion [[paper](https://arxiv.org/abs/2409.09808)] (2024.09.15)
- Mamba-ST: State Space Model for Efficient Style Transfer [[paper](https://arxiv.org/abs/2409.10385)] (2024.09.16)
- (IEEE Sensors Journal) AE-IRMamba: Low-Complexity Inverted Residual Mamba for Identification of Piezoelectric Ceramic and Optical Fiber Acoustic Emission Sensors Signals [[paper](https://ieeexplore.ieee.org/document/10682523)] (2024.09.17)
- Distillation-free Scaling of Large SSMs for Images and Videos [[paper](https://arxiv.org/abs/2409.11867)] (2024.09.18)
- GraspMamba: A Mamba-based Language-driven Grasp Detection Framework with Hierarchical Feature Learning [[paper](https://arxiv.org/abs/2409.14403)] (2024.09.22)
- Exploring Token Pruning in Vision State Space Models [[paper](https://arxiv.org/abs/2409.18962)] (2024.09.27)
- MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining [[paper](https://arxiv.org/abs/2410.00871)] (2024.10.01)
- Oscillatory State-Space Models [[paper](https://arxiv.org/abs/2410.03943)] (2024.10.04)
- HRVMamba: High-Resolution Visual State Space Model for Dense Prediction [[paper](https://arxiv.org/abs/2410.03174)] (2024.10.04)
- Mamba Capsule Routing Towards Part-Whole Relational Camouflaged Object Detection [[paper](https://arxiv.org/abs/2410.03987)] [[code](https://github.com/Liangbo-Cheng/mamba)] (2024.10.05)
- IGroupSS-Mamba: Interval Group Spatial-Spectral Mamba for Hyperspectral Image Classification [[paper](https://arxiv.org/abs/2410.05100)] (2024.10.07)
- Remote Sensing Image Segmentation Using Vision Mamba and Multi-Scale Multi-Frequency Feature Fusion [[paper](https://arxiv.org/abs/2410.05624)] (2024.10.08)
- MatMamba: A Matryoshka State Space Model [[paper](https://arxiv.org/abs/2410.06718)] [[code](https://github.com/ScaledFoundations/MatMamba)] (2024.10.09)
- (NeurIPS 2024) QuadMamba: Learning Quadtree-based Selective Scan for Visual State Space Model [[paper](https://arxiv.org/abs/2410.06806)] [[code](https://github.com/VISION-SJTU/QuadMamba)] (2024.10.09)
- CountMamba: Exploring Multi-directional Selective State-Space Models for Plant Counting [[paper](https://arxiv.org/abs/2410.07528)] (2024.10.10)
- GlobalMamba: Global Image Serialization for Vision Mamba [[paper](https://arxiv.org/abs/2410.10316)] (2024.10.14)
- V2M: Visual 2-Dimensional Mamba for Image Representation Learning [[paper](https://arxiv.org/abs/2410.10382)] (2024.10.14)
- MambaBEV: An efficient 3D detection model with Mamba2 [[paper](https://arxiv.org/abs/2410.12673)] (2024.10.16)
- MambaPainter: Neural Stroke-Based Rendering in a Single Step [[paper](https://arxiv.org/abs/2410.12524)] (2024.10.16)
- Rethinking Token Reduction for State Space Models [[paper](https://arxiv.org/abs/2410.14725)] (2024.10.16)
- Long-LRM: Long-sequence Large Reconstruction Model for Wide-coverage Gaussian Splats [[paper](https://arxiv.org/abs/2410.12781)] (2024.10.16)
- RemoteDet-Mamba: A Hybrid Mamba-CNN Network for Multi-modal Object Detection in Remote Sensing Images [[paper](https://arxiv.org/abs/2410.13532)] (2024.10.17)
- Quamba: A Post-Training Quantization Recipe for Selective State Space Models [[paper](https://arxiv.org/abs/2410.13229)] (2024.10.17)
- (NeurIPS 2024) MambaSCI: Efficient Mamba-UNet for Quad-Bayer Patterned Video Snapshot Compressive Imaging [[paper](https://arxiv.org/abs/2410.14214)] [[code](https://github.com/PAN083/MambaSCI)] (2024.10.18)
- Spatial-Mamba: Effective Visual State Space Models via Structure-Aware State Fusion [[paper](https://arxiv.org/abs/2410.15091)] [[code](https://github.com/EdwardChasel/Spatial-Mamba)] (2024.10.19)
- START: A Generalized State Space Model with Saliency-Driven Token-Aware Transformation [[paper](https://arxiv.org/abs/2410.16020)] [[code](https://github.com/lingeringlight/START)] (2024.10.21)
- Revealing and Mitigating the Local Pattern Shortcuts of Mamba [[paper](https://arxiv.org/abs/2410.15678)] (2024.10.21)
- SpikMamba: When SNN meets Mamba in Event-based Human Action Recognition [[paper](https://arxiv.org/abs/2410.16746)] (2024.10.22)
- Topology-aware Mamba for Crack Segmentation in Structures [[paper](https://arxiv.org/abs/2410.19894)] [[code](https://github.com/shengyu27/CrackMamba)] (2024.10.25)
- Adaptive Multi Scale Document Binarisation Using Vision Mamba [[paper](https://arxiv.org/abs/2410.22811)] (2024.10.30)
- MambaReg: Mamba-Based Disentangled Convolutional Sparse Coding for Unsupervised Deformable Multi-Modal Image Registration [[paper](https://arxiv.org/abs/2411.01399)] (2024.11.03)
- ShadowMamba: State-Space Model with Boundary-Region Selective Scan for Shadow Removal [[paper](https://arxiv.org/abs/2411.03260)] (2024.11.05)
- Towards 3D Semantic Scene Completion for Autonomous Driving: A Meta-Learning Framework Empowered by Deformable Large-Kernel Attention and Mamba Model [[paper](https://arxiv.org/abs/2411.03672)] (2024.11.06)
- MambaPEFT: Exploring Parameter-Efficient Fine-Tuning for Mamba [[paper](https://arxiv.org/abs/2411.03855)] (2024.11.06)
- SEM-Net: Efficient Pixel Modelling for image inpainting with Spatially Enhanced SSM [[paper](https://arxiv.org/abs/2411.06318)] [[code](https://github.com/ChrisChen1023/SEM-Net)] (2024.11.10)
- KMM: Key Frame Mask Mamba for Extended Motion Generation [[paper](https://arxiv.org/abs/2411.06481)] (2024.11.10)
- LFSamba: Marry SAM with Mamba for Light Field Salient Object Detection [[paper](https://arxiv.org/abs/2411.06652)] [[code](https://github.com/liuzywen/LFScribble)] (2024.11.11)
- AEROMamba: An efficient architecture for audio super-resolution using generative adversarial networks and state space models [[paper](https://arxiv.org/abs/2411.07364)] (2024.11.11)
- RAWMamba: Unified sRGB-to-RAW De-rendering With State Space Model [[paper](https://arxiv.org/abs/2411.11717)] (2024.11.18)
- EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality [[paper](https://arxiv.org/abs/2411.15241)] [[code](https://github.com/mlvlab/EfficientViM)] (2024.11.22)
- MambaIRv2: Attentive State Space Restoration [[paper](https://arxiv.org/abs/2411.15269)] [[code](https://github.com/csguoh/MambaIR)] (2024.11.22)
- Nd-BiMamba2: A Unified Bidirectional Architecture for Multi-Dimensional Data Processing [[paper](https://arxiv.org/abs/2411.15380)] [[code](https://github.com/Human9000/nd-Mamba2-torch)] (2024.11.22)
- MobileMamba: Lightweight Multi-Receptive Visual Mamba Network [[paper](https://arxiv.org/abs/2411.15941)] [[code](https://github.com/lewandofskee/MobileMamba)] (2024.11.24)
- MambaTrack: Exploiting Dual-Enhancement for Night UAV Tracking [[paper](https://arxiv.org/abs/2411.15761)] (2024.11.24)
- TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba [[paper](https://arxiv.org/abs/2411.17473)] [[code](https://github.com/xwmaxwma/TinyViM)] (2024.11.26)
- BadScan: An Architectural Backdoor Attack on Visual State Space Models [[paper](https://arxiv.org/abs/2411.17283)] [[code](https://github.com/OmSDeshmukh/BadScan)] (2024.11.26)
- SkelMamba: A State Space Model for Efficient Skeleton Action Recognition of Neurological Disorders [[paper](https://arxiv.org/abs/2411.19544)] (2024.11.29)
- MambaNUT: Nighttime UAV Tracking via Mamba and Adaptive Curriculum Learning [[paper](https://arxiv.org/abs/2412.00626)] (2024.12.01)
- (Expert Systems with Applications 2025) Evaluation of aggregate distribution uniformity using Vision Mamba-based dual networks for concrete aggregate segmentation [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417424029439)] (2024.12.05)
- Exploring Enhanced Contextual Information for Video-Level Object Tracking [[paper](https://arxiv.org/abs/2412.11023)] [[code](https://github.com/kangben258/MCITrack)] (2024.12.15)
- SegMAN: Omni-scale Context Modeling with State Space Models and Local Attention for Semantic Segmentation [[paper](https://arxiv.org/abs/2412.11890)] [[code](https://github.com/yunxiangfu2001/SegMAN)] (2024.12.16)
- GG-SSMs: Graph-Generating State Space Models [[paper](https://arxiv.org/abs/2412.12423)] (2024.12.17)
- (AAAI 2025) Robust Tracking via Mamba-based Context-aware Token Learning [[paper](https://arxiv.org/abs/2412.13611)] (2024.12.18)
- (AAAI 2025) Efficient Self-Supervised Video Hashing with Selective State Spaces [[paper](https://arxiv.org/abs/2412.14518)] [[code](https://github.com/gimpong/AAAI25-S5VH)] (2024.12.19)
- (ICASSP 2025) Trusted Mamba Contrastive Network for Multi-View Clustering [[paper](https://arxiv.org/abs/2412.16487)] [[code](https://github.com/HackerHyper/TMCN)] (2024.12.21)
- (IEEE Geoscience and Remote Sensing Letters) PPMamba: Enhancing Semantic Segmentation in Remote Sensing Imagery by SS2D [[paper](https://ieeexplore.ieee.org/document/10769411)] [[code](https://github.com/Jerrymo59/PPMambaSeg)] (2024.12.27)
- (WACV 2025) PTQ4VM: Post-Training Quantization for Visual Mamba [[paper](https://arxiv.org/abs/2412.20386)] [[code](https://github.com/YoungHyun197/ptq4vm)] (2024.12.29)
- H-MBA: Hierarchical MamBa Adaptation for Multi-Modal Video Understanding in Autonomous Driving [[paper](https://arxiv.org/abs/2501.04302)] [[code](https://github.com/Sranc3/H-MBA)] (2025.01.08)
- (Artificial Intelligence in Agriculture) PAB-Mamba-YOLO: VSSM assists in YOLO for aggressive behavior detection among weaned piglets [[paper](https://www.sciencedirect.com/science/article/pii/S2589721725000017)] (2025.01.01)
- (IEEE Geoscience and Remote Sensing Letters) A Mamba-Aware Spatial–Spectral Cross-Modal Network for Remote Sensing Classification [[paper](https://ieeexplore.ieee.org/document/10829637)] [[code](https://github.com/ru-willow/CMSI-Mamba)] (2025.01.06)
- H-MBA: Hierarchical MamBa Adaptation for Multi-Modal Video Understanding in Autonomous Driving [[paper](https://arxiv.org/abs/2501.04302)] (2025.01.08)
- MS-Temba : Multi-Scale Temporal Mamba for Efficient Temporal Action Detection [[paper](https://arxiv.org/abs/2501.06138)] [[code](https://github.com/thearkaprava/MS-Temba)] (2025.01.10)
- Mamba-MOC: A Multicategory Remote Object Counting via State Space Model [[paper](https://arxiv.org/abs/2501.06697)] (2025.01.12)
- Skip Mamba Diffusion for Monocular 3D Semantic Scene Completion [[paper](https://arxiv.org/abs/2501.07260)] (2025.01.13)
- WMamba: Wavelet-based Mamba for Face Forgery Detection [[paper](https://arxiv.org/abs/2501.09617)] (2025.01.16)
- MambaQuant: Quantizing the Mamba Family with Variance Aligned Rotation Methods [[paper](https://arxiv.org/abs/2501.13484)] [[code](https://github.com/mambaquant/mambaquant)] (2025.01.23)
- (NAACL-25) Mamba-Shedder: Post-Transformer Compression for Efficient Selective Structured State Space Models [[paper](https://arxiv.org/abs/2501.17088)] [[code](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning)] (2025.01.28)
- HSRMamba: Contextual Spatial-Spectral State Space Model for Single Hyperspectral Super-Resolution [[paper](https://arxiv.org/abs/2501.18500)] (2025.01.30)
- DAViMNet: SSMs-Based Domain Adaptive Object Detection [[paper](https://arxiv.org/abs/2502.11178)] [[code](https://github.com/enesdoruk/DAVimNet)] (2025.02.16)
- DCAMamba: Mamba-based Rapid Response DC Arc Fault Detection [[paper](https://arxiv.org/abs/2503.01264)] (2025.03.03)
- Mamba base PKD for efficient knowledge compression [[paper](https://arxiv.org/abs/2503.01727)] (2025.03.03)
- SSNet: Saliency Prior and State Space Model-based Network for Salient Object Detection in RGB-D Images [[paper](https://arxiv.org/abs/2503.02270)] (2025.03.04)
- JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba [[paper](https://arxiv.org/abs/2503.03437)] (2025.03.05)
- Spectral State Space Model for Rotation-Invariant~Visual~Representation~Learning [[paper](https://arxiv.org/abs/2503.06369)] (2025.03.09)
- Global-Aware Monocular Semantic Scene Completion with State Space Models [[paper](https://arxiv.org/abs/2503.06569)] (2025.03.09)
- HiSTF Mamba: Hierarchical Spatiotemporal Fusion with Multi-Granular Body-Spatial Modeling for High-Fidelity Text-to-Motion Generation [[paper](https://arxiv.org/abs/2503.06897)] (2025.03.10)
- MambaFlow: A Mamba-Centric Architecture for End-to-End Optical Flow Estimation [[paper](https://arxiv.org/abs/2503.07046)] (2025.03.10)
- (CVPR 2025) UniMamba: Unified Spatial-Channel Representation Learning with Group-Efficient Mamba for LiDAR-based 3D Object Detection [[paper](https://arxiv.org/abs/2503.12009)] (2025.03.15)
- (CVPR 2025) MambaIC: State Space Models for High-Performance Learned Image Compression [[paper](https://arxiv.org/abs/2503.12461)] [[code](https://github.com/AuroraZengfh/MambaIC)] (2025.03.16)
- Atlas: Multi-Scale Attention Improves Long Context Image Modeling [[paper](https://arxiv.org/abs/2503.12355)] (2025.03.16)
- MaTVLM: Hybrid Mamba-Transformer for Efficient Vision-Language Modeling [[paper](https://arxiv.org/abs/2503.13440)] (2025.03.17)
- DehazeMamba: SAR-guided Optical Remote Sensing Image Dehazing with Adaptive State Space Model [[paper](https://arxiv.org/abs/2503.13073)] [[code](https://github.com/mmic-lcl/Datasets-and-benchmark-code)] (2025.03.17)
- (AAAI 2025 Oral) Pose as a Modality: A Psychology-Inspired Network for Personality Recognition with a New Multimodal Dataset [[paper](https://arxiv.org/abs/2503.12912)] (2025.03.17)
- (ICLR 2025) State Space Model Meets Transformer: A New Paradigm for 3D Object Detection [[paper](https://arxiv.org/abs/2503.07046)] (2025.03.18)
- MamBEV: Enabling State Space Models to Learn Birds-Eye-View Representations [[paper](https://arxiv.org/abs/2503.13858)] (2025.03.18)
- DynamicVis: An Efficient and General Visual Foundation Model for Remote Sensing Image Understanding [[paper](https://arxiv.org/abs/2503.16426)] [[code](https://github.com/KyanChen/DynamicVis)] (2025.03.20)
- (CVPR 2025) Binarized Mamba-Transformer for Lightweight Quad Bayer HybridEVS Demosaicing [[paper](https://arxiv.org/abs/2503.16134)] [[code](https://github.com/Clausy9/BMTNet)] (2025.03.20)
- (CVPR 2025) SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer [[paper](https://arxiv.org/abs/2503.15934)] (2025.03.20)
- GLADMamba: Unsupervised Graph-Level Anomaly Detection Powered by Selective State Space Model [[paper](https://arxiv.org/abs/2503.17903)] [[code](https://github.com/Yali-Fu/GLADMamba)] (2025.03.23)
- EventMamba: Enhancing Spatio-Temporal Locality with State Space Models for Event-Based Video Reconstruction [[paper](https://arxiv.org/abs/2503.19721)] (2025.03.25)
- Q-MambaIR: Accurate Quantized Mamba for Efficient Image Restoration [[paper](https://arxiv.org/abs/2503.21970)] (2025.03.27)
- (ICME 2025) VADMamba: Exploring State Space Models for Fast Video Anomaly Detection [[paper](https://arxiv.org/abs/2503.21169)] (2025.03.27)
- vGamba: Attentive State Space Bottleneck for efficient Long-range Dependencies in Visual Recognition [[paper](https://arxiv.org/abs/2503.21262)] (2025.03.27)
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models [[paper](https://arxiv.org/abs/2503.22879)] (2025.03.28)
- Mesh Mamba: A Unified State Space Model for Saliency Prediction in Non-Textured and Textured Meshes [[paper](https://arxiv.org/abs/2504.01466)] (2025.04.02)
- (CVPR 2025) Mamba as a Bridge: Where Vision Foundation Models Meet Vision Language Models for Domain-Generalized Semantic Segmentation [[paper](https://arxiv.org/abs/2504.03193)] (2025.04.04)
- Dynamic Vision Mamba [[paper](https://arxiv.org/abs/2504.04787)] (2025.04.07)
- (CVPR 2025) DefMamba: Deformable Visual State Space Model [[paper](https://arxiv.org/abs/2504.05794)] (2025.04.08)
- ACMamba: Fast Unsupervised Anomaly Detection via An Asymmetrical Consensus State Space Model [[paper](https://arxiv.org/abs/2504.11781)] (2025.04.16)
- MVQA: Mamba with Unified Sampling for Efficient Video Quality Assessment [[paper](https://arxiv.org/abs/2504.16003)] (2025.04.22)
- HS-Mamba: Full-Field Interaction Multi-Groups Mamba for Hyperspectral Image Classification [[paper](https://arxiv.org/abs/2504.15612)] (2025.04.22)
- StereoMamba: Real-time and Robust Intraoperative Stereo Disparity Estimation via Long-range Spatial Dependencies [[paper](https://arxiv.org/abs/2504.17401)] (2025.04.24)
- MambaMoE: Mixture-of-Spectral-Spatial-Experts State Space Model for Hyperspectral Image Classification [[paper](https://arxiv.org/abs/2504.20509)] (2025.04.29)
- LMDepth: Lightweight Mamba-based Monocular Depth Estimation for Real-World Deployment [[paper](https://arxiv.org/abs/2505.00980)] (2025.05.02)
- WDMamba: When Wavelet Degradation Prior Meets Vision Mamba for Image Dehazing [[paper](https://arxiv.org/abs/2505.04369)]  [[code](https://github.com/SunJ000/WDMamba)] (2025.05.07
- HuMoGen Workshop) Dyadic Mamba: Long-term Dyadic Human Motion Synthesis [[paper](https://arxiv.org/abs/2505.09827)] (2025.05.14)
- (CVPR 2025) Mamba-Adaptor: State Space Model Adaptor for Visual Recognition [[paper](https://arxiv.org/abs/2505.12685)] (2025.05.19)
- FastMamba: A High-Speed and Efficient Mamba Accelerator on FPGA with Accurate Quantization [[paper](https://arxiv.org/abs/2505.18975)] (2025.05.25)
- Long-Context State-Space Video World Models [[paper](https://arxiv.org/abs/2505.20171)] (2025.05.26)
- Mamba-Driven Topology Fusion for Monocular 3-D Human Pose Estimation [[paper](https://arxiv.org/abs/2505.20611)] (2025.05.27)

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
* Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models [[paper](https://arxiv.org/abs/2408.10189)] (2024.07.14)
* ReMamba: Equip Mamba with Effective Long-Sequence Modeling [[paper](https://arxiv.org/abs/2408.15496)] (2024.08.28)
* Sparse Mamba: Reinforcing Controllability In Structural State Space Models [[paper](https://arxiv.org/abs/2409.00563)] (2024.08.31)
* DocMamba: Efficient Document Pre-training with State Space Model [[paper](https://arxiv.org/abs/2409.11887)] (2024.09.18)
* Can Mamba Always Enjoy the "Free Lunch"? [[paper](https://arxiv.org/abs/2410.03810)] (2024.10.04)
* Falcon Mamba: The First Competitive Attention-free 7B Language Model [[paper](https://arxiv.org/abs/2410.05355)] (2024.10.07)
* Stuffed Mamba: State Collapse and State Capacity of RNN-Based Long-Context Modeling [[paper](https://arxiv.org/abs/2410.07145)] (2024.10.09)
* (CIKM 2024) XRDMamba: Large-scale Crystal Material Space Group Identification with Selective State Space Model [[paper](https://dl.acm.org/doi/10.1145/3627673.3680006)] (2024.10.21)
* Bi-Mamba: Towards Accurate 1-Bit State Space Models [[paper](https://arxiv.org/abs/2411.11843)] (2024.11.18)
* Hymba: A Hybrid-head Architecture for Small Language Models [[paper](https://arxiv.org/abs/2411.13676)] (2024.11.20)
* Parameter Efficient Mamba Tuning via Projector-targeted Diagonal-centric Linear Transformation [[paper](https://arxiv.org/abs/2411.15224)] (2024.11.21)
* (AAAI 2025) DocMamba: Efficient Document Pre-training with State Space Model [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/34584)] [[code](https://github.com/Pengfei-Hu/DocMamba)] (2025.04.11)
* (ICLR 2025) LongMamba: Enhancing Mamba's Long Context Capabilities via Training-Free Receptive Field Enlargement [[paper](https://arxiv.org/abs/2504.16053)] [[code](https://github.com/GATECH-EIC/LongMamba)] (2025.04.22)
* Block-Biased Mamba for Long-Range Sequence Processing [[paper](https://arxiv.org/abs/2505.09022)] (2024.05.13)
* Efficient Unstructured Pruning of Mamba State-Space Models for Resource-Constrained Environments [[paper](https://arxiv.org/abs/2505.08299)] (2025.05.13)

<span id="head7"></span>

###  Multi-Modal
* (NeurIPS 2024) MambaTalk: Efficient Holistic Gesture Synthesis with Selective State Space Models [[paper](https://arxiv.org/abs/2403.09471)] [[code](https://github.com/kkakkkka/MambaTalk)] (2024.03.14)
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
* ML-Mamba: Efficient Multi-Modal Large Language Model Utilizing Mamba-2 [[paper](https://arxiv.org/abs/2407.19832)] (2024.06.29)
* (ACM MM 2024) MambaGesture: Enhancing Co-Speech Gesture Generation with Mamba and Disentangled Multi-Modality Fusion [[paper](https://arxiv.org/abs/2407.19976)] (2024.06.29)
* MUSE: Mamba is Efficient Multi-scale Learner for Text-video Retrieval [[paper](https://arxiv.org/abs/2408.10575)] [[code](https://github.com/hrtang22/MUSE)] (2024.08.20)
* Why mamba is effective? Exploit Linear Transformer-Mamba Network for Multi-Modality Image Fusion [[paper](https://arxiv.org/abs/2409.03223)] (2024.09.05)
* Mamba-Enhanced Text-Audio-Video Alignment Network for Emotion Recognition in Conversations [[paper](https://arxiv.org/abs/2409.05243)] (2024.09.08)
* Shaking Up VLMs: Comparing Transformers and Structured State Space Models for Vision & Language Modeling [[paper](https://arxiv.org/abs/2409.05395)] (2024.09.09)
* Mamba Fusion: Learning Actions Through Questioning [[paper](https://arxiv.org/abs/2409.11513)] (2024.09.17)
* DepMamba: Progressive Fusion Mamba for Multimodal Depression Detection [[paper](https://arxiv.org/abs/2409.15936)] (2024.09.24)
* EMMA: Empowering Multi-modal Mamba with Structural and Hierarchical Alignment [[paper](https://arxiv.org/abs/2410.05938)] (2024.10.08)
* MambaSOD: Dual Mamba-Driven Cross-Modal Fusion Network for RGB-D Salient Object Detection [[paper](https://arxiv.org/abs/2410.15015)] [[code](https://github.com/YueZhan721/MambaSOD)] (2024.10.19)
* AlignMamba: Enhancing Multimodal Mamba with Local and Global Cross-modal Alignment [[paper](https://arxiv.org/abs/2412.00833)] (2024.12.01)
* M$^3$amba: CLIP-driven Mamba Model for Multi-modal Remote Sensing Classification [[paper](https://arxiv.org/abs/2503.06446)] [[code](https://github.com/kaka-Cao/M3amba)] (2025.03.09)
* (AAAI 2025) MSAmba: Exploring Multimodal Sentiment Analysis with State Space Models [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/32120)] (2025.04.11)
* ISDrama: Immersive Spatial Drama Generation through Multimodal Prompting [[paper](https://arxiv.org/abs/2504.20630)] (2025.04.29)
* TF-Mamba: Text-enhanced Fusion Mamba with Missing Modalities for Robust Multimodal Sentiment Analysis [[paper](https://arxiv.org/abs/2505.14329)] (2025.05.20)

<span id="head29"></span>

### Spatio-Temporal

* Video Mamba Suite: State Space Model as a Versatile Alternative for Video Understanding [[paper](https://arxiv.org/abs/2403.09626)] [[code](https://github.com/opengvlab/video-mamba-suite)] (2024.03.12)
* (CVPR 2024 Precognition Workshop) VMRNN: Integrating Vision Mamba and LSTM for Efficient and Accurate Spatiotemporal Forecasting [[paper](https://arxiv.org/abs/2403.16536)] [[code](https://github.com/yyyujintang/VMRNN-PyTorch)] (2024.03.25)
* (IEEE TGRS 2024) ChangeMamba: Remote Sensing Change Detection with Spatio-Temporal State Space Model [[paper](https://arxiv.org/abs/2404.03425)] [[code](https://github.com/ChenHongruixuan/MambaCD)] (2024.04.04)
* ST-MambaSync: The Confluence of Mamba Structure and Spatio-Temporal Transformers for Precipitous Traffic Prediction [[paper](https://arxiv.org/abs/2404.15899)] (2024.04.24)
* SpoT-Mamba: Learning Long-Range Dependency on Spatio-Temporal Graphs with Selective State Spaces [[paper](https://arxiv.org/abs/2406.11244)] [[code](https://github.com/bdi-lab/SpoT-Mamba)] (2024.06.17)
* VideoMambaPro: A Leap Forward for Mamba in Video Understanding [[paper](https://arxiv.org/abs/2406.19006)] (2024.06.27)
* VFIMamba: Video Frame Interpolation with State Space Models [[paper](https://arxiv.org/abs/2407.02315)] [[code](https://github.com/MCG-NJU/VFIMamba)] (2024.07.02)
* VideoMamba: Spatio-Temporal Selective State Space Model [[paper](https://arxiv.org/abs/2407.08476)] [[code](https://github.com/jinyjelly/videomamba)] (2024.07.11)
* MambaVT: Spatio-Temporal Contextual Modeling for robust RGB-T Tracking [[paper](https://arxiv.org/abs/2408.07889)] (2024.08.15)
* DemMamba: Alignment-free Raw Video Demoireing with Frequency-assisted Spatio-Temporal Mamba [[paper](https://arxiv.org/abs/2408.10679)] (2024.08.20)
* Self-Supervised State Space Model for Real-Time Traffic Accident Prediction Using eKAN Networks [[paper](https://arxiv.org/abs/2409.05933)] [[code](http://github.com/KevinT618/SSL-eKamba)] (2024.09.09)
* (ADMA 2024) Spatial-Temporal Mamba Network for EEG-based Motor Imagery Classification [[paper](https://arxiv.org/abs/2409.09627)] (2024.09.15)
* (CCBR 2024) PhysMamba: Efficient Remote Physiological Measurement with SlowFast Temporal Difference Mamba [[paper](https://arxiv.org/abs/2409.12031)] (2024.09.18)

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
* LaMamba-Diff: Linear-Time High-Fidelity Diffusion Models Based on Local Attention and Mamba [[paper](https://arxiv.org/abs/2408.02615)] (2024.08.05)
* MADiff: Motion-Aware Mamba Diffusion Models for Hand Trajectory Prediction on Egocentric Videos [[paper](https://arxiv.org/abs/2409.02638)] [[code](https://irmvlab.github.io/madiff.github.io)] (2024.09.04)
* Mamba Policy: Towards Efficient 3D Diffusion Policy with Hybrid Selective State Models [[paper](https://arxiv.org/abs/2409.07163)] [[code](https://andycao1125.github.io/mamba_policy)] (2024.09.04)
* DiSPo: Diffusion-SSM based Policy Learning for Coarse-to-Fine Action Discretization [[paper](https://arxiv.org/abs/2409.14719)] (2024.09.23)
* (NeurIPS 2024) DiMSUM: Diffusion Mamba -- A Scalable and Unified Spatial-Frequency Method for Image Generation [[paper](https://arxiv.org/abs/2411.04168)] [[code](https://hao-pt.github.io/dimsum/)] (2024.11.06)
* (CVPR 2025 eLVM workshop) U-Shape Mamba: State Space Model for faster diffusion [[paper](https://arxiv.org/abs/2504.13499)] [https://github.com/ErgastiAlex/U-Shape-Mamba] (2025.04.18)
* Mamba-Diffusion Model with Learnable Wavelet for Controllable Symbolic Music Generation [[paper](https://arxiv.org/abs/2505.03314)] [[code](https://github.com/jinchengzhanggg/proffusion)] (2025.05.06)

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

- (IEEE BIBM 2024) MamMIL: Multiple Instance Learning for Whole Slide Images with State Space Models [[paper](https://arxiv.org/abs/2403.05160)] [[code](https://github.com/Vison307/MamMIL)] (2024.03.08)

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

- BioMamba: A Pre-trained Biomedical Language Representation Model Leveraging Mamba [[paper](https://arxiv.org/abs/2408.02600)] (2024.08.05)

- SMILES-Mamba: Chemical Mamba Foundation Models for Drug ADMET Prediction [[paper](https://arxiv.org/abs/2408.05696)] (2024.08.11)

- HMT-UNet: A hybird Mamba-Transformer Vision UNet for Medical Image Segmentation [[paper](https://arxiv.org/abs/2408.11289)] [[code](https://github.com/simzhangbest/hmt-unet)] (2024.08.21)

- Hierarchical Spatio-Temporal State-Space Modeling for fMRI Analysis [[paper](https://arxiv.org/abs/2408.13074)] (2024.08.23)

- MSVM-UNet: Multi-Scale Vision Mamba UNet for Medical Image Segmentation [[paper](https://arxiv.org/abs/2408.13735)] [[code](https://github.com/gndlwch2w/msvm-unet)] (2024.08.25)

- LoG-VMamba: Local-Global Vision Mamba for Medical Image Segmentation [[paper](https://arxiv.org/abs/2408.14415)] [[code](https://github.com/Oulu-IMEDS/LoG-VMamba)] (2024.08.26)

- Mamba2MIL: State Space Duality Based Multiple Instance Learning for Computational Pathology [[paper](https://arxiv.org/abs/2408.15032)] [[code](https://github.com/YuqiZhang-Buaa/Mamba2MIL)] (2024.08.27)

- MpoxMamba: A Grouped Mamba-based Lightweight Hybrid Network for Mpox Detection [[paper](https://arxiv.org/abs/2409.04218)] [[code](https://github.com/YubiaoYue/MpoxMamba)] (2024.09.06)

- OCTAMamba: A State-Space Model Approach for Precision OCTA Vasculature Segmentation [[paper](https://arxiv.org/abs/2409.08000)] [[code](https://github.com/zs1314/OCTAMamba)] (2024.09.12)

- Microscopic-Mamba: Revealing the Secrets of Microscopic Images with Just 4M Parameters [[paper](https://arxiv.org/abs/2409.07896)] [[code](https://github.com/zs1314/Microscopic-Mamba)] (2024.09.12)

- Learning Brain Tumor Representation in 3D High-Resolution MR Images via Interpretable State Space Models [[paper](https://arxiv.org/abs/2409.07746)] (2024.09.12)

- Tri-Plane Mamba: Efficiently Adapting Segment Anything Model for 3D Medical Images [[paper](https://arxiv.org/abs/2409.08492)] (2024.09.13)

- SPRMamba: Surgical Phase Recognition for Endoscopic Submucosal Dissection with Mamba [[paper](https://arxiv.org/abs/2409.12108)] (2024.09.18)

- MambaRecon: MRI Reconstruction with Structured State Space Models [[paper](https://arxiv.org/abs/2409.12401)] (2024.09.19)

- UU-Mamba: Uncertainty-aware U-Mamba for Cardiovascular Segmentation [[paper](https://arxiv.org/abs/2409.14305)] (2024.09.22)

- Protein-Mamba: Biological Mamba Models for Protein Function Prediction [[paper](https://arxiv.org/abs/2409.14617)] (2024.09.22)

- Multi-resolution visual Mamba with multi-directional selective mechanism for retinal disease detection [[paper](https://www.frontiersin.org/journals/cell-and-developmental-biology/articles/10.3389/fcell.2024.1484880/full)] (2024.10.11)

- UMambaAdj: Advancing GTV Segmentation for Head and Neck Cancer in MRI-Guided RT with UMamba and nnU-Net ResEnc Planner [[paper](https://arxiv.org/abs/2410.12940)] (2024.10.16)

- Taming Mambas for Voxel Level 3D Medical Image Segmentation [[paper](https://arxiv.org/abs/2410.15496)] (2024.10.20)

- R2Gen-Mamba: A Selective State Space Model for Radiology Report Generation [[paper](https://arxiv.org/abs/2410.18135)] [[code](https://github.com/YonghengSun1997/R2Gen-Mamba)] (2024.10.21)

- Bio2Token: All-atom tokenization of any biomolecular structure with Mamba [[paper](https://arxiv.org/abs/2410.19110)] (2024.10.24)

- SC-MAMBA2: Leveraging State-Space Models for Efficient Single-Cell Ultra-Long Transcriptome Modeling [[paper](https://www.biorxiv.org/content/10.1101/2024.09.30.615775)] (2024.10.26)

- KAN-Mamba FusionNet: Redefining Medical Image Segmentation with Non-Linear Modeling [[paper](https://arxiv.org/abs/2411.11926)] (2024.11.18)

- When Mamba Meets xLSTM: An Efficient and Precise Method with the XLSTM-VMUNet Model for Skin lesion Segmentation [[paper](https://arxiv.org/abs/2411.09363)] [[code](https://github.com/FangZhuoyi/XLSTM-VMUNet)] (2024.11.25)

- MambaU-Lite: A Lightweight Model based on Mamba and Integrated Channel-Spatial Attention for Skin Lesion Segmentation [[paper](https://arxiv.org/abs/2412.01405)] (2024.12.02)

- MamKPD: A Simple Mamba Baseline for Real-Time 2D Keypoint Detection [[paper](https://arxiv.org/abs/2412.01422)] (2024.12.02)

- (BIBM 2024) Mamba-SAM: An Adaption Framework for Accurate Medical Image Segmentation [[paper](https://ieeexplore.ieee.org/document/10821723)] (2024.12.03)

- MambaRoll: Physics-Driven Autoregressive State Space Models for Medical Image Reconstruction [[paper](https://arxiv.org/abs/2412.09331)] [[code](https://github.com/icon-lab/MambaRoll)] (2024.12.12)

- (NeurIPS 2024 workshop) BarcodeMamba: State Space Models for Biodiversity Analysis [[paper](https://export.arxiv.org/abs/2412.11084)] (2024.12.15)

- (NeurIPS 2024) Model Decides How to Tokenize: Adaptive DNA Sequence Tokenization with MxDNA [[paper](https://arxiv.org/abs/2412.13716)] [[code](https://github.com/qiaoqiaoLF/MxDNA)] (2024.12.18)

- KM-UNet KAN Mamba UNet for medical image segmentation [[paper](https://arxiv.org/abs/2501.02559)] [[code](https://github.com/2760613195/KM_UNet)] (2025.01.05)

- (IEEE Transactions on Geoscience and Remote Sensing) RSMamba: Biologically Plausible Retinex-Based Mamba for Remote Sensing Shadow Removal [[paper](https://ieeexplore.ieee.org/document/10833852)] (2025.01.08)

- (IEEE Internet of Things Journal) Fall-Mamba: A Multimodal Fusion and Masked Mamba-Based Approach for Fall Detection [[paper](https://ieeexplore.ieee.org/abstract/document/10833684)] [[code](https://github.com/DHUspeech/fall-mamba)] (2025.01.08)

- XFMamba: Cross-Fusion Mamba for Multi-View Medical Image Classification [[paper](https://arxiv.org/abs/2503.02619)] [[code](https://github.com/XZheng0427/XFMamba)] (2025.03.04)

- (Scientific Reports) Vision Mamba and xLSTM-UNet for medical image segmentation [[paper](https://www.nature.com/articles/s41598-025-88967-5)] (2025.03.10)

- Leveraging State Space Models in Long Range Genomics [[paper](https://arxiv.org/abs/2504.06304)] (2025.04.07)

- (ISBI 2025) OmniMamba4D: Spatio-temporal Mamba for longitudinal CT lesion segmentation [[paper](https://arxiv.org/abs/2504.09655)] (2025.04.13)

- Mamba-Based Ensemble learning for White Blood Cell Classification [[paper](https://arxiv.org/abs/2504.11438)] [[code](https://github.com/LewisClifton/Mamba-WBC-Classification)] (2025.04.15)

- Mamba Based Feature Extraction And Adaptive Multilevel Feature Fusion For 3D Tumor Segmentation From Multi-modal Medical Image [[paper](https://arxiv.org/abs/2504.21281)]  (2025.04.30)

- (TMI 2025) Mamba-Sea: A Mamba-based Framework with Global-to-Local Sequence Augmentation for Generalizable Medical Image Segmentation [[paper](https://ieeexplore.ieee.org/document/10980210)] [[code](https://github.com/orange-czh/Mamba-Sea)] (2025.04.30)

- MambaControl: Anatomy Graph-Enhanced Mamba ControlNet with Fourier Refinement for Diffusion-Based Disease Trajectory Prediction [[paper](https://arxiv.org/abs/2505.09965https://arxiv.org/abs/2505.09965)] (2025.05.15)

- Graph Mamba for Efficient Whole Slide Image Understanding [[paper](https://arxiv.org/abs/2505.17457)] (2025.05.23)

<span id="head9"></span>

### Tabular Data
* (MIPR 2024) MambaTab: A Simple Yet Effective Approach for Handling Tabular Data [[paper](https://arxiv.org/abs/2401.08867)] (2024.01.16)
* Mambular: A Sequential Model for Tabular Deep Learning [[paper](https://arxiv.org/abs/2401.08867)] [[code](https://github.com/basf/mamba-tabular)] (2024.08.12)
* (BIBM 2024) FT-Mamba: A Novel Deep Learning Model for Efficient Tabular Regression [[paper](https://ieeexplore.ieee.org/abstract/document/10822174)] (2024.12.03)

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
* DyGMamba: Efficiently Modeling Long-Term Temporal Dependency on Continuous-Time Dynamic Graphs with State Space Models [[paper](https://arxiv.org/abs/2408.04713)] (2024.08.08)
* DyG-Mamba: Continuous State Space Modeling on Dynamic Graphs [[paper](https://arxiv.org/abs/2408.06966)] (2024.08.13)
* Geometry Informed Tokenization of Molecules for Language Model Generation [[paper](https://arxiv.org/abs/2408.10120)] (2024.08.19)
* Mamba Meets Financial Markets: A Graph-Mamba Approach for Stock Price Prediction [[paper](https://arxiv.org/abs/2410.03707)] [[code](http://github.com/Ali-Meh619/SAMBA)] (2024.09.26)
* DG-Mamba: Robust and Efficient Dynamic Graph Structure Learning with Selective State Space Models [[paper](https://arxiv.org/abs/2412.08160)] (2024.12.11)
* (AAAI 2025) Selective Visual Prompting in Vision Mamba [[paper](https://arxiv.org/abs/2412.08947)] (2024.12.12)
* XYScanNet: An Interpretable State Space Model for Perceptual Image Deblurring [[paper](https://arxiv.org/abs/2412.10338)] (2024.12.13)
* GG-SSMs: Graph-Generating State Space Models [[paper](https://arxiv.org/abs/2412.12423)] (2024.12.17)
* (AAAI 2025) MOL-Mamba: Enhancing Molecular Representation with Structural & Electronic Insights [[paper](https://arxiv.org/abs/2412.16483)] (2024.12.21)
* Mamba-Based Graph Convolutional Networks: Tackling Over-smoothing with Selective State Space [[paper](https://arxiv.org/abs/2501.15461)] (2025.01.26)

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
* Unleashing the Potential of Mamba: Boosting a LiDAR 3D Sparse Detector by Using Cross-Model Knowledge Distillation [[paper](https://arxiv.org/abs/2409.11018)] (2024.09.17)
* (AAAI 2025) FlowMamba: Learning Point Cloud Scene Flow with Global Motion Propagation [[paper](https://arxiv.org/abs/2412.17366)] (2024.12.23)
* (WACV 2025, workshop) MambaTron: Efficient Cross-Modal Point Cloud Enhancement using Aggregate Selective State Space Modeling [[paper](https://arxiv.org/abs/2501.16384)] (2025.01.25)
* Spectral Informed Mamba for Robust Point Cloud Processing [[paper](https://arxiv.org/abs/2503.04953)] (2025.03.06)
* TFDM: Time-Variant Frequency-Based Point Cloud Diffusion with Mamba [[paper](https://arxiv.org/abs/2503.13004)] (2025.03.17)
* PillarMamba: Learning Local-Global Context for Roadside Point Cloud via Hybrid State Space Model [[paper](https://arxiv.org/abs/2505.05397)] (2025.05.08)
* SRMamba: Mamba for Super-Resolution of LiDAR Point Clouds [[paper](https://arxiv.org/abs/2505.10601)] (2025.05.15)
* Latent Mamba Operator for Partial Differential Equations [[paper](https://arxiv.org/abs/2505.19105)] (2025.05.25)
* (CVPR 2025) PMA: Towards Parameter-Efficient Point Cloud Understanding via Point Mamba Adapter [[paper](https://arxiv.org/abs/2505.20941)] [[code](https://github.com/zyh16143998882/PMA)] (2025.05.27)

<span id="head12"></span>
### Time Series
* Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling [[paper](https://arxiv.org/abs/2402.10211)] [[code](https://github.com/raunaqbhirangi/hiss)] [[homepage](https://hiss-csp.github.io/)] (2024.02.15)
* MambaStock: Selective state space model for stock prediction [[paper](https://arxiv.org/abs/2402.18959)] [[code](https://github.com/zshicode/MambaStock)] (2024.02.29)
* MambaLithium: Selective state space model for remaining-useful-life, state-of-health, and state-of-charge estimation of lithium-ion batteries [[paper](https://arxiv.org/abs/2403.05430)] [[code](https://github.com/zshicode/MambaLithium)] (2024.03.08)
* TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting [[paper](https://arxiv.org/abs/2403.09898)] [[code](https://github.com/atik-ahamed/timemachine)] (2024.03.14)
* Is Mamba Effective for Time Series Forecasting? [[paper](https://arxiv.org/abs/2403.11144)] [[code](https://github.com/wzhwzhwzh0921/S-D-Mamba)] (2024.03.17)
* SiMBA: Simplified Mamba-Based Architecture for Vision and Multivariate Time series [[paper](https://arxiv.org/abs/2403.15360)] [[code](https://github.com/badripatro/Simba)] (2024.03.22)
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
* EEG-SSM: Leveraging State-Space Model for Dementia Detection [[paper](https://arxiv.org/abs/2407.17801)] (2024.07.25)
* Simplified Mamba with Disentangled Dependency Encoding for Long-Term Time Series Forecasting [[paper](https://arxiv.org/abs/2408.12068)] (2024.08.22)
* Mamba or Transformer for Time Series Forecasting? Mixture of Universals (MoU) Is All You Need [[paper](https://arxiv.org/abs/2408.15997)] [[code](https://github.com/lunaaa95/mou/)] (2024.08.28)
* Integration of Mamba and Transformer -- MAT for Long-Short Range Time Series Forecasting with Application to Weather Dynamics [[paper](https://arxiv.org/abs/2409.08530)] (2024.09.13)
* SITSMamba for Crop Classification based on Satellite Image Time Series [[paper](https://arxiv.org/abs/2409.09673)] (2024.09.15)
* Mamba Meets Financial Markets: A Graph-Mamba Approach for Stock Price Prediction [[paper](https://arxiv.org/abs/2410.03707)] [[code](http://github.com/Ali-Meh619/SAMBA)] (2024.09.26)
* A SSM is Polymerized from Multivariate Time Series [[paper](https://arxiv.org/abs/2409.20310)] [[code](https://github.com/Joeland4/Poly-Mamba)] (2024.09.30)
* TIMBA: Time series Imputation with Bi-directional Mamba Blocks and Diffusion models [[paper](https://arxiv.org/abs/2410.05916)] (2024.10.08)
* Mamba4Cast: Efficient Zero-Shot Time Series Forecasting with State Space Models [[paper](https://arxiv.org/abs/2410.09385)] (2024.10.12)
* UmambaTSF: A U-shaped Multi-Scale Long-Term Time Series Forecasting Method Using Mamba [[paper](https://arxiv.org/abs/2410.11278)] (2024.10.15)
* MASER: Enhancing EEG Spatial Resolution With State Space Modeling [[paper](https://ieeexplore.ieee.org/document/10720236)] (2024.10.16)
* MambaCPU: Enhanced Correlation Mining with State Space Models for CPU Performance Prediction [[paper](https://arxiv.org/abs/2410.19297)] [[code](https://github.com/wredan/mamba-cpu)] (2024.10.25)
* SambaMixer: State of Health Prediction of Li-ion Batteries using Mamba State Space Models [[paper](https://arxiv.org/abs/2411.00233)] [[code](https://github.com/sascha-kirch/samba-mixer)] (2024.10.25)
* A Mamba Foundation Model for Time Series Forecasting [[paper](https://arxiv.org/abs/2411.02941)] (2024.11.05)
* BiT-MamSleep: Bidirectional Temporal Mamba for EEG Sleep Staging [[paper](https://arxiv.org/abs/2411.01589)] (2024.11.21)
* (BIBM 2024) EMO-Mamba: Multimodal Selective Structured State Space Model for Depression Detection [[paper](https://ieeexplore.ieee.org/abstract/document/10822789)] (2024.12.03)
* Attention Mamba: Time Series Modeling with Adaptive Pooling Acceleration and Receptive Field Enhancements [[paper](https://arxiv.org/abs/2504.02013)] (2025.04.02)
* ms-Mamba: Multi-scale Mamba for Time-Series Forecasting [[paper](https://arxiv.org/abs/2504.07654)] (2025.04.10)
* Byte Pair Encoding for Efficient Time Series Forecasting [[paper](https://arxiv.org/abs/2505.14411)] (2025.05.20)
* FR-Mamba: Time-Series Physical Field Reconstruction Based on State Space Model [[paper](https://arxiv.org/abs/2505.16083)] (2025.05.21)
* (ICML 2025) TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state [[paper](https://arxiv.org/abs/2505.20774)] [[code](https://github.com/xwmaxwma/TimePro)] (2025.05.27)

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
* SELD-Mamba: Selective State-Space Model for Sound Event Localization and Detection with Source Distance Estimation [[paper](https://arxiv.org/abs/2408.05057)] (2024.08.09)
* MusicMamba: A Dual-Feature Modeling Approach for Generating Chinese Traditional Music with Modal Precision [[paper](https://arxiv.org/abs/2409.02421)] (2024.09.04)
* TF-Mamba: A Time-Frequency Network for Sound Source Localization [[paper](https://arxiv.org/abs/2409.05034)] (2024.09.08)
* A Two-Stage Band-Split Mamba-2 Network for Music Separation [[paper](https://arxiv.org/abs/2409.06245)] [[code](https://github.com/baijinglin/TS-BSmamba2)] (2024.09.10)
* Rethinking Mamba in Speech Processing by Self-Supervised Models [[paper](https://arxiv.org/abs/2409.07273)] (2024.09.11)
* MambaFoley: Foley Sound Generation using Selective State-Space Models [[paper](https://arxiv.org/abs/2409.09162)] (2024.09.13)
* Wave-U-Mamba: An End-To-End Framework For High-Quality And Efficient Speech Super Resolution [[paper](https://arxiv.org/abs/2409.09337)] (2024.09.14)
* Leveraging Joint Spectral and Spatial Learning with MAMBA for Multichannel Speech Enhancement [[paper](https://arxiv.org/abs/2409.10376)] (2024.09.16)
* DeFT-Mamba: Universal Multichannel Sound Separation and Polyphonic Audio Classification [[paper](https://arxiv.org/abs/2409.12413)] (2024.09.19)
* (ICASSP 2025) Mamba-based Segmentation Model for Speaker Diarization [[paper](https://arxiv.org/abs/2410.06459)] [[code](https://github.com/nttcslab-sp/mamba-diarization)] (2024.10.09)
* CleanUMamba: A Compact Mamba Network for Speech Denoising using Channel Pruning [[paper](https://arxiv.org/abs/2410.11062)] (2024.10.14)
* (IEEE TCE) Selective State Space Model for Monaural Speech Enhancement [[paper](https://arxiv.org/abs/2411.06217)] (2024.11.09)
* (SLT 2024) Mamba-based Decoder-Only Approach with Bidirectional Speech Modeling for Speech Recognition [[paper](https://arxiv.org/abs/2411.06968)] [[code](https://github.com/YoshikiMas/madeon-asr)] (2024.11.11)
* XLSR-Mamba: A Dual-Column Bidirectional State Space Model for Spoofing Attack Detection [[paper](https://arxiv.org/abs/2411.10027)] (2024.11.15)
* MASV: Speaker Verification with Global and Local Context Mamba [[paper](https://arxiv.org/abs/2412.10989)] (2024.12.14)
* (Interspeech 2025) Universal Speech Enhancement with Regression and Generative Mamba [[paper](https://arxiv.org/abs/2505.21198)] (2025.05.27)

<span id="head14"></span>

### Recommendation 
* (RelKD@KDD 2024) Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models [[paper](https://arxiv.org/abs/2403.05430)] [[code](https://github.com/chengkai-liu/mamba4rec)] (2024.03.06)
* Uncovering Selective State Space Model's Capabilities in Lifelong Sequential Recommendation [[paper](https://arxiv.org/abs/2403.16371)] [[code](https://github.com/nancheng58/Rec-Mamba)] (2024.03.25)
* EchoMamba4Rec: Harmonizing Bidirectional State Space Models with Spectral Filtering for Advanced Sequential Recommendation [[paper](https://arxiv.org/abs/2403.16371)] (2024.06.04)
* (CIKM 2024) Behavior-Dependent Linear Recurrent Units for Efficient Sequential Recommendation [[paper](https://arxiv.org/abs/2406.12580)] [[code](https://github.com/chengkai-liu/RecBLR)] (2024.06.18)
* MLSA4Rec: Mamba Combined with Low-Rank Decomposed Self-Attention for Sequential Recommendation [[paper](https://arxiv.org/abs/2407.13135)] (2024.07.18)
* MaTrRec: Uniting Mamba and Transformer for Sequential Recommendation [[paper](https://arxiv.org/abs/2407.19239)] [[code](https://github.com/Unintelligentmumu/MaTrRec)] (2024.06.27)
* Bidirectional Gated Mamba for Sequential Recommendation [[paper](https://arxiv.org/abs/2408.11451)] [[code](https://github.com/ziwliu-cityu/SIMGA)] (2024.08.21)
* SSD4Rec: A Structured State Space Duality Model for Efficient Sequential Recommendation [[paper](https://arxiv.org/abs/2409.01192)] [[code](https://github.com/ZhangYifeng1995/SSD4Rec)] (2024.09.02)
* TiM4Rec: An Efficient Sequential Recommendation Model Based on Time-Aware Structured State Space Duality Model [[paper](https://arxiv.org/abs/2409.16182)] [[code](https://github.com/AlwaysFHao/TiM4Rec)] (2024.09.24)
* SS4Rec: Continuous-Time Sequential Recommendation with State Space Models [[paper](https://arxiv.org/abs/2502.08132)] [[code](https://github.com/XiaoWei-i/SS4Rec)] (2025.02.12)
* HMamba: Hyperbolic Mamba for Sequential Recommendation [[paper](https://arxiv.org/abs/2505.09205)] (2025.05.14)

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
* Context-aware Mamba-based Reinforcement Learning for social robot navigation [[paper](https://arxiv.org/abs/2408.02661)] (2024.08.05)
* PTrajM: Efficient and Semantic-rich Trajectory Learning with Pretrained Trajectory-Mamba [[paper](https://arxiv.org/abs/2408.04916)] (2024.08.09)
* Mamba as a motion encoder for robotic imitation learning [[paper](https://arxiv.org/abs/2409.02636)] (2024.09.04)
* DiSPo: Diffusion-SSM based Policy Learning for Coarse-to-Fine Action Discretization [[paper](https://arxiv.org/abs/2409.14719)] (2024.09.23)
* Uncertainty Representations in State-Space Layers for Deep Reinforcement Learning under Partial Observability [[paper](https://arxiv.org/abs/2409.16824)] (2024.09.25)
* Decision Transformer vs. Decision Mamba: Analysing the Complexity of Sequential Decision Making in Atari Games [[paper](https://arxiv.org/abs/2412.00725)] (2024.12.01)
* Learning Local Causal World Models with State Space Models and Attention [[paper](https://arxiv.org/abs/2505.02074)] (2025.05.04)
* MTIL: Encoding Full History with Mamba for Temporal Imitation Learning [[paper](https://arxiv.org/abs/2505.12410)] [[code](https://github.com/yulinzhouZYL/MTIL)] (2024.05.18)
* Mamba as Decision Maker: Exploring Multi-scale Sequence Modeling in Offline Reinforcement Learning [[paper](https://arxiv.org/abs/2406.02013)] [[code](https://github.com/AndyCao1125/MambaDM)] (2024.06.04)

<span id="head16"></span>
###  Survey
* State Space Model for New-Generation Network Alternative to Transformers: A Survey [[paper](https://arxiv.org/abs/2404.09516)] [[project](https://github.com/Event-AHU/Mamba_State_Space_Model_Paper_List)] (2024.04.15)
* Mamba-360: Survey of State Space Models as Transformer Alternative for Long Sequence Modelling: Methods, Applications, and Challenges [[paper](https://arxiv.org/abs/2404.16112)] [[project](https://github.com/badripatro/mamba360)] (2024.04.24)
* A Survey on Visual Mamba [[paper](https://arxiv.org/abs/2404.15956)] (2024.04.24)
* A Survey on Vision Mamba: Models, Applications and Challenges [[paper](https://arxiv.org/abs/2404.18861)] [[project](https://github.com/Ruixxxx/Awesome-Vision-Mamba-Models)] (2024.04.29)
* Vision Mamba: A Comprehensive Survey and Taxonomy [[paper](https://arxiv.org/abs/2405.04404)] [[project](https://github.com/lx6c78/Vision-Mamba-A-Comprehensive-Survey-and-Taxonomy)] (2024.05.07)
* Surveying Image Segmentation Approaches in Astronomy [[paper](https://arxiv.org/abs/2405.14238)] (2024.05.23)
* Computation-Efficient Era: A Comprehensive Survey of State Space Models in Medical Image Analysis [[paper](https://arxiv.org/abs/2406.03430)] [[project](https://github.com/xmindflow/Awesome_Mamba)] (2024.05.07)
* Surveying Image Segmentation Approaches in Astronomy [[paper](https://arxiv.org/abs/2408.01129)] (2024.08.02)
* A Survey on Mamba Architecture for Vision Applications [[paper](https://arxiv.org/abs/2502.07161)] (2025.02.11)
* Vision Mamba in Remote Sensing: A Comprehensive Survey of Techniques, Applications and Outlook [[paper](https://arxiv.org/abs/2505.00630)] (2025.05.01)

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
  * mamba2-minimal: A minimal, single-file implementation of the Mamba-2 model in PyTorch. [[URL](https://github.com/tommyip/mamba2-minimal)]
  * mamba.py: An efficient Mamba implementation in PyTorch and MLX. [[URL](https://github.com/alxndrTL/mamba.py)]
  * mamba.c: Inference of Mamba models in pure C and CUDA. [[URL](https://github.com/kroggen/mamba.c)]

<span id="head22"></span>
### Other Awesome Mamba List
* AvivBick/awesome-ssm-ml [[URL](https://github.com/AvivBick/awesome-ssm-ml)]
* yyyujintang/Awesome-Mamba-Papers [[URL](https://github.com/yyyujintang/Awesome-Mamba-Papers)]
* pengzhangzhi/Awesome-Mamba [[URL](https://github.com/pengzhangzhi/Awesome-Mamba)]

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

<span id="head28"></span>

## Citation

If you find this repository useful, please consider citing our paper:

```latex
@article{cai2024mambats,
  title={MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting},
  author={Cai, Xiuding and Zhu, Yaoyao and Wang, Xueyao and Yao, Yu},
  journal={arXiv preprint arXiv:2405.16440},
  year={2024}
}
```

<span id="head26"></span>

##  Acknowledgement

Thanks the template from [Awesome-Visual-Transformer](https://github.com/dk-liang/Awesome-Visual-Transformer) and [Awesome State-Space Resources for ML](https://github.com/AvivBick/awesome-ssm-ml/tree/main)
