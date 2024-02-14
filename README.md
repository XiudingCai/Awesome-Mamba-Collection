# Awesome-Mamba-Collection

![Awesome](https://awesome.re/badge.svg) ![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) ![Stars](https://img.shields.io/github/stars/XiudingCai/Awesome-Mamba-Collection)

Welcome to Awesome Mamba Resources! This repository is a curated collection of papers, tutorials, videos, and other valuable resources related to Mamba. Whether you're a beginner or an experienced user, this collection aims to provide a comprehensive reference for all things Mamba. Explore the latest research papers, dive into helpful tutorials, and discover insightful videos to enhance your understanding and proficiency in Mamba. Join us in this open collaboration to foster knowledge sharing and empower the Mamba community. Let's embark on an exciting journey with Mamba!

Feel free to modify and customize this introduction according to your preferences. Good luck with your repository! If you have any other questions, feel free to ask. -by ChatGPT

## Papers

### Architecture

* Mamba: Linear-Time Sequence Modeling with Selective State Spaces [[paper](https://arxiv.org/abs/2312.00752)] [[code](https://github.com/state-spaces/mamba)] (2023.12.01)
* MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts [[paper](https://arxiv.org/abs/2401.04081)] [[code](https://github.com/llm-random/llm-random)] (2024.01.08)
* BlackMamba: Mixture of Experts for State-Space Models [[paper](https://arxiv.org/abs/2402.01771)] [[code](https://github.com/zyphra/blackmamba)] (2024.02.01)

### Vision

- Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model [[paper](https://arxiv.org/abs/2401.09417)] [[code](https://github.com/hustvl/Vim)] (2024.01.17)
- VMamba: Visual State Space Model [[paper](https://arxiv.org/abs/2401.10166)] [[code](https://github.com/MzeroMiko/VMamba)] (2024.01.18)
- U-shaped Vision Mamba for Single Image Dehazing [[paper](https://arxiv.org/abs/2402.04139)] (2024.02.06)
- Scalable Diffusion Models with State Space Backbone [[paper](https://arxiv.org/abs/2402.05608)] [[code](https://github.com/feizc/dis)] (2024.02.08)
- Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data [[paper](https://arxiv.org/abs/2402.05892)] (2024.02.08)

### Language

* MambaByte: Token-free Selective State Space Model [[paper](https://arxiv.org/abs/2401.13660)] [[code](https://github.com/lucidrains/MEGABYTE-pytorch)] (2024.01.24)
* Is Mamba Capable of In-Context Learning? [[paper](https://arxiv.org/abs/2402.03170)] (2024.02.05)
* Can Mamba Learn How to Learn? A Comparative Study on In-Context Learning Tasks  [[paper](https://arxiv.org/abs/2402.04248)] (2024.02.06)

### Medical

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

### Tabular Data

* MambaTab: A Simple Yet Effective Approach for Handling Tabular Data [[paper](https://arxiv.org/abs/2401.08867)] (2024.01.16)

### Graph

* Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces [[paper](https://arxiv.org/abs/2402.00789)] [[code](https://github.com/bowang-lab/Graph-Mamba)] (2024.02.01)

## Tutorials

### Blogs

* The Annotated S4 [[URL](https://srush.github.io/annotated-s4/#part-1b-addressing-long-range-dependencies-with-hippo)]

### Videos

* S4: Efficiently Modeling Long Sequences with Structured State Spaces | Albert Gu [[URL](https://www.youtube.com/watch?v=luCBXCErkCs)]
* Mamba and S4 Explained: Architecture, Parallel Scan, Kernel Fusion, Recurrent, Convolution, Math [[URL](https://www.youtube.com/watch?v=8Q_tqwpTpVU)]

### Books

* Linear State‐Space Control Systems [[URL](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470117873)]
* Modeling sequences with structured state spaces [[URL](https://searchworks.stanford.edu/view/14784021)]

### Codes

* The official Mamba Repository is currently only available for Linux. [[URL](https://github.com/state-spaces/mamba)]
* If you are searching for a runnable implementation not focused on speed,
  * mamba-minimal: Simple, minimal implementation of the Mamba SSM in one file of PyTorch. [[URL](https://github.com/johnma2006/mamba-minimal/tree/master)] 
  * mamba.py: An efficient Mamba implementation in PyTorch and MLX. [[URL](https://github.com/alxndrTL/mamba.py)]

### Other Awesome Mamba List

* awesome-ssm-ml [[URL](https://github.com/AvivBick/awesome-ssm-ml)]
* Awesome-Mamba: Collect papers about Mamba [[URL](https://github.com/caojiaolong/Awesome-Mamba)]

## Contributions

🎉 Thank you for considering contributing to our Awesome Mamba Collection repository! 🚀

### Contribute in 3 Steps:

1. **Fork the Repo:** Fork this repo to your GitHub account.
2. **Edit Content:** Contribute by adding new resources or improving existing content in the `README.md` file.
3. **Create a Pull Request:** Open a pull request (PR) from your branch to the main repository.

### Guidelines

- Follow the existing structure and formatting.
- Ensure added resources are relevant to State Space Models in Machine Learning.
- Verify that links work correctly.

## Acknowledgement

Thanks the template from [Awesome-Visual-Transformer](https://github.com/dk-liang/Awesome-Visual-Transformer) and [Awesome State-Space Resources for ML](https://github.com/AvivBick/awesome-ssm-ml/tree/main)

