# Awesome Large Pre-trained Model-based OOD Detection[![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)

A curated list of awesome large pre-trained model-based OOD detection resources. Your contributions are always welcome!

## Content

- [Zero-shot OOD Detection Approaches](#zero-shot-ood-detection-approaches)
  - [Transitional work](#Transitional-work)
  - [DIffusion-based](#DIffusion-based)
  - [VLM-based](#VLM-based)
  - [LLM-based](#LLM-based)
  - [Zero-shot  ID detection](#Zero-shot-ID-detection)

- [Few-shot OOD Detection Approaches](#few-shot-ood-detection-approaches)
  - [Study](#Study)
  - [Meta-learning-based](#Meta-learning-based)
  - [Fine-tuning-based](#Fine-tuning-based)

- [Full-shot OOD Detection Approaches](#full-shot-ood-detection-approaches)



## Zero-shot OOD Detection Approaches

### **Transitional work**

- `One-Class-Anything`[Ge et al.]\[arxiv 2023]**Building One-class Detector for Anything: Open-vocabulary Zero-shot OOD Detection Using Text-image Models** [[PDF](https://arxiv.org/abs/2305.17207)]\[[CODE](https://github.com/gyhandy/One-Class-Anything)] 

- `...`[Fort et al.]\[NeurIPS 2021]**Exploring the Limits of Out-of-Distribution Detection**[[PDF](https://openreview.net/forum?id=j5NrN8ffXC)]\[[CODE](https://github.com/stanislavfort/exploring_the_limits_of_OOD_detection)]

### **VLM-based**

#### DIffusion-based

- `RONIN`[Nguyen et al.]\[arxiv 2024]**Zero-Shot Object-Level Out-of-Distribution Detection with Context-Aware Inpainting**[[PDF](https://arxiv.org/abs/2402.03292)]

#### CLIP-based

- `NegLabel`[Jiang et al.]\[ICLR 2024]**Negative Label Guided OOD Detection with Pretrained Vision-Language Models**[[PDF](https://openreview.net/forum?id=xUO1HXz4an)]\[[CODE](https://github.com/XueJiang16/NegLabel)]
- `CLIPN`[Wang et al.]\[ICCV 2022]**CLIPN for Zero-Shot OOD Detection: Teaching CLIP to Say No**[[PDF](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_CLIPN_for_Zero-Shot_OOD_Detection_Teaching_CLIP_to_Say_No_ICCV_2023_paper.html)]\[[CODE](https://github.com/xmed-lab/CLIPN)]
- `MCM`[Ming et al.]\[NeurIPS 2022]**Delving into Out-of-Distribution Detection with Vision-Language Representations** [[PDF](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e43a33994a28f746dcfd53eb51ed3c2d-Abstract-Conference.html)]\[[CODE](https://github.com/deeplearning-wisc/MCM)]
- `ZOC` [S'Esmaeilpour et al.]\[AAAI 2022]**Zero-Shot Out-of-Distribution Detection Based on the Pre-trained Model CLIP** [[PDF](https://arxiv.org/abs/2109.02748)]\[[CODE](https://github.com/sesmae/ZOC)]

### **LLM-based**

- `...`[Salimben]\[arxiv 2024]**Beyond fine-tuning: LoRA modules boost near-OOD detection and LLM security**[[PDF](https://dlsp2024.ieee-security.org/papers/dls2024-final19.pdf)]

- `VI-OOD`[Zhan et al.]\[arxiv 2024]**VI-OOD: A Unified Representation Learning Framework for Textual**[[PDF](https://arxiv.org/pdf/2404.06217.pdf)][[CODE](https://github.com/liam0949/LLM-OOD)]

- `...`[Bendou et al.]\[arxiv 2024]**LLM meets Vision-Language Models for Zero-Shot One-Class Classification**[[PDF](https://arxiv.org/pdf/2404.00675.pdf)]\[[CODE](https://github.com/ybendou/one-class-ZS)]

- `...`[Liu et al.]\[arxiv 2024]**How Good Are Large Language Models at Out-of-Distribution Detection?**[[PDF](https://arxiv.org/abs/2308.10261)]

- `...`[Huang et al.]\[arxiv 2024]**Out-of-Distribution Detection Using Peer-Class Generated by Large Language Model**[[PDF](https://synthical.com/article/baff2745-d4fd-48b7-ba8d-eeca7febf84d)]

- `...`[Dai el al.]\[EMNLP 2023]**Exploring Large Language Models for Multi-Modal Out-of-Distribution Detection**[[PDF](https://aclanthology.org/2023.findings-emnlp.351/)]

### Zero-shot  ID detection

- `GL-MCM`[Miyai et al.]\[arxiv 2023]**Zero-Shot In-Distribution Detection in Multi-Object Settings Using Vision-Language Foundation Models**[[PDF](https://arxiv.org/abs/2304.04521)]\[[CODE](https://github.com/AtsuMiyai/GL-MCM)]



## Few-shot OOD Detection Approaches

### Study

- `...`[Kim et al.]\[ICEIC 2024]**Comparison of Out-of-Distribution Detection Performance of CLIP-based Fine-Tuning Methods**
- `...`[Ming et al.]\[IJCV 2023]**How Does Fine-Tuning Impact Out-of-Distribution Detection for Vision-Language Models?**[[PDF](https://link.springer.com/article/10.1007/s11263-023-01895-7)]
- `DSGF`[Dong et al.]\[arxiv 2023]**Towards Few-shot Out-of-Distribution Detection**[[PDF](https://arxiv.org/abs/2311.12076)]
- `...`[Fort et al.]\[NeurIPS 2021]**Exploring the Limits of Out-of-Distribution Detection**[[PDF](https://openreview.net/forum?id=j5NrN8ffXC)]\[[CODE](https://github.com/stanislavfort/exploring_the_limits_of_OOD_detection)]

### Meta-learning-based

- `HyperMix`[Mehta et al.]\[WACV 2024]**HyperMix: Out-of-Distribution Detection and Classification in Few-Shot Settings**[[PDF](https://openaccess.thecvf.com/content/WACV2024/papers/Mehta_HyperMix_Out-of-Distribution_Detection_and_Classification_in_Few-Shot_Settings_WACV_2024_paper.pdf)]

- `OOD-MAML`[Jeong et al.]\[NeurIPS 2020]**OOD-MAML: Meta-Learning for Few-Shot Out-of-Distribution Detection and Classification**[[PDF](https://proceedings.neurips.cc/paper/2020/hash/28e209b61a52482a0ae1cb9f5959c792-Abstract.html)]\[[CODE](https://github.com/twj-KAIST/OOD-MAML)]

### Fine-tuning-based

- `NegPrompt`[Li et al.]\[CVPR 2024]**Learning Transferable Negative Prompts for Out-of-Distribution Detection**[[PDF](https://arxiv.org/abs/2404.03248)]\[[CODE](https://github.com/mala-lab/negprompt)]

- `LSN`[Nie et al.]\[ICLR 2024]**Out-of-Distribution Detection with Negative Prompts**[[PDF](https://openreview.net/forum?id=nanyAujl6e&referrer=%5Bthe%20profile%20of%20Jun%20Nie%5D(%2Fprofile%3Fid%3D~Jun_Nie1))]\[[CODE]()]

- `ID-like`[Bai et al.]\[CVPR 2024]**ID-like Prompt Learning for Few-Shot Out-of-Distribution Detection**[[PDF](https://arxiv.org/abs/2311.15243)]

- `LoCoOp`[Miyai et al.]\[NeurIPS 2023]**LoCoOp:Few-Shot Out-of-Distribution Detection via Prompt Learning**[[PDF](https://openreview.net/forum?id=UjtiLdXGMC)]\[[CODE](https://github.com/AtsuMiyai/LoCoOp)]
- `DSGF`[Dong et al.]\[arxiv 2023]**Towards Few-shot Out-of-Distribution Detection**[[PDF](https://arxiv.org/abs/2311.12076)]



## Full-shot OOD Detection Approaches

- `NPOS`[Tao et al.]\[ICLR 2023] **NON-PARAMETRIC OUTLIER SYNTHESIS**[[PDF](https://openreview.net/pdf?id=JHklpEZqduQ)]\[[CODE](https://github.com/deeplearning-wisc/npos)]
- `PT-OOD`[Miyai et al.]\[arxiv 2023]**CAN PRE-TRAINED NETWORKS DETECT FAMILIAR OUT-OF-DISTRIBUTION DATA?**[[PDF](https://arxiv.org/pdf/2310.00847.pdf)]\[[CODE](https://github.com/AtsuMiyai/PT-OOD)]
