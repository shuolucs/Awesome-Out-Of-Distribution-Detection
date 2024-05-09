# Awesome Training-agnostic OOD Detection[![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)

A curated list of awesome Training-agnostic OOD detection resources. Your contributions are always welcome!

## Content

- [Post-hoc Approaches](#post-hoc-approaches)
  * [Representation-based](#representation-based)
  * [Gradient-based](#gradient-based)
- [Test Time Adaptive Approaches](#test-time-adaptive-approaches)
  * [Model update-needed](#model-update-needed)
  * [Model update-free](#model-update-free)



## Post-hoc Approaches 

### Representation-based

- `NECO` [Ammar et al.]\[ICLR 2024]**NECO: NEural Collapse Based Out-of-distribution detection**[[PDF](https://arxiv.dosf.top/abs/2310.06823)]\[[CODE](https://gitlab.com/drti/neco)]

- `NAN` [Park et al.]\[ICCV 2023]**Understanding the Feature Norm for Out-of-Distribution Detection**[[PDF](https://openaccess.thecvf.com/content/ICCV2023/html/Park_Understanding_the_Feature_Norm_for_Out-of-Distribution_Detection_ICCV_2023_paper.html)]

- `VRA` [Xu et al.]\[NeurIPS 2023]**VRA: Variational Rectified Activation for Out-of-distribution Detection**[[PDF](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5c20c00504e0c049ec2370d0cceaf3c4-Abstract-Conference.html)]\[[CODE](https://github. com/zeroQiaoba/VRA)]

- `ReAct` [Sun et al.]\[NeurIPS 2021]**React: Out-of-distribution detection with rectified activations**[[PDF](https://proceedings.neurips.cc/paper/2021/hash/01894d6f048493d2cacde3c579c315a3-Abstract.html)]\[[CODE](https://github.com/deeplearning-wisc/react.git)]

- `ASH` [Djurisic et al.]\[ICLR 2023]**Extremely simple activation shaping for out-of-distribution detection**[[PDF](https://arxiv.dosf.top/abs/2209.09858)][[CODE](https://andrijazz.github.io/ash/)]

- `OptTSOOD`[Zhao et al.]\[ICLR 2024]**TOWARDS OPTIMAL FEATURE-SHAPING METHODS FOR OUT-OF-DISTRIBUTION DETECTION**[[PDF](https://arxiv.org/abs/2402.00865)]\[[CODE](https://github.com/Qinyu-Allen-Zhao/OptFSOOD)]

- `NAP`[Wan et al.]\[arxiv 2024]**Out-of-Distribution Detection using Neural Activation Prior**[[PDF](https://arxiv.org/abs/2402.18162)]

- `CoP&CoRP`[Fang et al.]\[arxiv 2024]**KERNEL PCA FOR OUT-OF-DISTRIBUTION DETECTION**[[PDF](https://arxiv.org/abs/2402.02949)]

- `...`[Cook et al.]\[arxiv 2024]**Feature Density Estimation for Out-of-Distribution Detection via Normalizing Flows**[[PDF](https://arxiv.org/abs/2402.06537)]

### Gradient-based

- [Lee et al.]\[ICIP 2020]**Gradients as a measure of uncertainty in neural networks**[[PDF](https://ieeexplore.ieee.org/abstract/document/9190679/)]

- `GradNorm` [Huang et al.]\[NeurIPS 2021]**On the importance of gradients for detecting distributional shifts in the wild**[[PDF](https://proceedings.neurips.cc/paper_files/paper/2021/hash/063e26c670d07bb7c4d30e6fc69fe056-Abstract.html)]\[[CODE](https://github.com/deeplearning-wisc/gradnorm_ood)]

- `GradOrth` [Behpour et al.]\[NuerIPS 2023]**GradOrth: A Simple yet Efficient Out-of-Distribution Detection with Orthogonal Projection of Gradients**[[PDF](https://proceedings.neurips.cc/paper_files/paper/2023/hash/77cf940349218069bbc230fc2c9c8a21-Abstract-Conference.html)]

- `GAIA` [Chen etal.]\[NeurIPS 2023]**GAIA: Delving into Gradient-based Attribution Abnormality for Out-of-distribution Detection**[[PDF](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fcdccd419c4dc471fa3b73ec97b53789-Abstract-Conference.html)]



## Test Time Adaptive Approaches

### Theoretical support 

- `...`[Fang et al.]\[NeurIPS 2022]**Is Out-of-Distribution Detection Learnable?**[[PDF](https://openreview.net/forum?id=sde_7ZzGXOE)]



- `UniEnt`[Gao et al.]\[arxiv 2024]**Unified Entropy Optimization for Open-Set Test-Time Adaptation**[[PDF](https://arxiv.org/pdf/2404.06065.pdf)]\[[CODE](https://github.com/gaozhengqing/UniEnt)]

### Model update-needed

- `SAL`[Du et al.]\[ICLR 2024]**HOW DOES UNLABELED DATA PROVABLY HELP OUT-OF-DISTRIBUTION DETECTION?** [[PDF](https://openreview.net/forum?id=jlEjB8MVGa)]\[[CODE](https://github.com/deeplearning-wisc/sal)]

- `ATTA`[Gao et al.]\[NeurIPS 2023]**ATTA: Anomaly-aware Test-Time Adaptation for Out-of-Distribution Detection in Segmentation**[[PDF](https://openreview.net/forum?id=bGcdjXrU2w&referrer=%5Bthe%20profile%20of%20Shipeng%20Yan%5D(%2Fprofile%3Fid%3D~Shipeng_Yan1))]\[[CODE](https://github.com/gaozhitong/ATTA)]
- `MOL`[Wu et al.]\[CVPR 2023]**Meta OOD Learning For Continuously Adaptive OOD Detection**[[PDF](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Meta_OOD_Learning_For_Continuously_Adaptive_OOD_Detection_ICCV_2023_paper.html)]
- `SODA`[Geng et al.]\[arxiv 2023]**SODA: Stream Out-of-Distribution Adaptation**[[PDF](https://openreview.net/forum?id=Ur4LqAOXIF&referrer=%5Bthe%20profile%20of%20Yixuan%20Li%5D(%2Fprofile%3Fid%3D~Yixuan_Li1))]
- `AUTO`[Yang et al.]\[arxiv 2023]**AUTO: Adaptive Outlier Optimization for Online Test-Time OOD Detection**[[PDF](https://arxiv.org/abs/2303.12267)]
- `WOODS`[Katz-Samuels et al.]\[ICML 2022]**Training OOD Detectors in their Natural Habitats**[[PDF](https://arxiv.org/abs/2202.03299)]\[[CODE](https://github.com/jkatzsam/woods_ood)]

### Model update-free

- `ETLT`[Fan et al.]\[arxiv 2023/CVPR 2024]**A Simple Test-Time Method for Out-of-Distribution Detection**[[PDF](https://arxiv.org/abs/2207.08210)]

- `GOODAT`[Wang et al.]\[AAAI 2024]**Towards Test-time Graph Out-of-Distribution Detection**[[PDF](https://arxiv.org/abs/2401.06176)]

- `AdaOOD`[Zhang et al.]\[arxiv 2023]**Model-free Test Time Adaptation for Out-Of-Distribution Detection**[[PDF](https://arxiv.org/abs/2311.16420)]



 
