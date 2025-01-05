# Resource Repository for Deep Learning Model Inversion Attacks and Defenses
This is a comprehensive resource repository for deep learning model inversion attacks and defenses research.


## Taxonomy of Model Inversion Attacks

|         Method         |                                Paper                                 |                                                    Publication                                                     |   Source   |
|:----------------------:|:-------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|:--------:|
|   DLG  | Deep Leakage from Gradients | NeurIPS'2019 |    [[Paper]](https://proceedings.neurips.cc/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html), [[Code]](https://github.com/mit-han-lab/dlg)    |
|      iDLG    |                              iDLG: Improved Deep Leakage from Gradients                           |                                              [arXiv](https://arxiv.org/abs/2001.02610)                                                   |     [Github](https://github.com/PatrickZH/Improved-Deep-Leakage-from-Gradients/blob/master/iDLG.py)     |
|  AGIC  |  AGIC: Approximate Gradient Inversion Attack on Federated Learning  |  [SRDS'2022](https://ieeexplore.ieee.org/abstract/document/9996844?casa_token=HW2g88ZKvyYAAAAA:ynaPpf6qzZY8ptc31j9lSHEIkP8B2skFskNLT3-xUjbdIK1mqtRGAT_ErtT1_beOGl0upNGNCSD1)  |  -  |
|  RGCIR  |  An effective and practical gradient inversion attack  |  [IJIS'2022](https://onlinelibrary.wiley.com/doi/10.1002/int.22997)  |  -  |
|  LOMMA  |  Re-Thinking Model Inversion Attacks Against Deep Neural Networks  |  [CVPR'2023](https://openaccess.thecvf.com/content/CVPR2023/html/Nguyen_Re-Thinking_Model_Inversion_Attacks_Against_Deep_Neural_Networks_CVPR_2023_paper.html)  |  [ https://ngoc-nguyen-0.github.io/re-thinking_model_inversion_attacks/]( https://ngoc-nguyen-0.github.io/re-thinking_model_inversion_attacks/)  |
|  EGIA  |  Egia: An external gradient inversion attack in federated learning  |  [TIFS'2023](https://ieeexplore.ieee.org/abstract/document/10209197?casa_token=8Z3tPnQDgvsAAAAA:sla4sO1caXCPVZrPFa62KkpjqDYcpuUAS2Y8UloY8lj0gJn3dZAqFbwcQwFdJICLoKvnCmmHdbOc)  |  [https://github.com/RuslandGadget/FCN-Inv](https://github.com/RuslandGadget/FCN-Inv)  |
|  SGI  |  High-Fidelity Gradient Inversion in Distributed Learning  |  [AAAI'20224](https://ojs.aaai.org/index.php/AAAI/article/view/29975)  |  [https://github.com/MiLab-HITSZ/2023YeHFGradInv](https://github.com/MiLab-HITSZ/2023YeHFGradInv)  |
|  GGI  |  GGI: Generative Gradient Inversion Attack in Federated Learning  |  [DOCS'2024](https://ieeexplore.ieee.org/document/10704504)  |  -  |
|  GI-NAS  |  GI-NAS: Boosting Gradient Inversion Attacks through Adaptive Neural Architecture Search  |  [arXiv](https://arxiv.org/abs/2405.20725)  |  -  |
|  GMI  |  The Secret Revealer: Generative Model-Inversion Attacks Against Deep Neural Networks  |  [CVPR'2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_The_Secret_Revealer_Generative_Model-Inversion_Attacks_Against_Deep_Neural_Networks_CVPR_2020_paper.html)  |  [https://github.com/AI-secure/GMI-Attack](https://github.com/AI-secure/GMI-Attack) |
|  VMI  |  Variational model inversion attacks  |  [NeurIPS'2021](https://proceedings.neurips.cc/paper/2021/hash/50a074e6a8da4662ae0a29edde722179-Abstract.html)  |  https://github.com/wangkua1/vmi|
|  KEDMI  |  Knowledge-Enriched Distributional Model Inversion Attacks  |  [ICCV'2021](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Knowledge-Enriched_Distributional_Model_Inversion_Attacks_ICCV_2021_paper.html)  |  [https://github.com/SCccc21/Knowledge-Enriched-DMI](https://github.com/SCccc21/Knowledge-Enriched-DMI)  |
|  PLG-MI  |  Pseudo Label-Guided Model Inversion Attack via Conditional Generative Adversarial Network  |  [AAAI'2023](https://ojs.aaai.org/index.php/AAAI/article/view/25442)  |  [https://github.com/LetheSec/PLG-MI-Attack](https://github.com/LetheSec/PLG-MI-Attack)  |
|  IF-GMI  |  A Closer Look at GAN Priors: Exploiting Intermediate Features for Enhanced Model Inversion Attacks  |  [ECCV'2024](https://link.springer.com/chapter/10.1007/978-3-031-73411-3_7)  |  [https://github.com/final-solution/IF-GMI](https://github.com/final-solution/IF-GMI)  |
|  MIRROR  |  MIRROR: Model Inversion for Deep Learning Network with High Fidelity  |  [NDSS'2022](https://www.ndss-symposium.org/ndss-paper/auto-draft-203/)  |  [https://model-inversion.github.io/mirror/](https://model-inversion.github.io/mirror/) |
|  PPA  |  Plug & Play Attacks: Towards Robust and Flexible Model Inversion Attacks  |  [ICML'2022](https://arxiv.org/pdf/2201.12179)  |  [https://github.com/LukasStruppek/Plug-and-Play-Attacks](https://github.com/LukasStruppek/Plug-and-Play-Attacks)  |
|  BREP-MI  |  Label-Only Model Inversion Attacks via Boundary Repulsion  |  [CVPR'2022](https://openaccess.thecvf.com/content/CVPR2022/html/Kahla_Label-Only_Model_Inversion_Attacks_via_Boundary_Repulsion_CVPR_2022_paper.html)  |  [https://github.com/m-kahla/Label-Only-Model-Inversion-Attacks-via-Boundary-Repulsion](https://github.com/m-kahla/Label-Only-Model-Inversion-Attacks-via-Boundary-Repulsion)  |
|  C2FMI  |  C2FMI: Corse-to-Fine Black-box Model Inversion Attack  |  [TDSC'2023](https://ieeexplore.ieee.org/document/10148574)  |  https://github.com/MiLabHITSZ/2022YeC2FMI  |
|  RLBMI  |  Reinforcement Learning-Based Black-Box Model Inversion Attacks  |  [CVPR'2023](https://openaccess.thecvf.com/content/CVPR2023/html/Han_Reinforcement_Learning-Based_Black-Box_Model_Inversion_Attacks_CVPR_2023_paper.html)  |  [https://github.com/HanGyojin/RLB-MI](https://github.com/HanGyojin/RLB-MI)  |
|  LOKT  |  Label-Only Model Inversion Attacks via Knowledge Transfer  |  [NeurIPS'2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d9827e811c5a205c1313fb950c072c7d-Abstract-Conference.html)  |  [https://ngoc-nguyen-0.github.io/lokt/](https://ngoc-nguyen-0.github.io/lokt/)  |
|  SecretGen  |  SecretGen: Privacy Recovery on Pre-trained Models via Distribution Discrimination  |  [ECCV'2022](https://link.springer.com/chapter/10.1007/978-3-031-20065-6_9)  |  [https://github.com/AI-secure/SecretGen](https://github.com/AI-secure/SecretGen)  |
|  GIFD  |  GIFD: A Generative Gradient Inversion Method with Feature Domain Optimization  |  [ICCV'2023](https://openaccess.thecvf.com/content/ICCV2023/html/Fang_GIFD_A_Generative_Gradient_Inversion_Method_with_Feature_Domain_Optimization_ICCV_2023_paper.html)  |  [https://github.com/ffhibnese/GIFD_Gradient_Inversion_Attack](https://github.com/ffhibnese/GIFD_Gradient_Inversion_Attack)  |
|  DMMIA  |   Model Inversion Attack via Dynamic Memory Learning  |  [MM'23](https://dl.acm.org/doi/abs/10.1145/3581783.3612072)  |  -  |
|  Patch-MI  |  Patch-MI: Enhancing Model Inversion Attacks via Patch-Based Reconstruction  |  [arXiv](https://arxiv.org/abs/2312.07040)  |  -  |
|  SIA-GAN  |  SIA-GAN: Scrambling Inversion Attack Using Generative Adversarial Network  |  [Access](https://ieeexplore.ieee.org/abstract/document/9537763)  |  [https://github.com/MADONOKOUKI/SIA-GAN](https://github.com/MADONOKOUKI/SIA-GAN)  |
|  FedInverse  |  FedInverse: Evaluating Privacy Leakage in Federated Learning  |  [ICLR'2024](https://openreview.net/forum?id=nTNgkEIfeb)  |  [https://github.com/Jun-B0518/FedInverse](https://github.com/Jun-B0518/FedInverse)  |










## Defenses Against Model Inversion Attacks

| Method | Paper  | Source | Code | 
|---|---|---|---|
| Row 1, Cell 1 | Row 1, Cell 2 | Row 1, Cell 3 |---|
| Row 2, Cell 1 | Row 2, Cell 2 | Row 2, Cell 3 |---|

## Evaluation Metrics in MI Attacks and Defenses


## Datasets for MI Attack Research
