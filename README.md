# Resource Repository for Deep Learning Model Inversion Attacks and Defenses
This is a comprehensive resource repository for deep learning model inversion attacks and defenses research.


## Taxonomy of Model Inversion Attacks

|         Method         |                                Paper                                 |                                                    Publication                                                     |   Source   |
|:----------------------:|:-------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|:--------:|
|   DLG  | Deep Leakage from Gradients | NeurIPS'2019 |    [[Paper]](https://proceedings.neurips.cc/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html), [[Code]](https://github.com/mit-han-lab/dlg)    |
|      iDLG    |                              iDLG: Improved Deep Leakage from Gradients                           |                                              arXiv     |  [[Paper]](https://arxiv.org/abs/2001.02610)                                                   |     [Github](https://github.com/PatrickZH/Improved-Deep-Leakage-from-Gradients/blob/master/iDLG.py))
|  AGIC  |  AGIC: Approximate Gradient Inversion Attack on Federated Learning  |  SRDS'2022  |  [[Paper]](https://ieeexplore.ieee.org/abstract/document/9996844?casa_token=HW2g88ZKvyYAAAAA:ynaPpf6qzZY8ptc31j9lSHEIkP8B2skFskNLT3-xUjbdIK1mqtRGAT_ErtT1_beOGl0upNGNCSD1)  |
|  RGCIR  |  An effective and practical gradient inversion attack  |  IJIS'2022  |  [[Paper]](https://onlinelibrary.wiley.com/doi/10.1002/int.22997)  |
|  LOMMA  |  Re-Thinking Model Inversion Attacks Against Deep Neural Networks  |  CVPR'2023  |  [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Nguyen_Re-Thinking_Model_Inversion_Attacks_Against_Deep_Neural_Networks_CVPR_2023_paper.html), [[Code]]( https://ngoc-nguyen-0.github.io/re-thinking_model_inversion_attacks/)  |
|  EGIA  |  Egia: An external gradient inversion attack in federated learning  |  TIFS'2023  |  [[Paper]](https://ieeexplore.ieee.org/abstract/document/10209197?casa_token=8Z3tPnQDgvsAAAAA:sla4sO1caXCPVZrPFa62KkpjqDYcpuUAS2Y8UloY8lj0gJn3dZAqFbwcQwFdJICLoKvnCmmHdbOc), [[Code]](https://github.com/RuslandGadget/FCN-Inv)  |
|  SGI  |  High-Fidelity Gradient Inversion in Distributed Learning  |  AAAI'2024  |  [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/29975), [[Code]](https://github.com/MiLab-HITSZ/2023YeHFGradInv)  |
|  GGI  |  GGI: Generative Gradient Inversion Attack in Federated Learning  |  DOCS'2024  |  [[Paper]](https://ieeexplore.ieee.org/document/10704504)  |
|  GI-NAS  |  GI-NAS: Boosting Gradient Inversion Attacks through Adaptive Neural Architecture Search  |  arXiv  |  [[Code】](https://arxiv.org/abs/2405.20725)  |
|  GMI  |  The Secret Revealer: Generative Model-Inversion Attacks Against Deep Neural Networks  |  CVPR'2020  |  [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_The_Secret_Revealer_Generative_Model-Inversion_Attacks_Against_Deep_Neural_Networks_CVPR_2020_paper.html), [[Code]](https://github.com/AI-secure/GMI-Attack) |
|  VMI  |  Variational model inversion attacks  |  NeurIPS'2021  |  [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/50a074e6a8da4662ae0a29edde722179-Abstract.html), [[Code]](https://github.com/wangkua1/vmi)|
|  KEDMI  |  Knowledge-Enriched Distributional Model Inversion Attacks  |  ICCV'2021  |  [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Knowledge-Enriched_Distributional_Model_Inversion_Attacks_ICCV_2021_paper.html), [[Code]](https://github.com/SCccc21/Knowledge-Enriched-DMI)  |
|  PLG-MI  |  Pseudo Label-Guided Model Inversion Attack via Conditional Generative Adversarial Network  |  AAAI'2023  |  [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/25442), [[Code]](https://github.com/LetheSec/PLG-MI-Attack)  |
|  IF-GMI  |  A Closer Look at GAN Priors: Exploiting Intermediate Features for Enhanced Model Inversion Attacks  |  ECCV'2024  |  [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-73411-3_7), [[Code]](https://github.com/final-solution/IF-GMI)  |
|  MIRROR  |  MIRROR: Model Inversion for Deep Learning Network with High Fidelity  |  NDSS'2022  |  [[Paper]](https://www.ndss-symposium.org/ndss-paper/auto-draft-203/), [[Code]](https://model-inversion.github.io/mirror/) |
|  PPA  |  Plug & Play Attacks: Towards Robust and Flexible Model Inversion Attacks  |  arXiv  |  [[Paper]](https://arxiv.org/pdf/2201.12179), [[Code]](https://github.com/LukasStruppek/Plug-and-Play-Attacks)  |
|  BREP-MI  |  Label-Only Model Inversion Attacks via Boundary Repulsion  |  CVPR'2022  |  [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Kahla_Label-Only_Model_Inversion_Attacks_via_Boundary_Repulsion_CVPR_2022_paper.html), [[Code]](https://github.com/m-kahla/Label-Only-Model-Inversion-Attacks-via-Boundary-Repulsion)  |
|  C2FMI  |  C2FMI: Corse-to-Fine Black-box Model Inversion Attack  |  TDSC'2023  |  [[Paper]](https://ieeexplore.ieee.org/document/10148574), [[Code]](https://github.com/MiLabHITSZ/2022YeC2FMI)  |
|  RLBMI  |  Reinforcement Learning-Based Black-Box Model Inversion Attacks  |  CVPR'2023  |  [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Han_Reinforcement_Learning-Based_Black-Box_Model_Inversion_Attacks_CVPR_2023_paper.html), [[Code]](https://github.com/HanGyojin/RLB-MI)  |
|  LOKT  |  Label-Only Model Inversion Attacks via Knowledge Transfer  |  NeurIPS'2023  |  [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d9827e811c5a205c1313fb950c072c7d-Abstract-Conference.html), [[Code]](https://ngoc-nguyen-0.github.io/lokt/)  |
|  SecretGen  |  SecretGen: Privacy Recovery on Pre-trained Models via Distribution Discrimination  |  ECCV'2022  |  [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-20065-6_9), [[Code]](https://github.com/AI-secure/SecretGen)  |
|  GIFD  |  GIFD: A Generative Gradient Inversion Method with Feature Domain Optimization  |  ICCV'2023  |  [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Fang_GIFD_A_Generative_Gradient_Inversion_Method_with_Feature_Domain_Optimization_ICCV_2023_paper.html), [[Code]](https://github.com/ffhibnese/GIFD_Gradient_Inversion_Attack)  |
|  DMMIA  |   Model Inversion Attack via Dynamic Memory Learning  |  MM'23  |  [[Paper]](https://dl.acm.org/doi/abs/10.1145/3581783.3612072)  |
|  Patch-MI  |  Patch-MI: Enhancing Model Inversion Attacks via Patch-Based Reconstruction  |  arXiv  |  [[Paper]](https://arxiv.org/abs/2312.07040)  |
|  SIA-GAN  |  SIA-GAN: Scrambling Inversion Attack Using Generative Adversarial Network  |  Access  |  [[Paper]](https://ieeexplore.ieee.org/abstract/document/9537763), [[Code]](https://github.com/MADONOKOUKI/SIA-GAN)  |
|  FedInverse  |  FedInverse: Evaluating Privacy Leakage in Federated Learning  |  ICLR'2024  |  [[Paper]](https://openreview.net/forum?id=nTNgkEIfeb), [[Code]](https://github.com/Jun-B0518/FedInverse)  |
|  DeepInversion  |  Dreaming to Distill: Data-Free Knowledge Transfer via DeepInversion  |  CVPR'2020  |  [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/html/Yin_Dreaming_to_Distill_Data-Free_Knowledge_Transfer_via_DeepInversion_CVPR_2020_paper.html), [[Code]](https://github.com/NVlabs/DeepInversion)  |
|  RL-GraphMI  |  Model Inversion Attacks Against Graph Neural Networks  |  TKDE'2023  |  [[Paper]](https://ieeexplore.ieee.org/abstract/document/9895303?casa_token=N1MQRPXQiQYAAAAA:7h3eUPfA1X6f4bR_oHDACnVNxq0KQaP8t3Leiyhhk9btTIC-DQcDRsrp1a60nU-dMU2EhCsG2g)  |
|  XAI  |  Exploiting Explanations for Model Inversion Attacks  |  ICCV'2021  |  [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_Exploiting_Explanations_for_Model_Inversion_Attacks_ICCV_2021_paper.html)  |
|  SMI  |  The Role of Class Information in Model Inversion Attacks against Image Deep Learning Classifiers  |  TDSC'2023  |  [[Paper]](https://ieeexplore.ieee.org/abstract/document/10225397?casa_token=7otFd3hFaF4AAAAA:6jpMBYenbS4j4G9MvMAbuLFPrmM8qBA1Wotkxrf-vp81rpvXK688OiHObOeujKDDD6ET5Cf9z2RD)  |
|  LSMI  |  Be Careful What You Smooth For: Label Smoothing Can Be a Privacy Shield but Also a Catalyst for Model Inversion Attacks  |  arXiv  |  [[Paper]](https://arxiv.org/abs/2310.06549) |
|  EMI  |  Reconstructing Training Data From Diverse ML Models by Ensemble Inversion  |  WACV'2022  |  [[Paper]](ontent/WACV2022/html/Wang_Reconstructing_Training_Data_From_Diverse_ML_Models_by_Ensemble_Inversion_WACV_2022_paper.html)  |



## Defenses Against Model Inversion Attacks
|         Method         |                                Paper                                 |                                                    Publication                                                     |   Source   |
|:----------------------:|:-------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|:--------:|
|  VRM  |  Gradient Inversion Attacks: Impact Factors Analyses and Privacy Enhancement  |  PAMI'2024  |  [[Paper]](https://ieeexplore.ieee.org/abstract/document/10604429)  |
|  DCS2  |  Concealing Sensitive Samples against Gradient Leakage in Federated Learning  |  AAAI'2024  |  [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/30171), [[Code]](https://github.com/JingWu321/DCS-2)  |
|  ATS  |  Privacy-Preserving Collaborative Learning With Automatic Transformation Search  |  CVPR'2021  |  [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Gao_Privacy-Preserving_Collaborative_Learning_With_Automatic_Transformation_Search_CVPR_2021_paper.html), [[Code]](https://github.com/gaow0007/ATSPrivacy)  |
|  Soteria  |  Soteria: Provable Defense Against Privacy Leakage in Federated Learning From Representation Perspective  |  CVPR'2021  |  [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_Soteria_Provable_Defense_Against_Privacy_Leakage_in_Federated_Learning_From_CVPR_2021_paper.html), [[Code]](https://github.com/jeremy313/Soteria)  |
|  Crafter  |  Crafter: Facial Feature Crafting against Inversion-based Identity Theft on Deep Models  |  arXiv  |  [[Paper]](https://arxiv.org/abs/2401.07205)  |
|  Quantization Enabled FL  |  Mixed quantization enabled federated learning to tackle gradient inversion attacks  |  CVPR'2023  |  [[Paper]](https://openaccess.thecvf.com/content/CVPR2023W/FedVision/html/Ovi_Mixed_Quantization_Enabled_Federated_Learning_To_Tackle_Gradient_Inversion_Attacks_CVPRW_2023_paper.html), [[Code]](https://github.com/PretomRoy/Defense-against-grad-inversion-attacks)  |
|  Sparse-coding Architecture  |  Improving Robustness to Model Inversion Attacks via Sparse Coding Architectures  |  arXive  |  [[Paper]](https://arxiv.org/abs/2403.14772)  |
|  Synthetic Data Generation with Privacy-preserving Techniques  |  Exploring Privacy-Preserving Techniques on Synthetic Data as a Defense Against Model Inversion Attacks  |  ISC'2023  |  [[Paper]](https://repository.ubn.ru.nl/bitstream/handle/2066/301262/301262.pdf?sequence=1)  |
|  Image Augmentation  |   An empirical analysis of image augmentation against model inversion attack in federated learning  |  Cluster Computering‘2022  |  [[Paper]](https://link.springer.com/article/10.1007/s10586-022-03596-1)  |
|  Additive Noise  |  Practical Defences Against Model Inversion Attacks for Split Neural Networks  |  arXiv  |  [[Paper]](https://arxiv.org/abs/2104.05743)  |
|  Privacy-guided Training  |  Reducing Risk of Model Inversion Using Privacy-Guided Training  |  arXiv  |  [[Paper]](https://arxiv.org/abs/2006.15877)  |
|  Statistical Features via Knowledge Distillation  |  Defending against gradient inversion attacks in federated learning via statistical machine unlearning  |  KBS'2024  |  [[Paper]](https://www.sciencedirect.com/science/article/pii/S0950705124006178)  |
|  PATROL  |  PATROL: Privacy-Oriented Pruning for Collaborative Inference Against Model Inversion Attacks  |  WACV'2024  |  [[Paper]](https://openaccess.thecvf.com/content/WACV2024/html/Ding_PATROL_Privacy-Oriented_Pruning_for_Collaborative_Inference_Against_Model_Inversion_Attacks_WACV_2024_paper.html)  |
|  DGP  |  Revisiting Gradient Pruning: A Dual Realization for Defending against Gradient Attacks  |  AAAI'2024  |  [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/28460)  |
|  Guardian  |  Guardian: Guarding against Gradient Leakage with Provable Defense for Federated Learning  |  WSDM'2024  |  [[Paper]](https://dl.acm.org/doi/10.1145/3616855.3635758)  |
|  pFGD  |  Mitigating Gradient Inversion Attacks in Federated Learning with Frequency Transformation  |  ESORICS'2023  |  [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-54129-2_44)  |
|  DGC  |  Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training  |  ICLR'2018  |  [[Paper]](https://openreview.net/forum?id=SkhQHMW0W),[[Code]](https://github.com/synxlin/deep-gradient-compression)  |  
|  Model Fragmentation, Shuffle and Aggregation  |  Model Fragmentation, Shuffle and Aggregation to Mitigate Model Inversion in Federated Learning  |  LANMAN'2021  |  [[Paper]](https://ieeexplore.ieee.org/document/9478813)  |
















## Evaluation Metrics in MI Attacks and Defenses


## Datasets for MI Attack Research
