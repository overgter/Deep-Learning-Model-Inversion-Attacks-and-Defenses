# 🔥🔥🔥Resource Repository for Deep Learning Model Inversion Attacks and Defenses🔥🔥🔥
**Model Inversion (MI) Attacks** have become a major privacy threat that utilizes various methods to reconstruct sensitive data from machine/deep learning models. This is a comprehensive resource repository for deep learning MI attacks and defenses research. <span style="color:blue;"> repository will be continuously maintained to ensure its relevance and usefulness. If you would like to contribute to this repository—for example, by including your paper or code—please send a message to the first author of our survey paper listed in the bottom of this page!</span>

## 🧨Taxonomy of Model Inversion Attacks🧨
### - Gradient Inversion Attacks
|         Method         |                                Paper                                 |                                                    Publication                                                     |   Source   |
|:----------------------:|:-------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|:--------:|
|   DLG  | Deep Leakage from Gradients | NeurIPS'2019 |    [[Paper]](https://proceedings.neurips.cc/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html), [[Code]](https://github.com/mit-han-lab/dlg)    |
|      iDLG    |                              iDLG: Improved Deep Leakage from Gradients                           |                                              arXiv‘2020     |  [[Paper]](https://arxiv.org/abs/2001.02610)                                                   |     [Github](https://github.com/PatrickZH/Improved-Deep-Leakage-from-Gradients/blob/master/iDLG.py))
|  AGIC  |  AGIC: Approximate Gradient Inversion Attack on Federated Learning  |  SRDS'2022  |  [[Paper]](https://ieeexplore.ieee.org/abstract/document/9996844?casa_token=HW2g88ZKvyYAAAAA:ynaPpf6qzZY8ptc31j9lSHEIkP8B2skFskNLT3-xUjbdIK1mqtRGAT_ErtT1_beOGl0upNGNCSD1)  |
|  RGCIR  |  An effective and practical gradient inversion attack  |  IJIS'2022  |  [[Paper]](https://onlinelibrary.wiley.com/doi/10.1002/int.22997)  |
|  LOMMA  |  Re-Thinking Model Inversion Attacks Against Deep Neural Networks  |  CVPR'2023  |  [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Nguyen_Re-Thinking_Model_Inversion_Attacks_Against_Deep_Neural_Networks_CVPR_2023_paper.html), [[Code]]( https://ngoc-nguyen-0.github.io/re-thinking_model_inversion_attacks/)  |
|  EGIA  |  Egia: An external gradient inversion attack in federated learning  |  TIFS'2023  |  [[Paper]](https://ieeexplore.ieee.org/abstract/document/10209197?casa_token=8Z3tPnQDgvsAAAAA:sla4sO1caXCPVZrPFa62KkpjqDYcpuUAS2Y8UloY8lj0gJn3dZAqFbwcQwFdJICLoKvnCmmHdbOc), [[Code]](https://github.com/RuslandGadget/FCN-Inv)  |
|  SGI  |  High-Fidelity Gradient Inversion in Distributed Learning  |  AAAI'2024  |  [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/29975), [[Code]](https://github.com/MiLab-HITSZ/2023YeHFGradInv)  |
|  GGI  |  GGI: Generative Gradient Inversion Attack in Federated Learning  |  DOCS'2024  |  [[Paper]](https://ieeexplore.ieee.org/document/10704504)  |
|  GI-NAS  |  GI-NAS: Boosting Gradient Inversion Attacks through Adaptive Neural Architecture Search  |  arXiv'2024  |  [[Paper]](https://arxiv.org/abs/2405.20725)  |
### - Generative Model-based Attacks
|         Method         |                                Paper                                 |                                                    Publication                                                     |   Source   |
|:----------------------:|:-------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|:--------:|
|  GMI  |  The Secret Revealer: Generative Model-Inversion Attacks Against Deep Neural Networks  |  CVPR'2020  |  [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_The_Secret_Revealer_Generative_Model-Inversion_Attacks_Against_Deep_Neural_Networks_CVPR_2020_paper.html), [[Code]](https://github.com/AI-secure/GMI-Attack) |
|  VMI  |  Variational model inversion attacks  |  NeurIPS'2021  |  [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/50a074e6a8da4662ae0a29edde722179-Abstract.html), [[Code]](https://github.com/wangkua1/vmi)|
|  KEDMI  |  Knowledge-Enriched Distributional Model Inversion Attacks  |  ICCV'2021  |  [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Knowledge-Enriched_Distributional_Model_Inversion_Attacks_ICCV_2021_paper.html), [[Code]](https://github.com/SCccc21/Knowledge-Enriched-DMI)  |
|  PLG-MI  |  Pseudo Label-Guided Model Inversion Attack via Conditional Generative Adversarial Network  |  AAAI'2023  |  [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/25442), [[Code]](https://github.com/LetheSec/PLG-MI-Attack)  |
|  IF-GMI  |  A Closer Look at GAN Priors: Exploiting Intermediate Features for Enhanced Model Inversion Attacks  |  ECCV'2024  |  [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-73411-3_7), [[Code]](https://github.com/final-solution/IF-GMI)  |
|  MIRROR  |  MIRROR: Model Inversion for Deep Learning Network with High Fidelity  |  NDSS'2022  |  [[Paper]](https://www.ndss-symposium.org/ndss-paper/auto-draft-203/), [[Code]](https://model-inversion.github.io/mirror/) |
|  PPA  |  Plug & Play Attacks: Towards Robust and Flexible Model Inversion Attacks  |  arXiv'2022  |  [[Paper]](https://arxiv.org/pdf/2201.12179), [[Code]](https://github.com/LukasStruppek/Plug-and-Play-Attacks)  |
|  BREP-MI  |  Label-Only Model Inversion Attacks via Boundary Repulsion  |  CVPR'2022  |  [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Kahla_Label-Only_Model_Inversion_Attacks_via_Boundary_Repulsion_CVPR_2022_paper.html), [[Code]](https://github.com/m-kahla/Label-Only-Model-Inversion-Attacks-via-Boundary-Repulsion)  |
|  C2FMI  |  C2FMI: Corse-to-Fine Black-box Model Inversion Attack  |  TDSC'2023  |  [[Paper]](https://ieeexplore.ieee.org/document/10148574), [[Code]](https://github.com/MiLabHITSZ/2022YeC2FMI)  |
|  RLBMI  |  Reinforcement Learning-Based Black-Box Model Inversion Attacks  |  CVPR'2023  |  [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Han_Reinforcement_Learning-Based_Black-Box_Model_Inversion_Attacks_CVPR_2023_paper.html), [[Code]](https://github.com/HanGyojin/RLB-MI)  |
|  LOKT  |  Label-Only Model Inversion Attacks via Knowledge Transfer  |  NeurIPS'2023  |  [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d9827e811c5a205c1313fb950c072c7d-Abstract-Conference.html), [[Code]](https://ngoc-nguyen-0.github.io/lokt/)  |
|  SecretGen  |  SecretGen: Privacy Recovery on Pre-trained Models via Distribution Discrimination  |  ECCV'2022  |  [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-20065-6_9), [[Code]](https://github.com/AI-secure/SecretGen)  |
|  GIFD  |  GIFD: A Generative Gradient Inversion Method with Feature Domain Optimization  |  ICCV'2023  |  [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Fang_GIFD_A_Generative_Gradient_Inversion_Method_with_Feature_Domain_Optimization_ICCV_2023_paper.html), [[Code]](https://github.com/ffhibnese/GIFD_Gradient_Inversion_Attack)  |
|  DMMIA  |   Model Inversion Attack via Dynamic Memory Learning  |  MM'23  |  [[Paper]](https://dl.acm.org/doi/abs/10.1145/3581783.3612072)  |
|  Patch-MI  |  Patch-MI: Enhancing Model Inversion Attacks via Patch-Based Reconstruction  |  arXiv'2024  |  [[Paper]](https://arxiv.org/abs/2312.07040)  |
|  SIA-GAN  |  SIA-GAN: Scrambling Inversion Attack Using Generative Adversarial Network  |  Access'2021  |  [[Paper]](https://ieeexplore.ieee.org/abstract/document/9537763), [[Code]](https://github.com/MADONOKOUKI/SIA-GAN)  |
|  FedInverse  |  FedInverse: Evaluating Privacy Leakage in Federated Learning  |  ICLR'2024  |  [[Paper]](https://openreview.net/forum?id=nTNgkEIfeb), [[Code]](https://github.com/Jun-B0518/FedInverse)  |

### - Miscellaneous MI Attacks
|         Method         |                                Paper                                 |                                                    Publication                                                     |   Source   |
|:----------------------:|:-------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|:--------:|
|  DeepInversion  |  Dreaming to Distill: Data-Free Knowledge Transfer via DeepInversion  |  CVPR'2020  |  [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/html/Yin_Dreaming_to_Distill_Data-Free_Knowledge_Transfer_via_DeepInversion_CVPR_2020_paper.html), [[Code]](https://github.com/NVlabs/DeepInversion)  |
|  RL-GraphMI  |  Model Inversion Attacks Against Graph Neural Networks  |  TKDE'2023  |  [[Paper]](https://ieeexplore.ieee.org/abstract/document/9895303?casa_token=N1MQRPXQiQYAAAAA:7h3eUPfA1X6f4bR_oHDACnVNxq0KQaP8t3Leiyhhk9btTIC-DQcDRsrp1a60nU-dMU2EhCsG2g)  |
|  XAI  |  Exploiting Explanations for Model Inversion Attacks  |  ICCV'2021  |  [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_Exploiting_Explanations_for_Model_Inversion_Attacks_ICCV_2021_paper.html)  |
|  SMI  |  The Role of Class Information in Model Inversion Attacks against Image Deep Learning Classifiers  |  TDSC'2023  |  [[Paper]](https://ieeexplore.ieee.org/abstract/document/10225397?casa_token=7otFd3hFaF4AAAAA:6jpMBYenbS4j4G9MvMAbuLFPrmM8qBA1Wotkxrf-vp81rpvXK688OiHObOeujKDDD6ET5Cf9z2RD)  |
|  LSMI  |  Be Careful What You Smooth For: Label Smoothing Can Be a Privacy Shield but Also a Catalyst for Model Inversion Attacks  |  ICLR'2024 |  [[Paper]](https://arxiv.org/abs/2310.06549), [[Code]](https://github.com/LukasStruppek/Plug-and-Play-Attacks) |
|  EMI  |  Reconstructing Training Data From Diverse ML Models by Ensemble Inversion  |  WACV'2022  |  [[Paper]](ontent/WACV2022/html/Wang_Reconstructing_Training_Data_From_Diverse_ML_Models_by_Ensemble_Inversion_WACV_2022_paper.html)  |

## 🧯Defenses Against Model Inversion Attacks🧯
### - Feature Perturbation/obfuscation
|         Method         |                                Paper                                 |                                                    Publication                                                     |   Source   |
|:----------------------:|:-------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|:--------:|
|  VRM  |  Gradient Inversion Attacks: Impact Factors Analyses and Privacy Enhancement  |  PAMI'2024  |  [[Paper]](https://ieeexplore.ieee.org/abstract/document/10604429)  |
|  DCS2  |  Concealing Sensitive Samples against Gradient Leakage in Federated Learning  |  AAAI'2024  |  [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/30171), [[Code]](https://github.com/JingWu321/DCS-2)  |
|  ATS  |  Privacy-Preserving Collaborative Learning With Automatic Transformation Search  |  CVPR'2021  |  [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Gao_Privacy-Preserving_Collaborative_Learning_With_Automatic_Transformation_Search_CVPR_2021_paper.html), [[Code]](https://github.com/gaow0007/ATSPrivacy)  |
|  Soteria  |  Soteria: Provable Defense Against Privacy Leakage in Federated Learning From Representation Perspective  |  CVPR'2021  |  [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_Soteria_Provable_Defense_Against_Privacy_Leakage_in_Federated_Learning_From_CVPR_2021_paper.html), [[Code]](https://github.com/jeremy313/Soteria)  |
|  Crafter  |  Crafter: Facial Feature Crafting against Inversion-based Identity Theft on Deep Models  |  NDSS'2024  |  [[Paper]](https://arxiv.org/abs/2401.07205), [[Code]](https://github.com/shimingwang98/facial_feature_crafting_against_inversion_based_identity_theft)  |
|  Quantization Enabled FL  |  Mixed quantization enabled federated learning to tackle gradient inversion attacks  |  CVPR'2023  |  [[Paper]](https://openaccess.thecvf.com/content/CVPR2023W/FedVision/html/Ovi_Mixed_Quantization_Enabled_Federated_Learning_To_Tackle_Gradient_Inversion_Attacks_CVPRW_2023_paper.html), [[Code]](https://github.com/PretomRoy/Defense-against-grad-inversion-attacks)  |
|  Sparse-coding Architecture  |  Improving Robustness to Model Inversion Attacks via Sparse Coding Architectures  |  ECCV'2024  |  [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-72989-8_7)  |
|  Synthetic Data Generation with Privacy-preserving Techniques  |  Exploring Privacy-Preserving Techniques on Synthetic Data as a Defense Against Model Inversion Attacks  |  ISC'2023  |  [[Paper]](https://repository.ubn.ru.nl/bitstream/handle/2066/301262/301262.pdf?sequence=1)  |
|  Image Augmentation  |   An empirical analysis of image augmentation against model inversion attack in federated learning  |  Cluster Computering‘2022  |  [[Paper]](https://link.springer.com/article/10.1007/s10586-022-03596-1)  |
|  Additive Noise  |  Practical Defences Against Model Inversion Attacks for Split Neural Networks  |  arXiv'2021  |  [[Paper]](https://arxiv.org/abs/2104.05743)  |
|  Privacy-guided Training  |  Reducing Risk of Model Inversion Using Privacy-Guided Training  |  arXiv'2020  |  [[Paper]](https://arxiv.org/abs/2006.15877)  |
|  Statistical Features via Knowledge Distillation  |  Defending against gradient inversion attacks in federated learning via statistical machine unlearning  |  KBS'2024  |  [[Paper]](https://www.sciencedirect.com/science/article/pii/S0950705124006178)  |
### - Gradient Pruning
|         Method         |                                Paper                                 |                                                    Publication                                                     |   Source   |
|:----------------------:|:-------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|:--------:|
|  PATROL  |  PATROL: Privacy-Oriented Pruning for Collaborative Inference Against Model Inversion Attacks  |  WACV'2024  |  [[Paper]](https://openaccess.thecvf.com/content/WACV2024/html/Ding_PATROL_Privacy-Oriented_Pruning_for_Collaborative_Inference_Against_Model_Inversion_Attacks_WACV_2024_paper.html)  |
|  DGP  |  Revisiting Gradient Pruning: A Dual Realization for Defending against Gradient Attacks  |  AAAI'2024  |  [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/28460)  |
|  Guardian  |  Guardian: Guarding against Gradient Leakage with Provable Defense for Federated Learning  |  WSDM'2024  |  [[Paper]](https://dl.acm.org/doi/10.1145/3616855.3635758)  |
|  pFGD  |  Mitigating Gradient Inversion Attacks in Federated Learning with Frequency Transformation  |  ESORICS'2023  |  [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-54129-2_44)  |
|  DGC  |  Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training  |  ICLR'2018  |  [[Paper]](https://openreview.net/forum?id=SkhQHMW0W),[[Code]](https://github.com/synxlin/deep-gradient-compression)  |  
### - Gradient Perturbation/Obfuscation
|         Method         |                                Paper                                 |                                                    Publication                                                     |   Source   |
|:----------------------:|:-------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|:--------:|
|  Model Fragmentation, Shuffle and Aggregation  |  Model Fragmentation, Shuffle and Aggregation to Mitigate Model Inversion in Federated Learning  |  LANMAN'2021  |  [[Paper]](https://ieeexplore.ieee.org/document/9478813)  |
|  Autoencoder-based Compression  |  Learned Model Compression for Efficient and Privacy-Preserving Federated Learning  |  OJ-COMS'2024  |  [[Paper]](https://d197for5662m48.cloudfront.net/documents/publicationstatus/200432/preprint_pdf/f26b764e017f327d30ea2de7147300c6.pdf)  |
|  GradPrivacy  |  Preserving Privacy of Input Features Across All Stages of Collaborative Learning  |  BdCloud'2024  |  [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10491741)  |
|  Quantization Enabled FL  |  Mixed Quantization Enabled Federated Learning to Tackle Gradient Inversion Attacks  |  CVPRW'2023  |  [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10208585)  |
### - Differential Privacy
|         Method         |                                Paper                                 |                                                    Publication                                                     |   Source   |
|:----------------------:|:-------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|:--------:|
|  Augmented DP  |  Robust transparency against model inversion attacks  |  TDSC'2020  |  [[Paper]](https://ieeexplore.ieee.org/document/9178452)  |
|  Class-level and Subclass-level DP  |  Broadening Differential Privacy for Deep Learning Against Model Inversion Attacks  |  Big Data'2020  |  [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9378274)  |
|  DP in Healthcare Models  |  Gradient Mechanism to Preserve Differential Privacy and Deter Against Model Inversion Attacks in Healthcare Analytics  |  EMBC'2020  |  [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9176834)  |
|  Local DP  |  Efficient and Privacy-Enhanced Federated Learning Based on Parameter Degradation  |  TSC'2024  |  [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10528912)  |
###  - Cryptographic Encryption
|         Method         |                                Paper                                 |                                                    Publication                                                     |   Source   |
|:----------------------:|:-------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|:--------:|
|  Secure Aggregation  |  AAIA: an efficient aggregation scheme against inverting attack for federated learning  |  IJIS'2023  |  [[Paper]](https://link.springer.com/content/pdf/10.1007/s10207-023-00670-6.pdf)  |
|  Perceptual Hashing  |   Privacy preserving facial recognition against model inversion attacks  |  GLOBECOM'2020  |  [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9322508)  |
|  CAHE  |  Privacy-preserving distributed deep learning via LWE-based Certificateless Additively Homomorphic Encryption (CAHE) |  JISA'2023  |  [[Paper]](https://pdf.sciencedirectassets.com/287016/1-s2.0-S2214212623X00039/1-s2.0-S2214212623000467/main.pdf  |
### - Model/Architecture Enhancement
|         Method         |                                Paper                                 |                                                    Publication                                                     |   Source   |
|:----------------------:|:-------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|:--------:|
|  PRECODE  |  PRECODE - A Generic Model Extension To Prevent Deep Gradient Leakage  |  WACV'2022  |  [[Paper]](https://openaccess.thecvf.com/content/WACV2022/papers/Scheliga_PRECODE_-_A_Generic_Model_Extension_To_Prevent_Deep_Gradient_WACV_2022_paper.pdf), [[Code]](https://github.com/dAI-SY-Group/PRECODE)  |
|  SecCNN  |  Construct a Secure CNN Against Gradient Inversion Attack  |  PAKDD'2024  |  [[Paper]](https://link.springer.com/chapter/10.1007/978-981-97-2259-4_19)  |
|  RVE-PFL  |  VE-PFL: Robust Variational Encoder-based Personalised Federated Learning against Model Inversion Attacks  |  TIFS'2024  |  [[Paper]](https://dl.acm.org/doi/abs/10.1109/TIFS.2024.3368879), [[Code]](https://github.com/UNSW-Canberra-2023/RVE-PFL)  |
|  ResSFL  |  Ressfl: A resistance transfer framework for defending model inversion attack in split federated learning  |  CVPR'2022  |  [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_ResSFL_A_Resistance_Transfer_Framework_for_Defending_Model_Inversion_Attack_CVPR_2022_paper.pdf), [[Code]](https://github.com/zlijingtao/ResSFL)  |
### - Miscellaneours MI Defenses
|         Method         |                                Paper                                 |                                                    Publication                                                     |   Source   |
|:----------------------:|:-------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|:--------:|
|  TL-DMI  |  Model Inversion Robustness: Can Transfer Learning Help?  |  CVPR'2024  |  [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Ho_Model_Inversion_Robustness_Can_Transfer_Learning_Help_CVPR_2024_paper.pdf), [[Code]](https://hosytuyen.github.io/projects/TL-DMI/)  |
|  BiDO  |  Bilateral Dependency Optimization: Defending Against Model-inversion Attacks  |  KDD'2022  |  [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3534678.3539376), [[Code]](https://github.com/AlanPeng0897/Defend_MI)  |
|  MID  |  Improving robustness to model inversion attacks via mutual information regularization.  |  AAAI'2021  |  [[Paper]](https://cdn.aaai.org/ojs/17387/17387-13-20881-1-2-20210518.pdf), [[Code]](https://github.com/Jiachen-T-Wang/mi-defense)  |


## 🫥Evaluation Metrics in MI Attacks and Defenses🫥

|         Method         |                                Paper                                 |                                                    Publication                                                     |   Source   |
|:----------------------:|:-------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|:--------:|
|  MSE  |  -  |  -  |  [[Link]](https://link.springer.com/book/9780387310732)  |
|  LPIPS  |  The Unreasonable Effectiveness of Deep Features as a Perceptual Metric  |  CVPR'2018  |  [[Paper]](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html)  |
|  PSNR  |   Visual distortion gauge based on discrimination of noticeable contrast changes  |  TCSVT'2005  |  [[Paper]](https://ieeexplore.ieee.org/document/1458831)  |
|  SSIM  |  Image quality assessment: from error visibility to structural similarity  |  ITIP'2004  |  [[Paper]](https://ieeexplore.ieee.org/document/1284395)  |
|  FID  |  GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium  |  NIPS'2017  |  [[Paper]](https://proceedings.neurips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html)  |
|  FSIM  |  FSIM: A feature similarity index for image quality assessment  |  ITIP'2011  |  [[Paper]](https://ieeexplore.ieee.org/document/5705575)  |
|  AVD  |  Absolute Variation Distance: An Inversion Attack Evaluation Metric for Federated Learning  |  ECIR'2024  |  [[Paper]](https://openreview.net/pdf?id=OoEIUohfcp)  |
|  RDLV  |  Do Gradient Inversion Attacks Make Federated Learning Unsafe?  |  TMI'2023  |  [[Paper]](https://ieeexplore.ieee.org/document/10025466)  |
|  IIP  |  See Through Gradients: Image Batch Recovery via GradInversion  |  CVPR'2021  |  [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Yin_See_Through_Gradients_Image_Batch_Recovery_via_GradInversion_CVPR_2021_paper.html)  |



## 😁Datasets for MI Attack Research😁
|         Dataset         |                                Name                                 |                                                    Publication                                                     |   Source   |
|:----------------------:|:-------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|:--------:|
|  MNIST  |  Modified National Institute of Standards and Technology database  |  -  |  [[Link]](https://yann.lecun.com/exdb/mnist/)  |
|  F-MNIST  |  Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms  |  arXiv'2017  |  [[Paper]](https://arxiv.org/abs/1708.07747), [[Link]](https://github.com/zalandoresearch/fashion-mnist)  |
|  CIFAR-10/CIFAR-100  |  Canadian Institute For Advanced Research  |  -  |  [[Link]](https://www.cs.toronto.edu/~kriz/cifar.html)  |
|  LFW   |  Labeled Faces in the Wild: A Database forStudying Face Recognition in Unconstrained Environments  |  INRIA'2008  |  [[Paper]](https://inria.hal.science/inria-00321923/document), [[Link]](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)  |
|  CelebA  |  Deep Learning Face Attributes in the Wild  |  ICCV'2015  |  [[Paper]](https://openaccess.thecvf.com/content_iccv_2015/html/Liu_Deep_Learning_Face_ICCV_2015_paper.html), [[Link]](https://github.com/jayantsolanki/CelebFace-data-analysis-using-Deep-Learning/blob/master/Deep%20learning%20face%20attributes%20in%20the%20wild.pdf)  |
|  ImageNet  |  ImageNet: A large-scale hierarchical image database  |  CVPR'2009  |  [[Paper]](https://ieeexplore.ieee.org/abstract/document/5206848), [[Link]](https://github.com/jiweibo/ImageNet)  |
|  FFHQ  |  A Style-Based Generator Architecture for Generative Adversarial Networks  |  PAMI'2021  |  [[Paper]](https://ieeexplore.ieee.org/document/8977347), [[Link]](https://github.com/NVlabs/stylegan)  |
|  ChestX-ray8  |  ChestX-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases  |  CVPR'2017  |  [[Paper]](https://ieeexplore.ieee.org/document/8099852), [[Link]](https://paperswithcode.com/dataset/chestx-ray8)  |
|  UBMD  |   A data-driven approach to predict the success of bank telemarketing  |  DSSs'2014  |  [[paper]](https://www.sciencedirect.com/science/article/pii/S016792361400061X?casa_token=LPZjxdKFl9QAAAAA:fRqXxSuxv9viR_TqgrTDkrwoJmXliRKpqvuEmuFhQQxUF6fctg8-3TyNaFalOlXKchlfdUT5oZfv), [[Link]](https://github.com/RistovaIvona/Bank-Marketing)  |
|  LDC  |  The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions  |  Sci Data'2018  |  [[Paper]](https://www.nature.com/articles/sdata2018161), [[Link]](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)  |
|  SVHN  |  Reading digits in natural images with unsupervised feature learning  |  NeurIPS'2011  |  [[Paper]](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37648.pdf), [[Link]](http://ufldl.stanford.edu/housenumbers/)  |


## 📂 Citation

If this resource repository helps to your research, we would greatly appreciate it if you could cite our papers below.

```bibtex
@article{Yang_Wang_Wu_Cai_Zhu_Wei_Zhang_Yang_Tang_Li_2025,
title={Deep learning model inversion attacks and defenses: a comprehensive survey},
volume={58}, ISSN={1573-7462}, DOI={10.1007/s10462-025-11248-0}, number={8},
journal={Artificial Intelligence Review},
author={Yang, Wencheng and Wang, Song and Wu, Di and Cai, Taotao and Zhu, Yanming and Wei, Shicheng and Zhang, Yiying and Yang, Xu and Tang, Zhaohui and Li, Yan},
year={2025}, month=may, pages={242}, language={en}}





