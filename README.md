# ADAPTIVE ADVERSARIAL CROSS-ENTROPY (AACE)
 
This repository contains reproduction code for the research paper titled **"ADAPTIVE ADVERSARIAL CROSS-ENTROPY LOSS FOR SHARPNESS-AWARE MINIMIZATION"**.  
This technique helps improve model generalization and performance in image classification tasks.

The paper has been accepted and presented at ICIP2024  
Project Page: http://www.vip.sc.e.titech.ac.jp/proj/AACE/AACE.html  
Paper Link: https://ieeexplore.ieee.org/document/10647582  
arXiv: https://arxiv.org/abs/2406.14329

## Training from Scratch
Use this command to train a model from scratch using SAM with AACE.  

```bash
python train.py --model <MODEL_NAME> --dataset <DATASET_NAME> --rho <RHO>
```
MODEL_NAME: "WideResNet", "PyramidNet"  
DATASET_NAME: "cifar100", "cifar10", "fashionmnist", "food101"  
RHO: rho value (default = 0.2)

For additional details on all parameters, please see [train.py](train.py)

## Inferencing Using Pre-Treained Weight
To obtain the results as shown in Table.2 and Table.3 in the paper, run the following command for inferencing using a pre-trained model trained by SAM with AACE.


```bash
python test.py --model <MODEL_NAME> --dataset <DATASET_NAME>
```

MODEL_NAME: "WideResNet", "PyramidNet"  
DATASET_NAME: "cifar100", "cifar10", "fashionmnist", "food101"

For additional details on all parameters, please see [test.py](test.py)

## Citation
Tanapat Ratchatorn and Masayuki Tanaka, **“Adaptive Adversarial Cross-Entropy Loss for Sharpness-Aware Minimization”**, IEEE International Conference on Image Processing (ICIP), October, 2024.

```
@INPROCEEDINGS{10647582,
  author={Ratchatorn, Tanapat and Tanaka, Masayuki},
  booktitle={2024 IEEE International Conference on Image Processing (ICIP)}, 
  title={Adaptive Adversarial Cross-Entropy Loss for Sharpness-Aware Minimization}, 
  year={2024},
  pages={479-485},
  doi={10.1109/ICIP51287.2024.10647582}}
```

