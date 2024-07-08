# ADAPTIVE ADVERSARIAL CROSS-ENTROPY (AACE)
 
This repository contains reproduction code for the research paper titled **"ADAPTIVE ADVERSARIAL CROSS-ENTROPY LOSS FOR SHARPNESS-AWARE MINIMIZATION"**.  
This technique helps improve model generalization and performance in image classification tasks.

The paper has been accepted for ICIP2024  
Project Page: http://www.vip.sc.e.titech.ac.jp/proj/AACE/AACE.html  
arXiv: https://arxiv.org/abs/2406.14329

## Training from Scratch
Use this command to train a model from scratch using SAM with AACE.  

```bash
python train.py --model <MODEL_NAME> --dataset <DATASET_NAME> --rho <RHO>
```
MODEL_NAME: "WideResNet", "PyramidNet"  
DATASET_NAME: "cifar100", "cifar10", "fashionmnist", "food101"  
RHO: rho value (default = 0.2)

See **train.py** for more parameters detail

## Inferencing Using Pre-Treained Weight
To obtain the results as shown in Table.2 and Table.3 in the paper, run the following command for inferencing using a pre-trained model trained by SAM with AACE.


```bash
python test.py --model <MODEL_NAME> --dataset <DATASET_NAME>
```

MODEL_NAME: "WideResNet", "PyramidNet"  
DATASET_NAME: "cifar100", "cifar10", "fashionmnist", "food101"

See **test.py** for more parameters detail

## Citation
Tanapat Ratchatorn and Masayuki Tanaka, **“Adaptive Adversarial Cross-Entropy Loss for Sharpness-Aware Minimization”**, IEEE International Conference on Image Processing (ICIP), October, 2024.
