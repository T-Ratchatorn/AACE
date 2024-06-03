# AACE
ADAPTIVE ADVERSARIAL CROSS-ENTROPY LOSS FOR SHARPNESS-AWARE MINIMIZATION

## Usage

To train a model using the SAM with AACE training script, you need to specify the model architecture, dataset, rho. 
You can use the following command:

```bash
python train.py --model <MODEL_NAME> --dataset <DATASET_NAME> --rho <RHO>

MODEL_NAME: "WideResNet", "PyramidNet"
DATASET_NAME: "cifar100", "cifar10", "fashionmnist", "food101"
RHO: rho value (default = 0.2)