import argparse
import torch
from statistics import mean
import os
import gdown

from model.wide_res_net import WideResNet
from model.PyramidNet import PyramidNet
from model.smooth_cross_entropy import smooth_crossentropy

from train import get_dataset


def get_model(model_name, data, device):
    
    if model_name == "WideResNet":
        batch_size = 256
        if data == "cifar100":
            model = WideResNet(data, 28, 10, 0, in_channels=3, labels=100).to(device)
            model_url = "https://drive.google.com/uc?id=1bhTIJF6bIsNukSry_bSlF0Co9K_MgoWH"
        elif data == "cifar10":
            model = WideResNet(data, 28, 10, 0, in_channels=3, labels=10).to(device)
            model_url = "https://drive.google.com/uc?id="
        elif data == "fashionmnist":
            model = WideResNet(data, 28, 10, 0, in_channels=1, labels=10).to(device)
            model_url = "https://drive.google.com/uc?id="
        elif data == "food101":
            model = WideResNet(data, 28, 10, 0, in_channels=3, labels=101).to(device)
            model_url = "https://drive.google.com/uc?id="
    elif model_name == "PyramidNet":
        batch_size = 64      
        if data == "cifar100":
            model = PyramidNet(data, depth=272, alpha=200, num_classes=100).to(device)
            model_url = "https://drive.google.com/uc?id=1SbLSAAMCobkW2ZJLNykqrQmxJuzmFMEU"
        elif data == "cifar10":
            model = PyramidNet(data, depth=272, alpha=200, num_classes=10).to(device)
            model_url = "https://drive.google.com/uc?id="
        elif data == "fashionmnist":
            model = PyramidNet(data, depth=272, alpha=200, num_classes=10).to(device)
            model_url = "https://drive.google.com/uc?id="
        elif data == "food101":
            model = PyramidNet(data, depth=272, alpha=200, num_classes=101).to(device)
            model_url = "https://drive.google.com/uc?id="
            
    checkpoint_path = f'./pretrained_weight/AACE_{model_name}_{data}.pth'
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        if not os.path.exists("./pretrained_weight/"):
            os.makedirs("./pretrained_weight/")
        print(f"Checkpoint not found at {checkpoint_path}, downloading...")
        gdown.download(model_url, checkpoint_path, quiet=False)
        print(f"Checkpoint downloaded to {checkpoint_path}")
        
    model.load_state_dict(torch.load(checkpoint_path))
    
    return model, batch_size
            

def test(model_name, dataset_name, gpu, threads):
    
    device = torch.device(gpu)
    model, batch_size = get_model(model_name, dataset_name, device)
    dataset = get_dataset(dataset_name, batch_size, threads)
    
    model.eval()
    
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataset.test:
            inputs, targets = (b.to(device) for b in batch)

            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets)
            total_loss += loss.sum().item()

            correct = torch.argmax(predictions, 1) == targets
            correct_predictions += correct.sum().item()
            total_samples += inputs.size(0)
    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    print(f'Average Loss: {average_loss:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model Architecture")
    parser.add_argument("--dataset", type=str, help="Dataset for training")
    parser.add_argument("--gpu", type=str, default="cuda:3", help="GPU device to use.")
    parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loading.")
    
    args = parser.parse_args()
    test(args.model, args.dataset, args.gpu, args.threads)
    