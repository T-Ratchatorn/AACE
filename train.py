import argparse
import torch
from statistics import mean
import csv

from model.wide_res_net import WideResNet
from model.PyramidNet import PyramidNet

from model.smooth_cross_entropy import smooth_crossentropy

from data_cifar100.cifar import Cifar100
from data_cifar10.cifar import Cifar10
from data_fashionmnist.fashionmnist import fashionmnist
from data_food101.food101 import Food101

from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from adversarial_cross_entropy import AdaptiveAdversrialCrossEntropy


def get_experiment_parameters(experiment, base_optimizer, data, device, rho):
    if experiment == "WideResNet":
        if data == "cifar100"
            model = WideResNet(dataset=data, 28, 10, 0, in_channels=3, labels=100).to(device)
            optimizer = SAM(model.parameters(), base_optimizer, rho=rho, lr=0.1, momentum=0.9, weight_decay=0.0005)
            scheduler = StepLR(optimizer, 0.1, 200)
            batch_size, epochs = 256, 200
        elif data == "cifar10":
            model = WideResNet(dataset=data, 28, 10, 0, in_channels=3, labels=10).to(device)
            optimizer = SAM(model.parameters(), base_optimizer, rho=rho, lr=0.1, momentum=0.9, weight_decay=0.0005)
            scheduler = StepLR(optimizer, 0.1, 200)
            batch_size, epochs = 256, 200
        elif data == "fashionmnist":
            model = WideResNet(dataset=data, 28, 10, 0, in_channels=1, labels=10).to(device)
            optimizer = SAM(model.parameters(), base_optimizer, rho=rho, lr=0.1, momentum=0.9, weight_decay=0.0005)
            scheduler = StepLR(optimizer, 0.1, 200)
            batch_size, epochs = 256, 200
        elif data == "food101":
            model = WideResNet(dataset=data, 28, 10, 0, in_channels=3, labels=101).to(device)
            optimizer = SAM(model.parameters(), base_optimizer, rho=rho, lr=0.1, momentum=0.9, weight_decay=0.0005)
            scheduler = StepLR(optimizer, 0.1, 200)
            batch_size, epochs = 256, 200
    elif experiment == "PyramidNet":
        if data == "cifar100"
            model = PyramidNet(dataset=data, depth=272, alpha=200, num_classes=100).to(device)
            optimizer = SAM(model.parameters(), base_optimizer, rho=rho, lr=0.1, momentum=0.9, weight_decay=0.0005)
            scheduler = StepLR(optimizer, 0.1, 200)
            batch_size, epochs = 64, 200
        elif data == "cifar10":
            model = PyramidNet(dataset=data, depth=272, alpha=200, num_classes=10).to(device)
            optimizer = SAM(model.parameters(), base_optimizer, rho=rho, lr=0.1, momentum=0.9, weight_decay=0.0005)
            scheduler = StepLR(optimizer, 0.1, 200)
            batch_size, epochs = 64, 200
        elif data == "fashionmnist":
            model = PyramidNet(dataset=data, depth=272, alpha=200, num_classes=10).to(device)
            optimizer = SAM(model.parameters(), base_optimizer, rho=rho, lr=0.1, momentum=0.9, weight_decay=0.0005)
            scheduler = StepLR(optimizer, 0.1, 200)
            batch_size, epochs = 64, 200
        elif data == "food101":
            model = PyramidNet(dataset=data, depth=272, alpha=200, num_classes=101).to(device)
            optimizer = SAM(model.parameters(), base_optimizer, rho=rho, lr=0.1, momentum=0.9, weight_decay=0.0005)
            scheduler = StepLR(optimizer, 0.1, 200)
            batch_size, epochs = 64, 200
    return model, optimizer, scheduler, batch_size, epochs


def get_dataset(data, batch_size, threads):
    if data == "cifar100":
        return Cifar100(batch_size, threads)
    elif data == "cifar10":
        return Cifar10(batch_size, threads)
    elif data == "fashionmnist":
        return fashionmnist(batch_size, threads)
    elif data == "food101":
        return Food101(batch_size, threads)


def train(model_name, dataset_name, rho, gpu, threads, use_grad_norm, result_dir):
    experiment = model_name
    data = dataset_name

    file_name = f"AACE_GradNorm:{use_grad_norm}_rho:{rho}"
    optimizer_save_path = result_dir + "/" + file_name + "_opt.pth"
    model_save_path = result_dir + "/" + file_name + "_model.pth"
    csv_path = result_dir + "/" + file_name + ".csv"

    
    header = ["epoch", "lr", "avg_ce_loss", "avg_aace_loss", "avg_grad_norm", "avg_perturbation", "train_loss", "train_acc", "val_loss", "val_acc"]
    with open(csv_path, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(header)
    
    initialize(seed=42)
    device = torch.device(gpu)

    log = Log(log_each=10)
    
    base_optimizer = torch.optim.SGD
    
    model, optimizer, scheduler, batch_size, epochs = get_experiment_parameters(experiment, base_optimizer, data, device, rho)
    dataset = get_dataset(data, batch_size, threads)
        
    for epoch in range(epochs):
        result_list = []
        model.train()
        log.train(len_dataset=len(dataset.train))
        
        grad_norm_list = []
        ew_norm_list = []
        
        sum_ce_loss = 0
        sum_perturb_loss = 0
        sum_loss_num = 0
        
        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            # first forward-backward step
            enable_running_stats(model)
            predictions = model(inputs)
            
            ce_loss = smooth_crossentropy(predictions, targets) #ce_loss is not used, just calculate it for observation purpose
            sum_ce_loss += ce_loss.sum().item()
            
            loss_function = AdaptiveAdversrialCrossEntropy()
            perturb_loss = loss_function(predictions, targets)
            perturb_loss.mean().backward()
            sum_perturb_loss += perturb_loss.sum().item()
    
            sum_loss_num += perturb_loss.size(0)
            
            grad_norm, ew_norm = optimizer.first_step(use_grad_norm=use_grad_norm, zero_grad=True)
            ew_norm_list.append(ew_norm)
            grad_norm_list.append(grad_norm)
            
            # second forward-backward step
            disable_running_stats(model)
            loss = smooth_crossentropy(model(inputs), targets)
            loss.mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)
        
        avg_ce_loss = sum_ce_loss/sum_loss_num
        avg_perturb_loss = sum_perturb_loss/sum_loss_num
        avg_ew_norm = mean(ew_norm_list)
        avg_grad_norm = mean(grad_norm_list)
        
        model.eval()
        train_loss, train_acc, lr = log.output_0()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())
        
        val_loss, val_acc = log.output_1()
        log.flush()

        result_list.extend([epoch, lr, avg_ce_loss, avg_perturb_loss, avg_grad_norm, avg_ew_norm, train_loss, train_acc, val_loss, val_acc])
        with open(csv_path, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(result_list)
    
    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model Architecture")
    parser.add_argument("--dataset", type=str, help="Dataset for training")
    parser.add_argument("--rho", type=int, default=2.0, help="Rho parameter for SAM.")
    parser.add_argument("--gpu", type=str, default="cuda:3", help="GPU device to use.")
    parser.add_argument("--threads", type=int, default=36, help="Number of threads for data loading.")
    parser.add_argument("--use_grad_norm", type=bool, default=False, help="Use gradient norm.")
    parser.add_argument("--result_dir", type=str, default="./results", help="Directory to save results.")
    
    args = parser.parse_args()
    train(args.model, args.dataset, args.rho, args.gpu, args.threads, args.use_grad_norm, args.result_dir)
    