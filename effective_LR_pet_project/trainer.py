import torch
import json 
from dataloaders import dataloaders
from resnet18_model import model
from metrics import compute_accuracy
from utils import *

with open("config.conf", "r") as f_json:
    config = json.loads(f_json.read())

is_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if is_gpu else "cpu") # Let's make sure GPU is available!

set_seed(config["seed"]) #Making experiment reproducible

if is_gpu:
    model.type(torch.cuda.FloatTensor)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().type(torch.cuda.FloatTensor)
else:
    criterion = torch.nn.CrossEntropyLoss()
    
debug_mode = False

def train_model(model, train_loader, val_loader, loss, optimizer, scheduler, log_name, init_lr, num_epochs):
    for epoch in range(num_epochs):
        model.train() 
        loss_accum = 0
        correct_samples = 0
        total_samples = 0
        for i_step, (x, y) in enumerate(train_loader):
            x_gpu = x.to(device)
            y_gpu = y.to(device)
            prediction = model(x_gpu)    
            loss_value = loss(prediction, y_gpu)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
         
            _, indices = torch.max(prediction, 1)
            correct_samples += torch.sum(indices == y_gpu)
            total_samples += y.shape[0]
            loss_accum += loss_value
        
        if scheduler: #it means now is default scheduler mode
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
        else: #it means now is effective learning rate mode
            # compute effective lr and reinitialize optimizer for the next epoch
            optimizer, current_lr = eff_lr(model, optimizer, init_lr)

        ave_loss = loss_accum / i_step
        train_accuracy = float(correct_samples) / total_samples
        val_accuracy = compute_accuracy(model, val_loader)

        with open(log_name, 'a') as f:
            f.write(f'{ave_loss} {train_accuracy } {val_accuracy} {current_lr}\n')
                   
        if epoch % 1000 == 0:
            print(f'{epoch}th has finished')
        
trainloader, testloader = dataloaders(config)

if __name__ == '__main__':
    for i in range(1, config["num_exp"]+1):
        print(f"Experiment {i} starts")
        config_ = config["experiments"][f"exp{i}"]
        optimizer = get_optimizer(model.parameters(), config_)
        if config_["sched"]:
            scheduler = get_scheduler(optimizer, config_)
            log_name = f'{config_["optim"]}_{config_["sched"]}_{config_["lr"]}lr_{config_["reg"]}reg.txt'
        else:
            log_name = f'{config_["lr"]}lr_{config_["reg"]}reg.txt'
        train_model(model, trainloader, testloader, criterion, 
                    optimizer, None, log_name, config_["lr"], config["epochs"])
        print(f"Experiment {i} is ready!")

    