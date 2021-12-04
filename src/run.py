import time
import wandb
import torch
import copy

from src.metrics import total_corrrect


def _save_model(model, save_dir, is_checkpoint=False):    
    
    if is_checkpoint:
        torch.save(model.state_dict(), save_dir / "best_model.pth.tar")
        
    else:
        torch.save(model.state_dict(), save_dir / "last_model.pth.tar")


def _train_epoch(model, dataloader, criterion, optimiser, device):
    
    model.train()

    epoch_loss = 0
    epoch_acc = 0
 
    for x, y in dataloader:
        # transfer signal, y to device
        x, y = x.to(device), y.to(device).reshape(-1)
        # clear gradients of model parameters
        optimiser.zero_grad()
        # forward pass
        logits = model(x)          
        # calculate metrics
        loss = criterion(logits, y)
        correct = total_corrrect(logits, y)
        # backward pass
        loss.backward()
        # update model parameters
        optimiser.step()
        # accumulate loss over batch
        epoch_loss += loss.item() / len(dataloader)
        epoch_acc += (100 * correct.item()) / len(dataloader)

        break

    return epoch_loss, epoch_acc


def _valid_epoch(model, dataloader, criterion, device):
    
    model.eval()

    epoch_loss = 0
    epoch_acc = 0
 
    for x, y in dataloader:
        # transfer x, y to device
        x, y = x.to(device), y.to(device).reshape(-1)
        # do not calculate gradients
        with torch.no_grad():
            # forward pass
            logits = model(x)
        # calculate metrics
        loss = criterion(logits, y)
        correct = total_corrrect(logits, y)
        # accumulate loss over batch
        epoch_loss += loss.item() / len(dataloader)
        epoch_acc += (100 * correct.item()) / len(dataloader)

        break

    return epoch_loss, epoch_acc
    

def run(model, train_loader, valid_loader, criterion, optimiser, scheduler, num_epochs, save_dir, device=torch.device("cpu"), wb_logging=False):
    
    if wb_logging: wandb.watch(model)

    train_time = 0.
    best_valid_acc = -1.

    for epoch in range(num_epochs):
        start_time = time.time() 

        train_loss, train_acc = _train_epoch(model, train_loader, criterion, optimiser, device)
        valid_loss, valid_acc = _valid_epoch(model, valid_loader, criterion, device)

        is_best = valid_acc > best_valid_acc
        if is_best:
            best_valid_acc = valid_acc
            _save_model(model, save_dir, is_checkpoint=True)

        if scheduler is not None:
            scheduler.step()

        end_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        to_print = "{}  |  epoch {:4d} of {:4d}  |  train loss {:06.3f}  |  train acc {:05.2f}  |  valid loss {:06.3f}  |  valid acc {:05.2f}  |  time: {}  "
        if is_best: to_print = to_print + "|  *" 
        print(to_print.format(save_dir.stem, epoch + 1, num_epochs, train_loss, train_acc, valid_loss, valid_acc, end_time))

        if wb_logging: wandb.log(dict(train={"loss": train_loss, "acc": train_acc}, valid={"loss": valid_loss, "acc": valid_acc}))
    
    _save_model(model, save_dir, train_time=train_time)
    