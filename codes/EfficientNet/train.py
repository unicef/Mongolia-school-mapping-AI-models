import torch
import torch.nn.functional as F

import os
from tqdm import tqdm


def train_one_epoch(model, trainloader, optimizer, criterion, device):
    
    model.train()
    
    train_running_loss = 0
    train_running_acc = 0
    
    for idx, (x, y) in enumerate(trainloader):
        
        x, y = x.to(device), y.to(device)
        
        preds = model(x)
        loss = criterion(preds, y)
        
        train_running_loss += loss.item()
        train_running_acc += (torch.max(preds.data,1)[-1] == y).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    epoch_loss = train_running_loss / len(trainloader)
    epoch_acc = train_running_acc / len(trainloader.dataset)
    
    return epoch_loss, epoch_acc


def train(epochs, model, train_dataloader, val_dataloader, optimizer, criterion, device, save_folder):
    
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    best_val_loss = 100000
    best_val_acc = 0

    for epoch in tqdm(range(epochs)):

        train_epoch_loss, train_epoch_acc = train_one_epoch(model, train_dataloader, optimizer, criterion, device)
        val_epoch_loss, val_epoch_acc = validate(model, val_dataloader, criterion, device)

        train_loss.append(train_epoch_loss)
        valid_loss.append(val_epoch_loss)

        train_acc.append(train_epoch_acc)
        valid_acc.append(val_epoch_acc)
        
        print('train loss: {:.4f}, val loss: {:.4f}'.format(train_epoch_loss, val_epoch_loss))
        print('train acc: {:.4f}, val acc: {:.4f}'.format(train_epoch_acc, val_epoch_acc))

        # save best model on val loss
        if val_epoch_loss < best_val_loss:
            print(f"Loss decreased from {best_val_loss} to {val_epoch_loss}")
            print(f"Saving model at epoch: {epoch + 1}")
            best_val_loss = val_epoch_loss

            save_model(save_dir = os.path.join(save_folder, f'best_model_on_val_loss_epoch_{epoch}.pth'), 
               epochs = epoch + 1, 
               model = model,
               optimizer = optimizer, 
               criterion = criterion)

        # save best model on val accuracy
        if val_epoch_acc > best_val_acc:
            print(f"Accuracy improved from {best_val_acc} to {val_epoch_acc}")
            print(f"Saving model at epoch: {epoch + 1}")
            best_val_acc = val_epoch_acc

            save_model(save_dir = os.path.join(save_folder, f'best_model_on_val_acc_epoch_{epoch}.pth'), 
               epochs = epoch + 1, 
               model = model,
               optimizer = optimizer, 
               criterion = criterion)
    
    # save last epoch model
    save_model(save_dir = os.path.join(save_folder, 'last.pth'),
                epochs = epoch + 1, 
                model = model, 
                optimizer = optimizer, 
                criterion = criterion)
    
    # history dictionary
    history = {"train_loss": train_loss,
               "train_acc": train_acc,
               "val_loss": valid_loss,
               "val_acc": valid_acc}
    
    return history


def validate(model, dataloader, criterion, device):
    
    model.eval()
    
    running_loss = 0
    running_acc = 0
    
    for x,y in dataloader:
        
        x,y = x.to(device), y.to(device)
        
        preds = model(x)
        
        loss = criterion(preds, y)
        
        running_loss += loss.item()
        running_acc += (torch.max(preds.data, 1)[-1] == y).sum().item()
        
    val_loss = running_loss / len(dataloader)
    val_acc = running_acc / len(dataloader.dataset)
    
    return val_loss, val_acc


def predict(model, dataloader, device, return_probs = True):
    
    all_predictions = []
    all_gts = []
    
    model.eval()
    
    for x, y in tqdm(dataloader):
        
        x,y = x.to(device), y.to(device)
        
        preds = model(x)
        preds = F.softmax(preds, dim = 1)
        
        if not return_probs: 
            preds = torch.argmax(preds, 1)
            
        preds = preds.detach().cpu().numpy()
        
        all_predictions += list(preds)
        all_gts += list(y.detach().cpu().numpy())
        
    return all_predictions, all_gts


def save_model(save_dir, epochs, model, optimizer, criterion):
    
    torch.save({'epochs': epochs,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': criterion},
                save_dir)
