import torch
from typing import Tuple, Dict
from tqdm.auto import tqdm


def train_step(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader, loss_fn:torch.nn.Module, optimizer:torch.optim.Optimizer, 
               device:torch.device) -> Tuple[float, float]:
    train_loss, train_acc = 0, 0
    # Put the model in train mode
    model.train()

    for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Transfer to prefered device
        X, y = X.to(device), y.to(device)
        # Forward pass
        y_pred = model(X)
        # Loss calculation
        loss = loss_fn(y_pred, y)
        # Accumulate the loss
        train_loss += loss.item()
        #  📌📌📌 want to know what's happening.
        optimizer.zero_grad()  # 📌📌📌 From here didn't understand.
        # Bakward propagation
        loss.backward()
        #  📌📌📌 want to know what's happening.
        optimizer.step()
        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class==y).sum().item()/len(y_pred)
    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss/len(dataloader)
    train_acc = train_acc/len(dataloader)
    return train_loss, train_acc


def val_step(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader, loss_fn:torch.nn.Module, 
              device:torch.device) -> Tuple[float, float]:
    val_loss, val_acc = 0, 0
    # Put the model on evaluation mode
    model.eval()
    with torch.inference_mode(): # 📌📌📌 Didn't understand why train and val proces are different?
        for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
            X, y = X.to(device), y.to(device)
            y_pred_logits = model(X)
            loss = loss_fn(y_pred_logits, y)
            val_loss += loss.item()
            y_pred_labels = torch.argmax(y_pred_logits, dim=1)
            val_acc += (y_pred_labels==y).sum().item()/len(y_pred_labels)
    
    # Adjust metrics to get average loss and accuracy per batch
    val_loss = val_loss/len(dataloader)
    val_acc = val_acc/len(dataloader)
    return val_loss, val_acc


def train(model:torch.nn.Module, train_dataloader:torch.utils.data.DataLoader, val_dataloader:torch.utils.data.DataLoader,
          loss_fn:torch.nn.Module, optimizer:torch.optim.Optimizer, epoch:int, device:torch.device) -> Dict[str, list]:
    results = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    model.to(device=device)

    for e in tqdm(range(epoch), total=epoch):
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device)
        val_loss, val_acc = val_step(model=model, dataloader=val_dataloader, loss_fn=loss_fn, device=device)
        print(f"|| Epoch {e+1} Summary ||")
        print(f"train_loss: {train_loss:0.6f} | train_accuracy: {train_acc:.6f} | val_loss: {val_loss:.6f} | val_accuracy: {val_acc:.6f}")
        results["train_loss"].append(train_loss); results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss); results["val_acc"].append(val_acc)
    return results