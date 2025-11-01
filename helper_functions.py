import os
import json
from typing import Dict, List

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Subset, DataLoader
from torch import nn

import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import v2


##################################################
# Seed Setting
##################################################
def set_seeds(seed:int=42):

    torch.manual_seed(seed=seed)
    torch.mps.manual_seed(seed=seed)

    return None


##################################################
# Dataset Builder
##################################################
def create_train_val_dataloaders(root_dir:str, 
                                 train_transformations:v2.Compose, 
                                 val_transformations:v2.Compose, 
                                 batch_size:int, 
                                 num_workers:int, 
                                 train_val_split:float=0.2):
    
    # Create two independent ImageFolder instances
    train_full_dataset = datasets.ImageFolder(root=root_dir, transform=train_transformations)
    val_full_dataset = datasets.ImageFolder(root=root_dir, transform=val_transformations)
    class_names = train_full_dataset.classes
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    total_samples = len(train_full_dataset)
    val_size = int(total_samples*train_val_split)
    train_size = total_samples-val_size
    print(f"Spitting dataset: Total={total_samples}, Train={train_size}, Validation={val_size}")
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_samples, generator=torch.Generator().manual_seed(42)).tolist()
    train_indices = indices[:train_size]
    val_indices=indices[train_size:]
    train_subset = Subset(train_full_dataset, train_indices)
    val_subset = Subset(val_full_dataset,val_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, class_names


##################################################
# Visualising datasets
##################################################
def visualize_dataset(dataset:DataLoader, 
                      class_names:List[str], 
                      num_images:int=32, 
                      cols:int=8, 
                      name:str=None):
    
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor, label_tensor = next(iter(dataset))
    rows = num_images//cols
    plt.figure(figsize=(cols*3, rows*3), dpi=300)
    for j in range(num_images):
        image = image_tensor[j] * IMAGENET_STD + IMAGENET_MEAN
        plt.subplot(rows, cols, j+1)
        plt.imshow(image.permute(1, 2, 0))
        plt.title(f"Label: {class_names[label_tensor[j]]}")
        plt.axis("off")
    if name is not None:
        plt.suptitle(name)
    plt.tight_layout()
    plt.show()

    return None


##################################################
# Plotting training curves
##################################################
def plot_loss_acc_curves(results:Dict[str, list], 
                         scatter:bool=True):
    
    train_loss, train_acc = results["train_loss"], results["train_acc"]
    val_loss, val_acc = results["val_loss"], results["val_acc"]
    epochs = torch.arange(len(train_loss)) + 1
    plt.figure(figsize=(18,6), dpi=300)
    plt.subplot(121)
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    if scatter: 
        plt.scatter(epochs, train_loss)
        plt.scatter(epochs, val_loss)
    plt.title("Loss Curve")
    plt.grid()
    plt.legend()
    plt.subplot(122)
    plt.plot(epochs, train_acc, label="Training Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    if scatter: 
        plt.scatter(epochs, train_acc)
        plt.scatter(epochs, val_acc)
    plt.title("Accuracy Curve")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    return None


##################################################
# Saving the model
##################################################
def save_model(model:nn.Module, 
               save_dir:str, 
               name:str="model"):
    
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}_weights.pth")
    torch.save(model.state_dict(), path)
    print(f"Save model to path:{path}")

    return None


##################################################
# Saving the training history
##################################################
def save_history(history:Dict[str, List], 
                 save_dir:str, 
                 name:str="training_history"):
    
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.json")
    with open(path, 'w') as f:
        json.dump(history, f)

    return None


##################################################
# Predict on test dataset and plot
##################################################
def pred_plot_dataset(dataset:DataLoader, 
                      model:nn.Module, 
                      class_names:List[str], 
                      device:torch.device, 
                      num_images:int=32, 
                      cols:int=8, 
                      name:str=None):
    
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cpu()
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cpu()
    rows = num_images//cols

    image_tensor, label_tensor_truth = next(iter(dataset))
    model.eval()
    model.to(device)
    with torch.inference_mode():
        label_tensor_output = model(image_tensor.to(device))
    label_tensor_pred_probs = torch.softmax(label_tensor_output, dim=1)
    label_tensor_pred = torch.argmax(label_tensor_pred_probs, dim=1).cpu()

    plt.figure(figsize=(cols*4, rows*4), dpi=300)
    for j in range(num_images):
        image = image_tensor[j] * IMAGENET_STD + IMAGENET_MEAN
        plt.subplot(rows, cols, j+1); plt.imshow(image.permute(1, 2, 0))
        true_label = class_names[label_tensor_truth[j]]; pred_label = class_names[label_tensor_pred[j]]
        title_text = f"True: {true_label}\nPred: {pred_label} (prob:{label_tensor_pred_probs.max().cpu():0.4f})"
        color = "green" if true_label==pred_label else "red"
        plt.title(title_text, color=color)
        plt.axis("off")
    if name is not None:
        plt.suptitle(name)
    plt.tight_layout()
    plt.show()
    
    return None