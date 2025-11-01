import torch, torchvision
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

##################################################
# Seed Setting
##################################################
def set_seeds(seed:int=42):
    torch.manual_seed(seed=seed)
    torch.mps.manual_seed(seed=seed)

##################################################
# Visualising datasets
##################################################
def visualize_dataset(dataset:torch.utils.data.DataLoader, class_names:List[str], num_images:int=32, cols:int=8, name:str=None):
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor, label_tensor = next(iter(dataset))
    rows = num_images//cols

    plt.figure(figsize=(cols*3, rows*3), dpi=300)
    for j in range(num_images):
        image = image_tensor[j] * IMAGENET_STD + IMAGENET_MEAN
        plt.subplot(rows, cols, j+1); plt.imshow(image.permute(1, 2, 0))
        plt.title(f"Label: {class_names[label_tensor[j]]}"); plt.axis("off")
    if name is not None:
        plt.suptitle(name)
    plt.tight_layout()
    plt.show()

##################################################
# Plotting training curves
##################################################
def plot_loss_acc_curves(results:Dict[str, list], scatter:bool=True):
    train_loss, train_acc = results["train_loss"], results["train_acc"]
    val_loss, val_acc = results["val_loss"], results["val_acc"]
    epochs = torch.arange(len(train_loss)) + 1

    plt.figure(figsize=(18,6), dpi=300)

    plt.subplot(121)
    plt.plot(epochs, train_loss, label="Training Loss"); plt.plot(epochs, val_loss, label="Validation Loss")
    if scatter: plt.scatter(epochs, train_loss); plt.scatter(epochs, val_loss)
    plt.title("Loss Curve"); plt.grid(); plt.legend()

    plt.subplot(122)
    plt.plot(epochs, train_acc, label="Training Accuracy"); plt.plot(epochs, val_acc, label="Validation Accuracy")
    if scatter: plt.scatter(epochs, train_acc); plt.scatter(epochs, val_acc)
    plt.title("Accuracy Curve"); plt.grid(); plt.legend()

    plt.tight_layout()
    plt.show()

##################################################
# Saving the model
##################################################
def save_model(model:torch.nn.Module, path:str):
    print(f"Save model to path:{path}")

##################################################
# Predict a target image and plot
##################################################
def pred_plot_target_image(image_path:str, class_names:List[str], device:torch.device, model:torch.nn.Module, 
                           transform:torchvision.transforms=None, image_size:int=224):
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    target_image = (target_image-torch.min(target_image))/(torch.max(target_image) - torch.min(target_image))

    if transform:
        target_image = transform(target_image)
    
    model.to(device=device)
    model.eval()
    with torch.inference_mode():
        target_image = target_image.unsqueeze(dim=0)
        target_image_pred = model(target_image.to(device=device))
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    plt.imshow(target_image.squeeze().permute(1,2,0))
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.4f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.4f}"
    plt.title(title)
    plt.axis("off")