import os
import torch
import random
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from PIL import Image
from dataset_loader import get_eurosat_dataloaders
from models.simple_cnn import SimpleCNN

def denormalize(tensor): #Undo the normalization for visualization.
    mean = torch.tensor([0.3444, 0.3809, 0.4082]).view(3,1,1)
    std = torch.tensor([0.1818, 0.1317, 0.1205]).view(3,1,1)
    return (tensor * std + mean).clamp(0,1)

def explain_multiple_samples(num_samples=20):
    model = SimpleCNN(num_classes=10)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()

    cam_extractor = GradCAM(model, target_layer="conv2")

    _, test_loader, class_names = get_eurosat_dataloaders()
    test_images, test_labels = next(iter(test_loader))  # one batch

    indices = random.sample(range(len(test_images)), min(num_samples, len(test_images)))

    for i, idx in enumerate(indices):
        image = test_images[idx]
        label = test_labels[idx].item()
        label_name = class_names[label]

        output = model(image.unsqueeze(0))
        pred_class = output.argmax(dim=1).item()
        pred_name = class_names[pred_class]

        activation_map = cam_extractor(pred_class, output)

        image_vis = denormalize(image).permute(1, 2, 0).numpy()
        heatmap = activation_map[0].squeeze().numpy()
        result = overlay_mask(Image.fromarray((image_vis * 255).astype('uint8')),
                              Image.fromarray((heatmap * 255).astype('uint8')),
                              alpha=0.5)

        save_dir = os.path.join("results", "gradcam", label_name)
        os.makedirs(save_dir, exist_ok=True)

        filename = f"sample{i+1}_true-{label_name}_pred-{pred_name}.png"
        save_path = os.path.join(save_dir, filename)
        result.save(save_path)

        print(f"Saved {filename}")

if __name__ == "__main__":
    explain_multiple_samples(num_samples=20)
