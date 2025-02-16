import os
import glob
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ISICSegmentationDataset(Dataset):
    """
    Expects:
      - image_dir: folder with images like ISIC_XXXXXXX.jpg
      - mask_dir:  folder with corresponding masks like ISIC_XXXXXXX_Segmentation.png
    """
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(228, 228)):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        
        # Look for .jpg images
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        if len(self.image_paths) == 0:
            print(f"[Warning] No .jpg images found in {image_dir}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)  # e.g. "ISIC_0000000.jpg"
        mask_name = filename.replace(".jpg", "_Segmentation.png")
        mask_path = os.path.join(self.mask_dir, mask_name)
        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")
        image = image.resize(self.target_size, Image.BILINEAR)
        mask  = mask.resize(self.target_size, Image.NEAREST)
        image = np.array(image, dtype=np.float32) / 255.0
        mask  = np.array(mask, dtype=np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        mask  = np.expand_dims(mask, axis=0)
        if self.transform:
            image, mask = self.transform(image, mask)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
    
if __name__ == "__main__":
    train_img_dir  = "images/training_data"
    train_mask_dir = "images/training_gt"
    test_img_dir   = "images/test_data"
    test_mask_dir  = "images/test_gt"
    batch_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = ISICSegmentationDataset(train_img_dir, train_mask_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = ISICSegmentationDataset(test_img_dir, test_mask_dir)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = torch.hub.load('facebookresearch/capi:main', 'capi_vitl14_lvd').to(device)
    full_features = []
    full_masks = []
    with torch.no_grad():
        for images, masks in tqdm(train_loader):
            images = images.to(device)
            masks  = masks.to(device)
            with torch.cuda.amp.autocast():
                _, _, features = model(images)
            full_features.append(features.flatten(start_dim=1).cpu())
            full_masks.append(masks.flatten(start_dim=1).cpu())
    full_features = torch.cat(full_features, dim=0)
    full_masks = torch.cat(full_masks, dim=0)
    torch.save(full_features.cpu(), "full_features.pt")
    torch.save(full_masks.cpu(), "full_masks.pt")
