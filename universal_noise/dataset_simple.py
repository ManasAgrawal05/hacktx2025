import os
import random
from typing import Dict, List, Tuple
from PIL import Image
from PIL import UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FaceDataset(Dataset):
    def __init__(self, image_paths: List[Tuple[str, str]], transform=None):
        """
        image_paths: list of (img_path, person_name)
        """
        self.image_paths = image_paths
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, person = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), person


def gather_images(root_dir: str) -> Dict[str, List[str]]:
    """
    Returns a dict mapping person_name -> list of valid image paths.
    Skips macOS metadata files (like '._*').
    """
    people = {}
    for person in os.listdir(root_dir):
        person_path = os.path.join(root_dir, person)
        if not os.path.isdir(person_path):
            continue

        imgs = []
        for f in os.listdir(person_path):
            if f.startswith("._"):  # skip macOS resource fork files
                continue
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            full_path = os.path.join(person_path, f)
            imgs.append(full_path)

        if imgs:
            people[person] = imgs
    return people


def split_people(people: Dict[str, List[str]], val_ratio: float = 0.2):
    """
    Splits people into train and validation sets (by person).
    """
    all_people = list(people.keys())
    random.shuffle(all_people)
    val_size = int(len(all_people) * val_ratio)
    val_people = set(all_people[:val_size])
    train_people = set(all_people[val_size:])

    train_imgs = [(img, p) for p in train_people for img in people[p]]
    val_imgs = [(img, p) for p in val_people for img in people[p]]

    return train_imgs, val_imgs


def build_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    num_workers: int = 2
):
    """
    Loads all images, splits by person, returns train and val dataloaders.
    """
    people = gather_images(root_dir)
    train_imgs, val_imgs = split_people(people, val_ratio)

    print(f"Found {len(people)} people total.")
    print(f"Train: {len(train_imgs)} images from {len(set([p for _, p in train_imgs]))} people")
    print(f"Val: {len(val_imgs)} images from {len(set([p for _, p in val_imgs]))} people")

    train_ds = FaceDataset(train_imgs)
    val_ds = FaceDataset(val_imgs)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = build_dataloaders("lfw-deepfunneled")
