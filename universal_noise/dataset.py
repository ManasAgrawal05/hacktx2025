import os
import random
import csv
from itertools import combinations, product
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Tuple, Dict


from PIL import Image
from collections import Counter
import os


def check_image_dimensions(root_dir: str, max_samples: int = None):
    """
    Scans all image files under root_dir and prints the unique dimension configurations.
    If max_samples is given, it limits how many images are checked for speed.
    """
    dimensions = Counter()
    count = 0

    for person in os.listdir(root_dir):
        person_path = os.path.join(root_dir, person)
        if not os.path.isdir(person_path):
            continue
        for img_file in os.listdir(person_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(person_path, img_file)
            try:
                with Image.open(img_path) as img:
                    dimensions[img.size] += 1  # (width, height)
            except Exception as e:
                print(f"⚠️ Error reading {img_path}: {e}")
            count += 1
            if max_samples and count >= max_samples:
                break
        if max_samples and count >= max_samples:
            break

    print(f"\nScanned {count} images.")
    print(f"Found {len(dimensions)} unique image dimension configurations:\n")
    for dim, freq in dimensions.most_common():
        print(f"  {dim}: {freq} images")


class TripletDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str, int]], transform=None):
        """
        pairs: list of (img1_path, img2_path, label)
               label = 1 for match, 0 for mismatch
        """
        self.pairs = pairs
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        return self.transform(img1), self.transform(img2), label


def read_people_csv(csv_path: str) -> Dict[str, int]:
    people = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('name', '').strip()
            images = row.get('images', '').strip()
            if not name or not images:
                print(f"Skipping invalid row: {row}")
                continue
            try:
                people[name] = int(images)
            except ValueError:
                print(f"Skipping row with non-integer image count: {row}")
                continue
    return people


def build_pairs(people: Dict[str, int], root_dir: str, limit: int):
    """Generate match and mismatch pairs up to a limit for a given split."""
    match_pairs = []
    mismatch_pairs = []

    # --- MATCH PAIRS ---
    for person, count in people.items():
        if count > 1:
            img_paths = [
                os.path.join(
                    root_dir,
                    person,
                    f"{person}_{i+1:04d}.jpg") for i in range(count)]
            match_pairs.extend(list(combinations(img_paths, 2)))

    # --- MISMATCH PAIRS ---
    people_list = list(people.keys())
    all_imgs = {
        p: [os.path.join(root_dir, p, f"{p}_{i+1:04d}.jpg") for i in range(n)]
        for p, n in people.items()
    }

    while len(mismatch_pairs) < limit:
        p1, p2 = random.sample(people_list, 2)
        img1 = random.choice(all_imgs[p1])
        img2 = random.choice(all_imgs[p2])
        mismatch_pairs.append((img1, img2))

    random.shuffle(match_pairs)
    match_pairs = [(a, b, 1) for a, b in match_pairs[:limit]]
    mismatch_pairs = [(a, b, 0) for a, b in mismatch_pairs[:limit]]

    total_match_possible = sum((n * (n - 1)) // 2 for n in people.values() if n > 1)
    total_mismatch_possible = 0
    counts = list(people.values())
    for i in range(len(counts)):
        for j in range(i + 1, len(counts)):
            total_mismatch_possible += counts[i] * counts[j]

    return match_pairs, mismatch_pairs, total_match_possible, total_mismatch_possible


def split_people(people: Dict[str, int], val_size: int = 100, val_limit: int = 5000):
    """
    Split people into train/val ensuring the val set can produce enough match pairs.
    The val split will only include people with >=2 images, and we keep adding people
    until their total theoretical match pairs (nC2) >= val_limit.
    """
    # Separate groups
    multi_image_people = {p: n for p, n in people.items() if n > 1}
    single_image_people = {p: n for p, n in people.items() if n <= 1}

    # Compute theoretical matches for each person
    candidates = list(multi_image_people.items())
    random.shuffle(candidates)

    val_people = {}
    total_possible_matches = 0
    for p, n in candidates:
        val_people[p] = n
        total_possible_matches += (n * (n - 1)) // 2
        if len(val_people) >= val_size and total_possible_matches >= val_limit:
            break

    # In case we still didn’t reach enough matches
    if total_possible_matches < val_limit:
        print(f"⚠️ Not enough match pairs even after selecting {len(val_people)} people "
              f"({total_possible_matches} possible, need {val_limit}). "
              f"Will continue with all multi-image people.")
        val_people = multi_image_people

    val_set = set(val_people.keys())
    train_people = {p: people[p] for p in people if p not in val_set}

    return train_people, val_people


def main(
    csv_path="people.csv",
    root_dir="lfw_deepfunneled",
    train_limit=100000,
    val_limit=10000,
    val_size=100,
    batch_size=32
):
    people = read_people_csv(csv_path)
    train_people, val_people = split_people(people, val_size, val_limit=val_limit)

    # separate limits for train and val
    train_match, train_mismatch, train_match_total, train_mismatch_total = build_pairs(
        train_people, root_dir, train_limit)
    val_match, val_mismatch, val_match_total, val_mismatch_total = build_pairs(
        val_people, root_dir, val_limit)

    print(f"Train set: {len(train_people)} people")
    print(f"  Theoretical matches: {train_match_total}, mismatches: {train_mismatch_total}")
    print(f"  Sampled matches: {len(train_match)}, mismatches: {len(train_mismatch)}")
    print(f"Val set: {len(val_people)} people")
    print(f"  Theoretical matches: {val_match_total}, mismatches: {val_mismatch_total}")
    print(f"  Sampled matches: {len(val_match)}, mismatches: {len(val_mismatch)}")

    train_pairs = train_match + train_mismatch
    val_pairs = val_match + val_mismatch

    train_ds = TripletDataset(train_pairs)
    val_ds = TripletDataset(val_pairs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = main()
    check_image_dimensions("lfw-deepfunneled")
