import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


ATTR_NAMES: list[str] = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
    "Wearing_Necktie", "Young",
]


class CelebADataset(Dataset):
    """CelebA aligned face dataset with 40 binary attributes.

    Attributes are stored as {-1, 1} in the CSV; we convert to {0.0, 1.0}.
    Images are normalised to [-1, 1] (Tanh range).
    """

    def __init__(self, data_dir: str, split: str = "train", img_size: int = 64):
        assert split in ("train", "val", "test")

        self.img_dir = os.path.join(
            data_dir, "img_align_celeba", "img_align_celeba"
        )

        attrs_df = pd.read_csv(os.path.join(data_dir, "list_attr_celeba.csv"))
        part_df = pd.read_csv(os.path.join(data_dir, "list_eval_partition.csv"))

        # Merge on image_id — never assume both CSVs share the same row order
        merged = attrs_df.merge(part_df, on="image_id")

        split_id = {"train": 0, "val": 1, "test": 2}[split]
        subset = merged[merged["partition"] == split_id]

        self.filenames = subset["image_id"].values
        raw = subset[ATTR_NAMES].values.astype("float32")
        # {-1, 1} → {0, 1}
        self.attrs = torch.from_numpy((raw + 1.0) / 2.0)

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        path = os.path.join(self.img_dir, self.filenames[idx])
        img = Image.open(path).convert("RGB")
        return self.transform(img), self.attrs[idx]
