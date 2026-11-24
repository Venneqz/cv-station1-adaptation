from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DATEN = Path("data/fgvc-aircraft-2013b/data")


def lade_liste(split):
    with open(DATEN / f"images_variant_{split}.txt", encoding="utf-8") as f:
        # nur am ersten Leerzeichen trennen, Namen wie "BAE 146-200" enthalten selbst eins
        return [zeile.strip().split(" ", 1) for zeile in f]


def lade_klassen():
    with open(DATEN / "variants.txt", encoding="utf-8") as f:
        return {zeile.strip(): nummer for nummer, zeile in enumerate(f)}


class FlugzeugDataset(Dataset):
    def __init__(self, split, mean, std):  # split: "train", "val" oder "test"
        self.paare = lade_liste(split)
        self.klassen = lade_klassen()
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.paare)

    def __getitem__(self, i):
        nummer, klasse = self.paare[i]
        bild = Image.open(DATEN / "images" / f"{nummer}.jpg").convert("RGB")
        breite, hoehe = bild.size
        bild = bild.crop((0, 0, breite, hoehe - 20))  # Fotografen-Signatur: untere 20 Pixel
        bild = bild.resize((224, 224))
        return self.tensor_transform(bild), self.klassen[klasse]
