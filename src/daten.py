import PIL.Image
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset
from torchvision import transforms

BASE = Path(__file__).resolve().parent.parent
BILDER = BASE / "data" / "fgvc-aircraft-2013b" / "data" / "images"
ERGEBNISSE =  BASE / "ergebnisse" / "zuschnitt-test"

#Fotografen Signatur weg bekommen (untere 20 Pixel)
def crop_signature(image):
    w, h = image.size
    return image.crop((0, 0, w, h - 20))

#Variante 1 stauchen auf 224x224
def variant_1(image):
    return image.resize((224, 224))

#Liste aus (Bildnummer, Klassenname) lesen
def lade_liste(datei):
    paare = []
    with open(datei, encoding="utf-8") as f:
        for zeile in f:
            zeile = zeile.strip()
            if not zeile:
                continue
            nummer, klasse = zeile.split(" ", 1)   # nur am ersten Leerzeichen"BAE 146-200" haben selbst eins
            paare.append((nummer, klasse))
    return paare

#Alle 100 Klassennamen lesen und jedem eine feste Nummer 0-99 geben
def lade_klassen(datei):
    with open(datei, encoding="utf-8") as f:
        namen = [zeile.strip() for zeile in f if zeile.strip()]
    return {name: nummer for nummer, name in enumerate(namen)}

#Umwandlung Bild -> Tensor (3x224x224, Werte 0-1) und Normalisierung mit ImageNet-Werten
tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#Dataset: liefert Bild i als Tensor zusammen mit der Klassennummer
class FlugzeugDataset(Dataset):
    def __init__(self, split):   # split: "train", "val" oder "test"
        self.paare = lade_liste(BILDER.parent / f"images_variant_{split}.txt")
        self.klassen = lade_klassen(BILDER.parent / "variants.txt")

    def __len__(self):
        return len(self.paare)

    def __getitem__(self, i):
        nummer, klasse = self.paare[i]
        image = PIL.Image.open(BILDER / f"{nummer}.jpg").convert("RGB")
        image = variant_1(crop_signature(image))
        return tensor_transform(image), self.klassen[klasse]

#Läuft nur bei direktem Start nicht bei import
if __name__ == "__main__":
    #Test 1: Bild laden Signatur entfernen auf 224x224 stauchen und speichern
    name = "0063835"
    image = crop_signature(PIL.Image.open(BILDER / f"{name}.jpg"))
    variant_1(image).save(ERGEBNISSE / f"{name}_variant_1.jpg")

    #Test 2: Trainingsliste lesen
    train = lade_liste(BILDER.parent / "images_variant_train.txt")
    print("Länge:", len(train), "(erwartet 3334)")
    print("Erste drei:", train[:3])

    #Test 3: Klassen-Nummern und Bilder pro Klasse
    klassen = lade_klassen(BILDER.parent / "variants.txt")
    print("Klassen:", len(klassen), "(erwartet 100)")
    print("Erste drei:", list(klassen.items())[:3])
    anzahl = Counter(klasse for _, klasse in train)
    print("Bilder pro Klasse: min", min(anzahl.values()), "max", max(anzahl.values()))
    print("Klassen im Training, die nicht in variants.txt stehen:", set(anzahl) - set(klassen))

    #Test 4: Dataset
    for split in ["train", "val", "test"]:
        print(split, "Länge:", len(FlugzeugDataset(split)))
    tensor, label = FlugzeugDataset("train")[0]
    print("Form:", tuple(tensor.shape), "Typ:", tensor.dtype, "Klasse:", label)
    print("Wertebereich:", float(tensor.min()), "bis", float(tensor.max()))
