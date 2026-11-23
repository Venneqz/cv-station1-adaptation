import time

import torch
from torch.utils.data import DataLoader

from src.daten import FlugzeugDataset
from src.modelle import lade_modell

GRAKA = "cuda"
DURCHLAEUFE = 100

if __name__ == "__main__":
    daten = FlugzeugDataset("train", (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    loader = DataLoader(daten, batch_size=32, shuffle=True)

    start = time.time()
    for bilder, klassen in loader:
        pass
    dauer = time.time() - start
    print(f"Nur Daten laden: {dauer:.1f} s für {len(daten)} Bilder, {len(daten) / dauer:.0f} Bilder/s")

    modell = lade_modell("vit_base_patch16_224").to(GRAKA)
    modell.eval()
    bilder = torch.randn(32, 3, 224, 224).to(GRAKA)

    with torch.no_grad():
        modell(bilder)  # Aufwärmen: der erste Aufruf bereitet die GPU-Kernel vor
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(DURCHLAEUFE):
            modell(bilder)
        torch.cuda.synchronize()
        dauer = time.time() - start

    anzahl = DURCHLAEUFE * len(bilder)
    print(f"Nur Netz rechnen: {dauer:.1f} s für {anzahl} Bilder, {anzahl / dauer:.0f} Bilder/s")
