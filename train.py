import time

import torch
from torch.utils.data import DataLoader

from src.daten import FlugzeugDataset
from src.modelle import lade_modell, nur_kopf_trainieren

GRAKA = "cuda"  # auch bei AMD (ROCm)
EPOCHEN = 3
LERNRATE = 1e-3
WORKER = 4


def main():
    train_daten = FlugzeugDataset("train")
    val_daten = FlugzeugDataset("val")
    train_loader = DataLoader(train_daten, batch_size=32, shuffle=True,
                              num_workers=WORKER, persistent_workers=True)
    val_loader = DataLoader(val_daten, batch_size=32,
                            num_workers=WORKER, persistent_workers=True)

    modell = lade_modell()
    nur_kopf_trainieren(modell)
    modell = modell.to(GRAKA)

    loss_funktion = torch.nn.CrossEntropyLoss()
    optimierer = torch.optim.AdamW(modell.head.parameters(), lr=LERNRATE)

    for epoche in range(EPOCHEN):
        start = time.time()

        modell.train()
        train_loss = 0
        train_richtig = 0
        for bilder, klassen in train_loader:
            bilder = bilder.to(GRAKA)
            klassen = klassen.to(GRAKA)

            optimierer.zero_grad()
            ausgabe = modell(bilder)
            loss = loss_funktion(ausgabe, klassen)
            loss.backward()
            optimierer.step()

            train_loss += loss.item() * len(klassen)
            train_richtig += (ausgabe.argmax(dim=1) == klassen).sum().item()

        modell.eval()
        val_loss = 0
        val_richtig = 0
        with torch.no_grad():
            for bilder, klassen in val_loader:
                bilder = bilder.to(GRAKA)
                klassen = klassen.to(GRAKA)

                ausgabe = modell(bilder)
                loss = loss_funktion(ausgabe, klassen)

                val_loss += loss.item() * len(klassen)
                val_richtig += (ausgabe.argmax(dim=1) == klassen).sum().item()

        dauer = time.time() - start
        print(
            f"Epoche {epoche + 1}: "
            f"Train Loss {train_loss / len(train_daten):.3f}, richtig {100 * train_richtig / len(train_daten):.1f} % | "
            f"Val Loss {val_loss / len(val_daten):.3f}, richtig {100 * val_richtig / len(val_daten):.1f} % | "
            f"{dauer:.0f} s"
        )


# Pflicht bei num_workers > 0 unter Windows: die Helfer importieren diese Datei neu
if __name__ == "__main__":
    main()
