import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.daten import FlugzeugDataset
from src.modelle import baue_modell, zaehle_parameter

GRAKA = "cuda"  # auch bei AMD (ROCm)
EPOCHEN = 20
LERNRATEN = {"lp": 1e-3, "cnn_lp": 1e-2, "cnn_ft": 3e-4, "lora_r4": 3e-4, "lora_r8": 3e-4, "lora_r16": 3e-4,
             "cnn21k_lp": 3e-4, "cnn21k_ft": 1e-4}
WORKER = 4
SEED = 67
ERGEBNISSE = Path(__file__).resolve().parent / "ergebnisse"


def main(verfahren):
    modell_datei = ERGEBNISSE / f"{verfahren}_bestes.pt"
    lernrate = LERNRATEN[verfahren]
    torch.manual_seed(SEED)  # vor baue_modell, damit Kopf und LoRA-Matrizen gleich starten
    modell = baue_modell(verfahren)
    _, trainierbar = zaehle_parameter(modell)
    mean = modell.pretrained_cfg["mean"]
    std = modell.pretrained_cfg["std"]
    print(f"Verfahren: {verfahren}, trainierbare Parameter: {trainierbar:,}, Lernrate: {lernrate}")
    print(f"Normierung: mean {mean}, std {std}")
    modell = modell.to(GRAKA)

    train_daten = FlugzeugDataset("train", mean, std)
    val_daten = FlugzeugDataset("val", mean, std)
    train_loader = DataLoader(train_daten, batch_size=32, shuffle=True,
                              num_workers=WORKER, persistent_workers=True)
    val_loader = DataLoader(val_daten, batch_size=32,
                            num_workers=WORKER, persistent_workers=True)

    loss_funktion = torch.nn.CrossEntropyLoss()
    trainierbare = [p for p in modell.parameters() if p.requires_grad]
    optimierer = torch.optim.AdamW(trainierbare, lr=lernrate)

    bester_val_loss = float("inf")
    beste_epoche = 0

    lauf_start = time.time()
    for epoche in range(EPOCHEN):
        start = time.time()

        # cnn_lp: im train()-Modus würde BatchNorm seine Laufmittelwerte an unsere Bilder anpassen
        if verfahren == "cnn_lp":
            modell.eval()
        else:
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
        val_loss_mittel = val_loss / len(val_daten)

        gespeichert = ""
        if val_loss_mittel < bester_val_loss:
            bester_val_loss = val_loss_mittel
            beste_epoche = epoche + 1
            torch.save(modell.state_dict(), modell_datei)
            gespeichert = " *"

        print(
            f"Epoche {epoche + 1}: "
            f"Train Loss {train_loss / len(train_daten):.3f}, richtig {100 * train_richtig / len(train_daten):.1f} % | "
            f"Val Loss {val_loss_mittel:.3f}, richtig {100 * val_richtig / len(val_daten):.1f} % | "
            f"{dauer:.0f} s{gespeichert}"
        )

    print(f"bestes Modell: Epoche {beste_epoche} (Val Loss {bester_val_loss:.3f})")
    print(f"Gesamtdauer: {(time.time() - lauf_start) / 60:.1f} min")
    print(f"Grafikspeicher (Höchstwert): {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


# Pflicht bei num_workers > 0 unter Windows: die Helfer importieren diese Datei neu
if __name__ == "__main__":
    main(sys.argv[1])
