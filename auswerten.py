import csv
import sys

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.daten import FlugzeugDataset
from src.modelle import baue_modell, zaehle_trainierbare
from train import BATCH, ERGEBNISSE, GRAKA, WORKER

ERGEBNISSE_DATEI = ERGEBNISSE / "ergebnisse.csv"


def main(verfahren):
    modell_datei = ERGEBNISSE / f"{verfahren}_bestes.pt"
    vorhersagen_datei = ERGEBNISSE / f"{verfahren}_vorhersagen.csv"

    modell = baue_modell(verfahren)
    modell.load_state_dict(torch.load(modell_datei))
    modell = modell.to(GRAKA)
    modell.eval()

    test_daten = FlugzeugDataset("test", modell.pretrained_cfg["mean"], modell.pretrained_cfg["std"])
    test_loader = DataLoader(test_daten, batch_size=BATCH, num_workers=WORKER)

    wahr = []
    vorhergesagt = []
    with torch.no_grad():
        for bilder, klassen in test_loader:
            ausgabe = modell(bilder.to(GRAKA))
            wahr += klassen.tolist()
            vorhergesagt += ausgabe.argmax(dim=1).cpu().tolist()

    richtig = sum(w == v for w, v in zip(wahr, vorhergesagt))
    genauigkeit = richtig / len(wahr)
    macro_f1 = f1_score(wahr, vorhergesagt, average="macro")
    print(f"Verfahren: {verfahren}")
    print(f"Test-Genauigkeit: {100 * genauigkeit:.1f} % ({richtig} von {len(wahr)})")
    print(f"Macro-F1:         {100 * macro_f1:.1f} %")

    # newline="" verhindert unter Windows Leerzeilen zwischen den CSV-Zeilen
    with open(vorhersagen_datei, "w", newline="") as f:
        schreiber = csv.writer(f)
        schreiber.writerow(["wahr", "vorhergesagt"])
        schreiber.writerows(zip(wahr, vorhergesagt))

    neu = not ERGEBNISSE_DATEI.exists()
    with open(ERGEBNISSE_DATEI, "a", newline="") as f:
        schreiber = csv.writer(f)
        if neu:
            schreiber.writerow(["verfahren", "trainierbare_parameter", "test_genauigkeit_prozent", "macro_f1_prozent"])
        schreiber.writerow([verfahren, zaehle_trainierbare(modell), f"{100 * genauigkeit:.2f}", f"{100 * macro_f1:.2f}"])


# Pflicht bei num_workers > 0 unter Windows: die Helfer importieren diese Datei neu
if __name__ == "__main__":
    main(sys.argv[1])
