import csv

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.daten import FlugzeugDataset
from src.modelle import lade_modell, nur_kopf_trainieren, zaehle_parameter
from train import GRAKA, MODELL_DATEI, WORKER

VORHERSAGEN_DATEI = MODELL_DATEI.parent / "lp_vorhersagen.csv"
ERGEBNISSE_DATEI = MODELL_DATEI.parent / "ergebnisse.csv"


def main():
    modell = lade_modell()
    nur_kopf_trainieren(modell)  # nur fürs Zählen der trainierbaren Parameter
    modell.load_state_dict(torch.load(MODELL_DATEI))
    modell = modell.to(GRAKA)
    modell.eval()

    test_daten = FlugzeugDataset("test")
    test_loader = DataLoader(test_daten, batch_size=32, num_workers=WORKER)

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
    print(f"Test-Genauigkeit: {100 * genauigkeit:.1f} % ({richtig} von {len(wahr)})")
    print(f"Macro-F1:         {100 * macro_f1:.1f} %")

    # newline="" verhindert unter Windows Leerzeilen zwischen den CSV-Zeilen
    with open(VORHERSAGEN_DATEI, "w", newline="") as f:
        schreiber = csv.writer(f)
        schreiber.writerow(["wahr", "vorhergesagt"])
        schreiber.writerows(zip(wahr, vorhergesagt))

    _, trainierbar = zaehle_parameter(modell)
    neu = not ERGEBNISSE_DATEI.exists()
    with open(ERGEBNISSE_DATEI, "a", newline="") as f:
        schreiber = csv.writer(f)
        if neu:
            schreiber.writerow(["verfahren", "trainierbare_parameter", "test_genauigkeit_prozent", "macro_f1_prozent"])
        schreiber.writerow(["lp", trainierbar, f"{100 * genauigkeit:.2f}", f"{100 * macro_f1:.2f}"])


# Pflicht bei num_workers > 0 unter Windows: die Helfer importieren diese Datei neu
if __name__ == "__main__":
    main()
