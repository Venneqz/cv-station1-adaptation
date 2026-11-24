import matplotlib.pyplot as plt
import pandas as pd

from src.daten import lade_klassen
from train import ERGEBNISSE

VERFAHREN_VERWECHSLUNGEN = "lora_r16"
ANZAHL_FEHLER = 5

# Name im Diagramm und Lage der Beschriftung: (Versatz x, Versatz y in Punkten, Ausrichtung).
NAMEN = {
    "lp": ("ViT-LP", (5, -3, "left")),
    "lora_r4": ("LoRA r=4", (-5, -3, "right")),
    "lora_r8": ("LoRA r=8", (0, -11, "center")),
    "lora_r16": ("LoRA r=16", (0, 6, "center")),
    "cnn21k_lp": ("CNN-LP", (5, -3, "left")),
    "cnn21k_ft": ("CNN-FT", (-5, -3, "right")),
}


def genauigkeit_gegen_parameter():
    tabelle = pd.read_csv(ERGEBNISSE / "ergebnisse.csv")
    tabelle = tabelle[tabelle["verfahren"].isin(NAMEN)]
    ist_cnn = tabelle["verfahren"].str.startswith("cnn")

    plt.rcParams["font.size"] = 8
    plt.figure(figsize=(3.4, 2.4))
    for name, gruppe, farbe in [("ViT-B/16", tabelle[~ist_cnn], "tab:blue"),
                                ("ResNetV2-50", tabelle[ist_cnn], "tab:orange")]:
        plt.scatter(gruppe["trainierbare_parameter"], gruppe["test_genauigkeit_prozent"],
                    color=farbe, label=name, s=18)
    for zeile in tabelle.itertuples():
        name, (dx, dy, ausrichtung) = NAMEN[zeile.verfahren]
        plt.annotate(name, (zeile.trainierbare_parameter, zeile.test_genauigkeit_prozent),
                     xytext=(dx, dy), textcoords="offset points", ha=ausrichtung)

    plt.xscale("log")
    plt.xlabel("trainierbare Parameter")
    plt.ylabel("Test-Genauigkeit in %")
    plt.margins(x=0.1, y=0.15)
    plt.legend(loc="lower right", labelspacing=0.3, borderpad=0.3, handletextpad=0.3)
    plt.grid(alpha=0.3)
    plt.savefig(ERGEBNISSE / "genauigkeit_parameter.pdf", bbox_inches="tight")
    plt.close()


def haeufigste_verwechslungen():
    vorhersagen = pd.read_csv(ERGEBNISSE / f"{VERFAHREN_VERWECHSLUNGEN}_vorhersagen.csv")
    namen = list(lade_klassen())  # Position in der Liste = Klassennummer
    fehler = vorhersagen[vorhersagen["wahr"] != vorhersagen["vorhergesagt"]]

    print(f"Häufigste Verwechslungen ({VERFAHREN_VERWECHSLUNGEN}):")
    for (wahr, vorhergesagt), anzahl in fehler.value_counts().head(ANZAHL_FEHLER).items():
        print(f"{namen[wahr]} wird mit {namen[vorhergesagt]} verwechselt, {anzahl}-mal")


if __name__ == "__main__":
    genauigkeit_gegen_parameter()
    haeufigste_verwechslungen()
