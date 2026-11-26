# Station 1 – Adaptation (Deep Learning for Computer Vision, WS 2026/27)
**Abgabe-Repo** MIF-2.W - Deep Learning for Computer Vision WS2026/2027 Prof. Dr.-Ing. Jan Rexilius\
Mattis Wellenbüscher, Abgabe 26. Nov 2026, 08:00\
Training eines bestehenden Modells & Vergleich mit „klassischer Architektur“ 

Forschungsfrage: Wie effizient lässt sich ein vortrainierter ViT an eine neue Aufgabe anpassen?

**Dazu zählt:**
- Auswahl eines geeigneten Datensatzes
- ViT: linear probing, Training einer selbst gewählten ViT-Architektur auf eigenen Daten (nur Kopf nachtrainieren)
- ViT: LoRA Low-Rank Adaptation mit unterschiedlichen Parametern
- CNN: Training einer CNN-Architektur
- Vergleich der Modelle
- Diskutieren Sie Ihre Ergebnisse

**Repo-Aufbau:**
| Datei / Ordner | Inhalt |
|---|---|
| `src/daten.py` | Datensatz laden, Copyright-Streifen entfernen, auf 224 × 224 stauchen, normieren |
| `src/modelle.py` | ViT-B/16 und ResNets laden, einfrieren, eigene LoRA-Schicht |
| `train.py` | Training eines Verfahrens, speichert das beste Modell nach Validierungsverlust |
| `auswerten.py` | Test-Genauigkeit und Macro-F1, schreibt `ergebnisse/ergebnisse.csv` und die Vorhersagen |
| `ergebnisse/` | Kennzahlen, Vorhersagen je Lauf, Abbildungen, `logs/` mit den Trainingslogs |

**Starten** (aus dem Repo-Ordner, Daten unter `data/fgvc-aircraft-2013b`):
    python train.py <verfahren>
    python auswerten.py <verfahren>

**Verfahren und Namen in der Ausarbeitung:**
| Code | Ausarbeitung |
|---|---|
| `lp` | ViT-LP |
| `lora_r4`, `lora_r8`, `lora_r16` | LoRA r = 4, 8, 16 |
| `cnn21k_lp`, `cnn21k_ft` | CNN-LP, CNN-FT (ResNetV2-50, ImageNet-21k) |
| `cnn_lp`, `cnn_ft` | ResNet-50 (ImageNet-1k), nur im Anhang |

**Abgabeformat:**
1. PDF-Dokument max. 6 Seiten
   - Template für Format: https://github.com/cvpr-org/author-kit/releases/tag/CVPR2026-v1(latex)
2. Quellcode / Link auf Git-Repository
   - Trainingsbilder nicht in ILIAS hochladen

**AUSARBEITUNG FÜR STATION1/STATION2:**\
Die Ausarbeitungen für Station 1 und Station 2 müssen wie folgt aufgebaut sein:
- Abstract (ca. 100-150 Wörter)
- Einleitung (ca. ½ - ¾ Seite)
  - konkrete Beiträge der Arbeit
- Stand der Forschung (ca. ¾ - 1 Seite)
  - min. 5 wissenschaftliche Quellen
  - nur für die jeweilige Aufgabe relevante Arbeiten
- Methode (ca. 1 - 1½ Seiten)
  - Datensatz, Modelle, Parameter, …
- Ergebnisse (ca. 1 - 1½ Seiten)
- Diskussion (ca. 1 - 1½ Seiten)
- Referenzen (zählt nicht zu den 6 Seiten)
- Anhang soweit notwendig (zählt nicht zu den 6 Seiten)

**PROJEKT-CHECKLISTE**
- Terminmanagement
  - Roadmap, Arbeitspakete, Meilensteine
- Qualitätsmanagement
  - Anforderungen und Merkmale
- Risikomanagement
  - Beschreibung
  - Bewertung (Wahrscheinlichkeit & Auswirkung)
  - Aktion