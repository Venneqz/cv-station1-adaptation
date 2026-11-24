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

**Repo - Aufbau:**
| Ordner | Inhalt |
|---|---|
| `src/` | eigener Code (Daten laden, Modelle, Training, Auswertung) |
| `ergebnisse/` | Kennzahlen als CSV, Diagramme, Konfusionsmatrizen |
| `ausarbeitung/` | Bericht im CVPR-2026-LaTeX-Template (Vorgaben, s.u.) |

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