import timm

ANZAHL_KLASSEN = 100  # 100 Flugzeug-Varianten


def lade_modell():
    # Vortrainierter Vision Transformer (ViT-Base, 16x16-Kacheln, 224 Pixel), wie VL 03, Folie 51.
    # num_classes=100 ersetzt den ImageNet-Kopf durch einen neuen Kopf mit 100 Ausgängen.
    return timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=ANZAHL_KLASSEN)


def nur_kopf_trainieren(modell):
    # Alles einfrieren ...
    for parameter in modell.parameters():
        parameter.requires_grad = False
    # ... und nur den neuen Kopf wieder freigeben.
    for parameter in modell.head.parameters():
        parameter.requires_grad = True


def zaehle_parameter(modell):
    gesamt = sum(p.numel() for p in modell.parameters())
    trainierbar = sum(p.numel() for p in modell.parameters() if p.requires_grad)
    return gesamt, trainierbar


if __name__ == "__main__":
    modell = lade_modell()

    nur_kopf_trainieren(modell)
    gesamt, trainierbar = zaehle_parameter(modell)
    print(f"Parameter gesamt:      {gesamt:,}")
    print(f"Davon trainierbar:     {trainierbar:,}")
    print(f"Anteil trainierbar:    {100 * trainierbar / gesamt:.2f} %")
