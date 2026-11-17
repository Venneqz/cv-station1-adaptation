import timm

ANZAHL_KLASSEN = 100


def lade_modell():
    return timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=ANZAHL_KLASSEN)


def nur_kopf_trainieren(modell):
    for parameter in modell.parameters():
        parameter.requires_grad = False
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
