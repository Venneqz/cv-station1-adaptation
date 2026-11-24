import timm
import torch
import torch.nn as nn

ANZAHL_KLASSEN = 100
VIT = "vit_base_patch16_224"
CNN = "resnet50"
CNN_21K = "resnetv2_50x1_bit.goog_in21k"


class LoraLinear(nn.Module):
    def __init__(self, original, r, alpha, dropout):
        super().__init__()
        self.original = original
        d_in = original.in_features
        d_out = original.out_features
        self.A = nn.Parameter(torch.randn(d_in, r) / d_in ** 0.5)
        self.B = nn.Parameter(torch.zeros(r, d_out))  # Nullen: am Anfang ändert LoRA nichts
        self.dropout = nn.Dropout(dropout)
        self.skala = alpha / r

    def forward(self, x):
        return self.original(x) + self.dropout(x) @ self.A @ self.B * self.skala


def lade_modell(name):
    return timm.create_model(name, pretrained=True, num_classes=ANZAHL_KLASSEN)


def nur_kopf_trainieren(modell):
    for parameter in modell.parameters():
        parameter.requires_grad = False
    # get_classifier(): der Kopf heißt beim ViT "head", beim ResNet "fc"
    for parameter in modell.get_classifier().parameters():
        parameter.requires_grad = True


def lora_einbauen(modell, r, alpha=16, dropout=0.05):
    nur_kopf_trainieren(modell)  # vor dem Einbau, sonst würden A und B mit eingefroren
    for block in modell.blocks:
        block.attn.qkv = LoraLinear(block.attn.qkv, r, alpha, dropout)
        block.attn.proj = LoraLinear(block.attn.proj, r, alpha, dropout)


def baue_modell(verfahren):
    if verfahren == "lp":
        modell = lade_modell(VIT)
        nur_kopf_trainieren(modell)
        return modell
    if verfahren.startswith("lora_r"):
        r = int(verfahren.removeprefix("lora_r"))
        modell = lade_modell(VIT)
        lora_einbauen(modell, r)
        return modell
    if verfahren == "cnn_lp":
        modell = lade_modell(CNN)
        nur_kopf_trainieren(modell)
        return modell
    if verfahren == "cnn_ft":
        return lade_modell(CNN)
    if verfahren == "cnn21k_lp":
        modell = lade_modell(CNN_21K)
        nur_kopf_trainieren(modell)
        return modell
    if verfahren == "cnn21k_ft":
        return lade_modell(CNN_21K)
    raise ValueError(f"unbekanntes Verfahren: {verfahren}")


def zaehle_trainierbare(modell):
    return sum(p.numel() for p in modell.parameters() if p.requires_grad)
