import torch
from torch import nn

class NeumoniaDetector(nn.Module):
    def __init__(self, input_channels: int, output_shape: int):
        super().__init__()

        self.conv_block = nn.Sequential(
            # Primera capa
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Segunda capa
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Reduce dimensiones de 112x112 a 56x56
        )

        # Bloque de clasificación
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32 * 56 * 56, out_features=128),
            nn.ReLU(),

            # Apaga aleatoriamente el 50% de las neuronas durante el entrenamiento para forzar al modelo a generalizar y no memorizar el dataset.
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        # Flujo: Imagen -> Convoluciones -> Clasificador -> Logits (Con ayuda de IA)
        return self.classifier(self.conv_block(x))