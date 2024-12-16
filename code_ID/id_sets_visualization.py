import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from IdentificationDataset2 import NoisyID


# Funktion zur Visualisierung der aneinandergereihten inputvecs
def visualize_inputvecs(dataset, ax, title):
    # Lade die Daten aus dem Dataset
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    
    all_inputs = []

    for inputvec, target in dataloader:
        all_inputs.append(inputvec.squeeze().detach().numpy())  # Direkt die Dimension [280] behalten

    all_inputs = np.concatenate(all_inputs, axis=0)  # Array der Form [num_samples, 280]

    # Zeige alle inputvecs als 2D-Bild (num_samples, 280)
    ax.imshow(all_inputs.T, aspect='auto', cmap='gray', vmax=1, vmin=0)
    ax.set_title(title)


def compute_mse(dataset_1, dataset_2, name):
    num_samples = len(dataset_1)
    error = 0.0
    for s in range(num_samples):
        input_1, _ = dataset_1[s]
        input_2, _ = dataset_2[s]
        error += torch.nn.functional.mse_loss(input_1.squeeze(), input_2.squeeze())
    print(f"{name}:\t{error/num_samples:.4e}")

# Erstelle die 5 verschiedenen Datasets mit unterschiedlichen Variation Factors
datasets = [NoisyID(clean=True, test=True),
            NoisyID(noiseReductionMethod=None, dereverberationMethod=None, test=True),
            NoisyID(noiseReductionMethod="baseline", dereverberationMethod="baseline", test=True),
            NoisyID(noiseReductionMethod="neural net", dereverberationMethod="baseline", test=True),
            NoisyID(noiseReductionMethod="neural net", dereverberationMethod="neural net", test=True)]
titles = ['clean', 'noisy', 'BL + BL', 'BL + N-Net', 'R-Net + N-Net']

# Erstelle die Subplots für die Visualisierungen
fig, axs = plt.subplots(1, 5, figsize=(20, 10))

# Führe die Visualisierung der inputvecs für jedes Dataset aus und zeige es in den Subplots an
for i, dataset in enumerate(datasets):
    visualize_inputvecs(dataset, axs[i], titles[i])
    compute_mse(datasets[0], datasets[i], titles[i])

# Verbessere das Layout
plt.tight_layout()
plt.show()
