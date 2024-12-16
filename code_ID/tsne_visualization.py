import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from IdentificationDataset2 import NoisyID
from IdentificationDataset import CleanID #old


# Lade dein Dataset
S_clean = NoisyID(clean=True)
S_no_no = CleanID(train=True)#NoisyID(noiseReductionMethod=None, dereverberationMethod=None)
S_bl_bl = NoisyID(noiseReductionMethod="baseline", dereverberationMethod="baseline")
S_bl_nn = NoisyID(noiseReductionMethod="baseline", dereverberationMethod="neural net")
S_nn_bl = NoisyID(noiseReductionMethod="neural net", dereverberationMethod="baseline")
S_nn_nn = NoisyID(noiseReductionMethod="neural net", dereverberationMethod="neural net")

def visualize_tsne(dataset, ax, title):
    print(f'Num samples: {len(dataset)}')
    # Lade die Daten aus dem Dataset
    all_inputs = np.zeros((len(dataset), 280))
    all_labels = []

    for i in range(len(dataset)):
        inp, label = dataset[i]
        all_inputs[i,:] = inp.detach().numpy()
        all_labels.append(np.argmax(label))

    print(len(all_labels))



    # Anwenden von t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(all_inputs)
    print(1)
    # Visualisieren in der übergebenen Achse (ax)
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels, cmap='jet', alpha=0.8)
    print(2)
    ax.set_title(title)
    classes = np.unique(all_labels)  # Alle eindeutigen Klassen
    #handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(i / 6), markersize=10, label=f'{i}') for i in classes]
    #ax.legend(handles=handles, title='Speaker')
    return scatter  # Um den Colorbar später zu erzeugen

# Erstelle die zwei verschiedenen Datasets mit unterschiedlichen Variation Factors
names = ['clean', 'BL + BL', 'noisy direct', 'R-Net + BL-NR', 'N-Net + BL-DR', 'R-Net + N-Net']  # Zum Beispiel: Faktor 1.0 und 2.0
datasets = [S_clean, S_bl_bl, S_no_no, S_bl_nn, S_nn_bl, S_nn_nn]

# Erstelle die Subplots für die Visualisierungen (2 Subplots nebeneinander)
fig, axs = plt.subplots(2, 3, figsize=(21, 12))  # 1x2 Subplot

# Führe die t-SNE Visualisierung für jedes Dataset aus und zeige es in den Subplots an
for i, dataset in enumerate(datasets):
    scatter = visualize_tsne(dataset, axs[i//3][i%3], names[i])

# Verbessere das Layout
plt.tight_layout()
plt.show()