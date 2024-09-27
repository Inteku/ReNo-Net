import torch
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 16),
            nn.Sigmoid(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.Softmax()
        )

    def forward(self, x):
        return self.layers(x)
    
def visualize_classification(net, ax):
    # Erzeuge ein Gitter von Punkten im Bereich [-1, 1] für beide Dimensionen
    x = np.linspace(-50, 70, 100)
    y = np.linspace(-50, 50, 100)
    x_grid, y_grid = np.meshgrid(x, y)
    points = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    # Konvertiere das Gitter in einen PyTorch-Tensor
    input_tensor = torch.tensor(points, dtype=torch.float32)

    # Führe die Vorwärtsdurchläufe durch, um die Klassifikation für jedes Gitterpunkt zu erhalten
    with torch.no_grad():
        output_tensor = net(input_tensor)
        predictions = torch.argmax(output_tensor, dim=1)

    # Visualisiere die Klassifikation mit verschiedenen Farben
    ax.scatter(x_grid, y_grid, c=predictions.numpy(), cmap='viridis', alpha=0.5)

# Transformationen für die Bilder
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])

# Laden des MNIST-Datensets
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Extrahieren von 200 Samples
num_samples = 2000
selected_indices = torch.randperm(len(mnist_dataset))[:num_samples]

# Erstellen des Tensors und der Label-Liste
data_tensor = torch.stack([mnist_dataset[i][0] for i in selected_indices])
labels = [mnist_dataset[i][1] for i in selected_indices]

# Reshape des Tensors auf die gewünschte Größe (200x784)
data_tensor = data_tensor.view(num_samples, -1)

# Überprüfen der Größen
print("Shape des Daten-Tensors:", data_tensor.shape)
print("Länge der Labels-Liste:", len(labels))


tsne_data = TSNE(n_components=2, random_state=818, perplexity=30)
tsne_data = tsne_data.fit_transform(data_tensor)
labels=torch.Tensor(labels)


classifier = SimpleNet()
epochs = 800
loss_progress = []
vloss_progress = []
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=8e-5)
train_set = torch.Tensor(tsne_data[0:1700])
train_labels = labels[0:1700].long()
val_set = torch.Tensor(tsne_data[1700:2000])
val_labels = labels[1700:2000].long()

loss_progress.append(criterion(classifier(train_set),train_labels).item())
vloss_progress.append(criterion(classifier(val_set),val_labels).item())
for e in range(epochs):
    print(e+1,end='\r')
    prediction = classifier(train_set)
    loss = criterion(prediction, train_labels)
    loss.backward()
    loss_progress.append(loss.item())
    optimizer.step()
    with torch.no_grad():
        vprediction = classifier(val_set)
        vloss = criterion(vprediction, val_labels)
        vloss_progress.append(vloss.item())

plt.figure()
plt.plot(loss_progress)
plt.plot(vloss_progress)
plt.ylim(bottom=0)
plt.show()

fig, ax = plt.subplots()
visualize_classification(classifier, ax)
#plt.show()

#plt.figure()
for id in range(10):
    plt.scatter(tsne_data[labels == id, 0], tsne_data[labels == id, 1], label=str(id), marker='o', edgecolors='k')
    #plt.scatter(tsne_data[labels == id], 0*tsne_data[labels == id], label=str(id), marker='o', edgecolors='k')
plt.legend()
plt.title('Unsupervised distinction of handwritten digits using t-SNE')
plt.show()