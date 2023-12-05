import os
import pandas as pd
import torch
from skimage import io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import timeit

class CatsAndDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations) # 24996 images
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)
        
        return (image, y_label)
    
class ThreeCNN(nn.Module):
    def __init__(self):
        super(ThreeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3) # convolutional layer #2
        self.conv3 = nn.Conv2d(64, 128, 3)  # Added another convolutional layer # 3
        self.fc1 = nn.Linear(128 * 10 * 10, 128)  # Adjusted input size based on the added layer
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))  # Added another convolutional layer
        x = x.view(-1, 128 * 10 * 10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleCNN(nn.Module):
	def __init__(self):

		super(SimpleCNN, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3) 		# Inputs: grey-scale (1-3), filters (feature maps), kernal size (3x3)
		self.pool = nn.MaxPool2d(2, 2)			# Pooling window size (2x2), pooling stride (by default it is 1)
		self.fc1 = nn.Linear(32 * 49 * 49, 128) # Feature map size (13x13) times stack of maps (32), Output neurons of the layer
		self.fc2 = nn.Linear(128, 2)			# Input neurons, output neurons (labels)


	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))	# First forward pass to convolution layer
		x = x.view(-1, 32 * 49 * 49)			# Resize everything for the fully connected layer
		x = F.relu(self.fc1(x))					# Through relu
		x = self.fc2(x)							# Fully connected layer
		return x								# Return

start = timeit.default_timer()
print("COMMENCER\n")

batch_size = 256

dataset = CatsAndDogsDataset(csv_file='data.csv', root_dir='./PetImages/All_grey/', transform = transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [19996, 5000])
trainloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)           
testloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

print("Creating Model\n")
model = ThreeCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
print("Starting Training...\n")
# Training
epochs = 20
training_losses = []  # List to store training losses

for epoch in range(epochs):
    print("Training Epoch " + str(epoch + 1) + "...")
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(trainloader)
    training_losses.append(epoch_loss)
    print(f"Epoch {epoch + 1} - Training loss: {epoch_loss}\n")

print("Training finished...\n")

print("Starting testing...\n")
correct_count, all_count = 0, 0
for images, labels in testloader:
      for i in range(len(labels)):
            img = images[i].view(1, 1, 100, 100) # 1, # of channels, pixel dimension, pixel dimension
            with torch.no_grad():
                  logps = model(img)
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if true_label == pred_label:
                   correct_count += 1
            all_count += 1
print(f"Number Of Images Tested = {all_count}")
print(f"Model Accuracy = {(correct_count/all_count):.2f}")

plt.plot(training_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.legend()
plt.show()

stop = timeit.default_timer()
print('Time: ', stop - start)  
print("FIN")