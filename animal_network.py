import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# One Hot encoding for labels
def one_hot(labels, num_classes):
	return np.eye(num_classes)[labels]

class SimpleCNN(nn.Module):

	def __init__(self):

		super(SimpleCNN, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3) 		# inputs: grey-scale (1-3), filters (feature maps), kernal size (3x3)
		self.pool = nn.MaxPool2d(2, 2)			# pooling window size (2x2), pooling stride (by default it is 1)
		self.fc1 = nn.Linear(32 * 13 * 13, 128) # feature map size (13x13) times stack of maps (32), Output neurons of the layer
		self.fc2 = nn.Linear(128, 2)			# Input neurons, output neurons (labels)


	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))	# First forward pass to convolution layer
		x = x.view(-1, 32 * 13 * 13)			# resize everything for the fully connected layer
		x = F.relu(self.fc1(x))					# through relu
		x = self.fc2(x)							# Fully connected layer
		return x								# return

print("Reading in binary image data...")
x = pickle.load(open('./Animal-Recognition-Network/x.pkl', 'rb'))
y = pickle.load(open('./Animal-Recognition-Network/y.pkl', 'rb'))
print("Reading in images complete.")

num = len(x)
train_images = x[:num] 		                #Normalize each image color gradient
train_labels = one_hot(y[:num], 2)
test_images = x[num:]
test_labels = y[num:]

train_data = [train_images, train_labels]
trainloader = DataLoader(train_data, batch_size=64, shuffle=True)

print("Creating Model")
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 64

print("Starting Training...")
# Training
epochs = 1
for epoch in range(epochs):
	print("Training Epoch " + str(epoch+1) + "...")
	running_loss = 0
	for images, labels in trainloader:
		optimizer.zero_grad()
		output = model(images)
		loss = criterion(output, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
	print(f"Epoch {epoch+1} - Training loss: {running_loss/len(trainloader)}")
print("Training finished...")
