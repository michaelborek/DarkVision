import torch
import torch.optim as optim
import torch.nn as nn
from model import Resnet18 
from utils import DataLoaderFunc

dataset = '../Dataset_Of_animal_Images/NewData/train/'
train_loader = DataLoaderFunc(dataset)

num_classes = 3
num_epochs = 5
learning_rate = 0.001

model = Resnet18(num_classes=num_classes, use_pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0
    for images, labels in train_loader:  
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        labels = labels.squeeze(1).long()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

torch.save(model.state_dict(), 'resnet18_model.pth')