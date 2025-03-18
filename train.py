import torch
import torch.optim as optim
import torch.nn as nn
from model import Resnet18 
from utils import DataLoaderFunc

val_dataset = '../Dataset_Of_animal_Images/NewData/val/'
val_loader = DataLoaderFunc(val_dataset)
train_dataset = '../Dataset_Of_animal_Images/NewData/train/'
train_loader = DataLoaderFunc(train_dataset)

num_classes = 3
num_epochs = 5
learning_rate = 0.001

model = Resnet18(num_classes=num_classes, use_pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    # Training Phase
    model.train()  
    running_loss = 0.0
    for images, labels in train_loader:  
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss_avg = running_loss / len(train_loader)
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss_avg = val_loss / len(val_loader)
    val_accuracy = correct / total * 100

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_avg:.4f}, Validation Loss: {val_loss_avg:.4f}, Validation Accuracy: {val_accuracy:.2f}%")


torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': running_loss / len(train_loader),
    'num_classes': num_classes 
}, 'checkpoint.pth')