import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from model import Resnet18  
from utils import DataLoaderFunc
import pandas as pd

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
checkpoint = torch.load("checkpoint.pth", map_location=device)
model = Resnet18(num_classes=checkpoint['num_classes'], use_pretrained=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Load the test dataset
test_dataset = '../Dataset_Of_animal_Images/NewData/test/'
test_loader = DataLoaderFunc(test_dataset, 5, False)

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds, probas = model.prediction(outputs)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = accuracy_score(all_labels, all_preds)
conf_mat = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds)

print("Test Accuracy: {:.2f}%".format(accuracy * 100))
print("Confusion Matrix:")
print(conf_mat)
print("Classification Report:")
print(report)

df_preds = pd.DataFrame({
    "True": all_labels,
    "Pred": all_preds
})
df_preds.to_csv("predictions.csv", index=False)

with open("metrics_summary.txt", "w") as f:
    f.write("Test Accuracy: {:.2f}%\n\n".format(accuracy * 100))
    f.write("Confusion Matrix:\n")
    # Write the confusion matrix row by row
    for row in conf_mat:
        row_str = ", ".join(str(x) for x in row)
        f.write(row_str + "\n")
    f.write("\nClassification Report:\n")
    # Write the classification report line by line
    for line in report.splitlines():
        f.write(line + "\n")