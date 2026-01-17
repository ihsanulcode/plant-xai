import os
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# ----------------------- Dataloader -------------------------------
def create_train_val_dataloaders(data_dir, batch_size, num_workers=4):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'valid': transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }

    # Create datasets using specific transformations
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
        'valid': datasets.ImageFolder(os.path.join(data_dir, 'valid'), data_transforms['valid']),
    }

    # Create dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'valid': DataLoader(image_datasets['valid'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    return dataloaders


# ----------------------- Active GPU Mode -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ----------------------- Custom CNN ------------------------------------
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()

        # -------- Block 1 --------
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # -------- Block 2 --------
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # -------- Block 3 --------
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # -------- Block 4 --------
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 15 * 15, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)
        return x


# Model Loader function
def load_custom_cnn(num_classes):
    return CustomCNN(num_classes)

# --------------------- Model Training Section ---------------------
class ModelTrainer:
    def __init__(self, model, dataloaders, criterion, optimizer, num_epochs, save_dir):
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.save_dir = save_dir

        # Store loss & accuracy
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        # Store precision, recall, f1
        self.train_precisions = []
        self.train_recalls = []
        self.train_f1s = []

        self.val_precisions = []
        self.val_recalls = []
        self.val_f1s = []

    def train(self):
        os.makedirs(self.save_dir, exist_ok=True)

        train_loader = self.dataloaders['train']
        val_loader = self.dataloaders['valid']

        for epoch in range(self.num_epochs):
            # --------------------- TRAINING ---------------------
            self.model.train()
            running_train_loss = 0.0
            correct_train = 0
            total_train = 0

            train_all_preds = []
            train_all_labels = []

            for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} - Training'):
                inputs, labels = inputs.to(device), labels.to(device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_train_loss += loss.item() * inputs.size(0)

                # Predictions
                _, predicted = torch.max(outputs, 1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

                train_all_preds.extend(predicted.cpu().tolist())
                train_all_labels.extend(labels.cpu().tolist())

            train_loss = running_train_loss / len(train_loader.dataset)
            train_acc = correct_train / total_train

            # Train precision recall f1
            train_precision = precision_score(train_all_labels, train_all_preds,
                                              average='macro', zero_division=0)
            train_recall = recall_score(train_all_labels, train_all_preds,
                                        average='macro', zero_division=0)
            train_f1 = f1_score(train_all_labels, train_all_preds,
                                average='macro', zero_division=0)

            # --------------------- VALIDATION ---------------------
            self.model.eval()
            running_val_loss = 0.0
            correct_val = 0
            total_val = 0

            val_all_preds = []
            val_all_labels = []

            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} - Validation'):
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    running_val_loss += loss.item() * inputs.size(0)

                    _, predicted = torch.max(outputs, 1)
                    correct_val += (predicted == labels).sum().item()
                    total_val += labels.size(0)

                    val_all_preds.extend(predicted.cpu().tolist())
                    val_all_labels.extend(labels.cpu().tolist())

            val_loss = running_val_loss / len(val_loader.dataset)
            val_acc = correct_val / total_val

            # Val precision recall f1
            val_precision = precision_score(val_all_labels, val_all_preds,
                                            average='macro', zero_division=0)
            val_recall = recall_score(val_all_labels, val_all_preds,
                                      average='macro', zero_division=0)
            val_f1 = f1_score(val_all_labels, val_all_preds,
                              average='macro', zero_division=0)

            # Save all metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            self.train_precisions.append(train_precision)
            self.train_recalls.append(train_recall)
            self.train_f1s.append(train_f1)

            self.val_precisions.append(val_precision)
            self.val_recalls.append(val_recall)
            self.val_f1s.append(val_f1)

            # Print summary
            print(
                f'Epoch [{epoch+1}/{self.num_epochs}] - '
                f'Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%, '
                f'Prec: {train_precision:.4f}, Rec: {train_recall:.4f}, F1: {train_f1:.4f} | '
                f'Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, '
                f'Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, F1: {val_f1:.4f}'
            )

            # Save model for each epoch
            torch.save(self.model.state_dict(),
                       os.path.join(self.save_dir, f'model_epoch_{epoch+1}.pth'))

        # Save curves & Excel
        self.plot_learning_curves(os.path.join(self.save_dir, "training_curves.png"))
        self.save_metrics_to_excel()

        print("Training completed successfully.")

    # ---------------------- PLOT CURVES ----------------------
    def plot_learning_curves(self, fig_path):
        plt.figure(figsize=(12, 5))

        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()

        # Accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label="Train Accuracy")
        plt.plot(self.val_accuracies, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.legend()

        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()

    # ---------------------- SAVE METRICS ----------------------
    def save_metrics_to_excel(self):
        df = pd.DataFrame({
            "Epoch": list(range(1, len(self.train_losses) + 1)),

            "Train Loss": self.train_losses,
            "Train Accuracy": self.train_accuracies,
            "Train Precision": self.train_precisions,
            "Train Recall": self.train_recalls,
            "Train F1": self.train_f1s,

            "Val Loss": self.val_losses,
            "Val Accuracy": self.val_accuracies,
            "Val Precision": self.val_precisions,
            "Val Recall": self.val_recalls,
            "Val F1": self.val_f1s,
        })

        save_path = os.path.join(self.save_dir, "customModel_metrics.xlsx")
        df.to_excel(save_path, index=False)
        print(f"Metrics saved to {save_path}")


# -------------------- Main Execution --------------------
if __name__ == "__main__":
    # Count the number of classes
    train_dir = "dataset/train"
    classes = [folder for folder in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, folder))]
    num_classes = len(classes)

    # Training params setup
    data_dir = 'dataset'
    save_dir = "Model_Outputs/customModel_Softmax"
    batch_size = 32
    num_epochs = 20
    dataloaders = create_train_val_dataloaders(data_dir, batch_size)
    model = load_custom_cnn(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ModelTrainer
    trainer = ModelTrainer(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        save_dir=save_dir
    )

    # Train the model
    trainer.train()