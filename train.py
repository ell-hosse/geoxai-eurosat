import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset_loader import get_eurosat_dataloaders
from models.simple_cnn import SimpleCNN

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, class_names = get_eurosat_dataloaders()
    num_classes = len(class_names)

    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss:.4f}, Accuracy: {100.*correct/total:.2f}%")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")

if __name__ == '__main__':
    train()
