import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18


def get_resnet18(num_classes=10):
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, num_classes)
    return model


'''def train_target_model(batch_size=128, num_epochs=50, device='cuda'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)

    N = len(dataset)
    n_train = int(0.8 * N)
    n_rest = N - n_train

    train_set, rest_set = random_split(dataset, [n_train, n_rest])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(rest_set, batch_size=batch_size, shuffle=False)

    model = get_resnet18().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()


    print("🚀 Training target ResNet18 …")
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            _, pred = logits.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Loss={running_loss/len(train_loader):.4f} "
              f"Acc={correct/total:.4f}")



    return model, train_set, rest_set'''

def train_target_model(batch_size=128, num_epochs=50, device='cuda'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)

    N = len(dataset)
    n_train = int(0.8 * N)
    n_rest = N - n_train

    train_set, rest_set = random_split(dataset, [n_train, n_rest])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(rest_set, batch_size=batch_size, shuffle=False)

    model = get_resnet18().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()


    print("🚀 Training target ResNet18 …")
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            _, pred = logits.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Loss={running_loss/len(train_loader):.4f} "
              f"Acc={correct/total:.4f}")

    shadow_model = get_resnet18().to(device)
    opt = optim.Adam(shadow_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    print("🚀 Training shadow ResNet18 …")
    for epoch in range(num_epochs):
        shadow_model.train()
        running_loss, correct, total = 0, 0, 0

        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            logits = shadow_model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            _, pred = logits.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()

        print(f"[Epoch {epoch + 1}/{num_epochs}] "
              f"Loss={running_loss / len(test_loader):.4f} "
              f"Acc={correct / total:.4f}")

    return model,shadow_model, train_set, rest_set
