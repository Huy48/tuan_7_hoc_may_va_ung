from flask import Flask, render_template, jsonify
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision
from torch.optim import SGD
from torchvision.transforms import transforms
from io import BytesIO
import base64

# Cấu hình Flask
app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Hàm hiển thị hình ảnh
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close()
    return img_str

@app.route('/')
def home():
    # Lấy ảnh từ batch đầu tiên trong trainloader
    images, _ = next(iter(trainloader))
    img_str = imshow(torchvision.utils.make_grid(images))
    
    # Trả về ảnh và mô hình
    return render_template('index.html', img_data=img_str)

@app.route('/train')
def train():
    n_features = 32 * 32 * 3
    model = getModel(n_features).to(device)
    lr = 0.1
    optim = SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n_epochs = 35
    train_losses, test_losses, test_accuracies = [], [], []

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        total = 0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optim.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optim.step()
            running_loss += loss.item()

        # Đánh giá trên tập test
        test_loss, test_accuracy = evaluate(model, testloader, loss_fn)
        train_losses.append(running_loss / len(trainloader))
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    # Tạo đồ thị Loss và Accuracy
    buffer = BytesIO()
    plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.title('Loss Epochs')
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.subplot(122)
    plt.title('Accuracy Epoch')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.legend()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    graph_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close()

    return render_template('training_results.html', img_data=graph_img)

def getModel(n_features):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_features, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    return model

def evaluate(model, testloader, criterion):
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return test_loss / len(testloader), accuracy

if __name__ == '__main__':
    app.run(debug=True)
