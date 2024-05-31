import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
import time
from dataset import CUB 
from torch.utils.tensorboard import SummaryWriter

def main():
    # Set the start time
    time1 = time.time()
    writer = SummaryWriter()

    # Define image size and normalization parameters
    IMAGE_SIZE = 448
    TRAIN_MEAN = [0.48560741861744905, 0.49941626449353244, 0.43237713785804116]
    TRAIN_STD = [0.2321024260764962, 0.22770540015765814, 0.2665100547329813]
    TEST_MEAN = [0.4862169586881995, 0.4998156522834164, 0.4311430419332438]
    TEST_STD = [0.23264268069040475, 0.22781080253662814, 0.26667253517177186]

    # Dataset path
    path = 'CUB_200_2011'

    # Define data preprocessing
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(TEST_MEAN, TEST_STD)
    ])

    # Load datasets
    train_dataset = CUB(path, train=True, transform=train_transforms)
    test_dataset = CUB(path, train=False, transform=test_transforms)

    # Setup DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # ResNet18 
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 200)  # 200 bird classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # 设置不同的学习率
    optimizer = optim.SGD([
        {'params': model.fc.parameters(), 'lr': 0.01},  # 对于新的输出层使用更高的学习率
        {'params': (p for n, p in model.named_parameters() if 'fc' not in n), 'lr': 0.001}  # 对于其余层使用较低的学习率
    ], momentum=0.9)

    def train_model(num_epochs=10):
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            total_samples = 0

            for images, labels in train_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

            epoch_loss = running_loss / total_samples
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')
            writer.add_scalar('Loss/Train', epoch_loss, epoch)

    def evaluate_model(epoch=10):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy}%')
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)

    train_model()
    evaluate_model()
    writer.close()

    # Calculate total time elapsed
    time2 = time.time()
    print(f"Total Time: {time2 - time1} seconds")

if __name__ == '__main__':
    main()
