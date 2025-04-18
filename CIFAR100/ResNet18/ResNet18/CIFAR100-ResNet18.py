import torch
import torchvision
import torchvision.transforms as transforms
from torcheval.metrics import MulticlassAccuracy
from torcheval.metrics.functional import multiclass_accuracy

DATA_DIR = "../../data"

# Hyperparameters
##################
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
##################

# Loss function
#######################################
criterion = torch.nn.CrossEntropyLoss()
#######################################


def load_data(train: bool = True, data_dir: str = DATA_DIR):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=train, download=True, transform=transform
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=train, num_workers=6
    )

    return dataloader


# Main block
if __name__ == "__main__":
    # Get device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = load_data(train=True)
    test_loader = load_data(train=False)

    resnet18 = torchvision.models.resnet18()
    resnet18.to(device)

    optimizer = torch.optim.Adam(resnet18.parameters(), lr=LEARNING_RATE)

    metric = MulticlassAccuracy(device=device)

    for epoch in range(EPOCHS):
        # Loss for one eopch
        running_loss = 0.0

        for i, data in enumerate(train_loader, start=0):
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = resnet18(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)

            running_loss += loss.item()
            if i % 100 == 99:
                print(
                    f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f} accuracy: {multiclass_accuracy(predicted, labels.to(device)):.3f}"
                )
                running_loss = 0.0
    print("Finished Training")

    resnet18.eval()
    metric = MulticlassAccuracy(device=device)
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = resnet18(inputs.to(device))

            _, predicted = torch.max(outputs.data, 1)

            metric.update(predicted, labels.to(device))

        print(f"Accuracy on the test images: {metric.compute():.3f}")
