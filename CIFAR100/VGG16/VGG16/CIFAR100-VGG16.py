import torch
import torchvision
import torchvision.transforms as transforms

DATA_DIR = "../data"

# Hyperparameters
##################
EPOCHS = 10
BATCH_SIZE = 32
##################

# Loss function
#######################################
criterion = torch.nn.CrossEntropyLoss()
#######################################


def load_data(train=True, data_dir=DATA_DIR):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=train, download=True, transform=transform
    )

    return dataset


# Main block
if __name__ == "__main__":
    # Get device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vgg16 = torchvision.models.vgg16()
    vgg16.to(device)

    optimizer = torch.optim.Adam(vgg16.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        # Loss for one eopch
        running_loss = 0.0
