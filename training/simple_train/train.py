import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import runhouse as rh


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class MyDataset(data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)


def train_model(learning_rate, batch_size, num_epochs):
    # Load the MNIST dataset and apply transformations
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )

    # Extract the training data and targets
    train_data = (
            train_dataset.data.view(-1, 28 * 28).float() / 255.0
    )  # Flatten and normalize the input
    train_targets = train_dataset.targets

    dataset = MyDataset(train_data, train_targets)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize your model
    model = MyModel()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

    # Save the model state on the cluster, and return a file object pointing to the path on the cluster's file system
    # where it is stored
    model_file = rh.file(name="simple_model")
    model_file.write(data=model.state_dict())

    return model_file


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()

    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")

    args = parser.parse_args()

    # Launch the SageMaker cluster (if not already up)
    sm_cluster = rh.sagemaker_cluster(name="rh-sagemaker-training",
                                      profile="sagemaker",
                                      instance_type="ml.g5.2xlarge").up_if_not().save()

    remote_train = rh.function(train_model).to(sm_cluster)

    # Call the training stub, which executes remotely on the SageMaker compute
    remote_model = remote_train(learning_rate=args.learning_rate,
                                batch_size=args.batch_size,
                                num_epochs=args.epochs)

    print(f"Saved model state on cluster to path: {remote_model.path}")

