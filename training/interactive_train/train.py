import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import runhouse as rh


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
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


class TrainModel(rh.Module):
    def __init__(self,
                 learning_rate=0.001,
                 batch_size=64,
                 num_epochs=10,
                 output_path="~/.cache/runhouse/models/output"):
        super().__init__()
        self._output_path = output_path

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.criterion = nn.CrossEntropyLoss()
        self.model = None
        self.optimizer = None
        self.dataloader = None
        self.output_folder = None

    def load_dataloader(self):
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
        return data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def load_model(self):
        self.model = MyModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.dataloader = self.load_dataloader()

    def load_model_for_inference(self):
        # Instantiate the model
        if self.model is None:
            self.load_model()

        # Use the first checkpoint for this example
        checkpoint_path = self.output_folder.ls()[0]

        # Load the saved checkpoint
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Set the model to evaluation mode
        eval_model = self.model.eval()
        return eval_model

    def load_output_folder(self):
        self.output_folder = rh.folder(path=Path(self._output_path).expanduser()).mkdir()

    def save_checkpoint(self, epoch):
        if self.output_folder is None:
            self.load_output_folder()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        checkpoint_path = Path(self.output_folder.path) / f'epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

    def upload_outputs_to_s3(self):
        if self.output_folder is None:
            self.load_output_folder()

        self.output_folder.to("s3")

    def train_epoch(self):
        if self.model is None:
            self.load_model()

        for batch_idx, (inputs, targets) in enumerate(self.dataloader):
            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

    def inference(self, model, input_data):
        # Convert input data to a tensor if it's not already
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data)

        with torch.no_grad():
            # Ensure gradients are not computed
            output = model(input_data)

        return output


class MyDataset(data.Dataset):
    def __init__(self, data, targets):
        super().__init__()
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    sm_cluster = rh.sagemaker_cluster(name="rh-sagemaker-training",
                                      profile="sagemaker",
                                      instance_type="ml.g5.2xlarge").up_if_not().save()

    # Set random seed for reproducibility
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()

    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--output-path", type=str, default="~/.cache/runhouse/models/output",
                        help="Path to folder on the cluster to store model checkpoints")

    args = parser.parse_args()
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.epochs
    output_path = args.output_path

    # Checks if the module already exists on the cluster
    # If it does not, will put the module on the cluster and return the remote module object
    remote_model = TrainModel(learning_rate,
                              batch_size,
                              num_epochs,
                              output_path).get_or_to(sm_cluster, name="interactive_train")

    # Run training on the cluster
    for epoch in range(num_epochs):
        print(f"Running epoch {epoch} on cluster")
        remote_model.train_epoch()
        remote_model.save_checkpoint(epoch)

    # Example batch inference on some MNIST images
    model = remote_model.load_model_for_inference()
    batch_size = 5
    batch_data = torch.randn(batch_size, 784)
    output = remote_model.inference(model, batch_data)

    # Configure aws credentials on the cluster
    remote_model.system.sync_secrets(["aws"])

    # Send the checkpoints folder from the cluster to a s3 bucket
    remote_model.upload_outputs_to_s3()
