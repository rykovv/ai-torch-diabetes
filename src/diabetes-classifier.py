# feedforward nn diabetes classifier
# predict whether a person has diabetes or not

from typing import Tuple

# vector, matrix operations
import numpy as np

# nn modeling
import torch
import torch.nn as nn

# preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# dataset building
from torch.utils.data import Dataset, DataLoader


# load dataset with pandas
def load_dataset() -> pd.DataFrame:
    """Load dataset as Pandas DataFrame"""
    return pd.read_csv("data/diabetes.csv")


def preprocess(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess input data extracting X (input) and Y (output)"""
    # iloc uses interger base indexing on pd DataFrames
    # values returns numpy representation of data
    X = data.iloc[:, 0:-1].values
    slY = data.iloc[:, -1].values
    blY = [y == "positive" for y in slY]
    # float64 for coherency with X
    Y = np.array(blY, dtype="float64")

    return X, Y


def normalize(features: np.ndarray, sMethod: str) -> np.ndarray:
    """Normalize every feature to be in specific range"""

    if sMethod == "minmax":
        mmsc = MinMaxScaler((-1, 1))
        return mmsc.fit_transform(features)

    sc = StandardScaler()
    return sc.fit_transform(features)


def to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array)


# build custom data set for data loading during training
class DatasetFF(Dataset):
    """Custom dataset class to feed into Pytorch loader"""

    def __init__(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        super().__init__()
        self.X = X
        self.Y = Y

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.Y[index]

    def __len__(self) -> int:
        return len(self.X)


data = load_dataset()

X, Y = preprocess(data)

X = normalize(X, "minmax")

X = to_tensor(X)
# unsqueeze is needed to add one dimension for bce to work properly
# shape from [768] -> [768, 1]
# X.shape [768, 7]
# Y.shape [768, 1]
Y = to_tensor(Y).unsqueeze(1)

dataset = DatasetFF(X, Y)
loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)


# build model class
class DiabetesClassifier(nn.Module):
    """Feedforward NN model for diabetes classification"""

    def __init__(self, in_features: int, out_features: int) -> None:
        super(DiabetesClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features, 5)
        self.fc2 = nn.Linear(5, 4)
        self.fc3 = nn.Linear(4, 3)
        self.fc4 = nn.Linear(3, 2)
        self.fc5 = nn.Linear(2, out_features)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = self.fc4(out)
        out = self.tanh(out)
        out = self.fc5(out)
        return self.sigmoid(out)


in_features: int = X.shape[1]
out_features: int = Y.shape[1]
net = DiabetesClassifier(in_features, out_features)

# losses are averaged over observations for each minibatch
criterion = nn.BCELoss(reduction="mean")
# will use SGD with Momentum
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)


def train(
    net: nn.Module,
    epochs: int,
    loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
) -> None:
    for epoch in range(epochs):
        for inputs, labels in loader:
            inputs = inputs.float()
            labels = labels.float()
            # same as net.forward(inputs)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # Model parameters are stored in the nn
            # but optimizer keeps a parameters buffer
            # which is a copy of the model parameters.
            # Need to zero out because the default
            # action is to accumulate which is not
            # necessary in this case.
            optimizer.zero_grad()
            # back prop
            loss.backward()
            # weights update (w = w - lr*gradient)
            optimizer.step()

        # Accuracy calculation
        # consider positive if output > 0.5
        output = (outputs > 0.5).float()
        accuracy = (output == labels).float().mean()
        print(f"Epoch {epoch}/{epochs}: loss {loss:.3f}, accuracy {accuracy:.3f}")


epochs = 200
train(net, epochs, loader, criterion, optimizer)


def predict(
    net: nn.Module, inputs: torch.Tensor, threashold: torch.float64
) -> torch.Tensor:
    return net.forward(inputs) > threashold
