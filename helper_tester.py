import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class ModelTesterMetrics():
    def __init__(self) -> None:
        self.loss       = torch.nn.L1Loss()
        self.activation = torch.nn.Identity()

        self.loss_values     : list[float] = []
        self.accuracy_values : list[float] = [] 
        
        self.x_pred  : list[int] = []
        self.y_truth : list[int] = []

    def reset(self) -> None:
        self.loss_values     = []
        self.accuracy_values = []
        self.x_pred          = []
        self.y_truth         = []

    def compute_loss(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        return self.loss(x, y)
    
    def compute_accuracy(self, x : torch.Tensor, y : torch.Tensor) -> float:
        
        x = self.activation(x)
        x = x.to('cpu').detach()
        y = y.to('cpu').detach()

        ax = torch.argmax(x, dim = 1)
        ay = torch.argmax(y, dim = 1)

        self.x_pred  += ax.tolist()
        self.y_truth += ay.tolist()

        total_correct = torch.sum(ax == ay)
        batch_size    = x.shape[0]

        accuracy = total_correct / batch_size
        return float(accuracy.item())

    def compute(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        ls = self.compute_loss(x, y)
        ac = self.compute_accuracy(x, y)
        self.loss_values.append(ls.item())
        self.accuracy_values.append(ac)
        return ls
    
    def average_loss(self) -> float:
        return np.mean(self.loss_values)

    def average_accuracy(self) -> float:
        return np.mean(self.accuracy_values) 

    def report(self) -> str:
        return classification_report(self.y_truth, self.x_pred)

    def confusion(self) -> str:
        return str(confusion_matrix(self.y_truth, self.x_pred))
    
if __name__ == "__main__":
    print("Tester !")

    metrics = ModelTesterMetrics()
    metrics.reset()

    x = torch.rand(1, 3)
    y = torch.rand(1, 3)

    ls = metrics.compute(x, y)
    print(ls)

    print("Test Accuracy :", metrics.average_accuracy())
    print("Test Loss     :", metrics.average_loss())