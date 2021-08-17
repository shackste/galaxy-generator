import matplotlib.pyplot as plt
import torch


class Losses:
    """ Container for all kinds of losses of Neural Networks 
        
        Parameter
        ---------
        type: str, loss category used as plot title
        label: str, identifier, used as plot label
        log: bool, indicate whether y-axis is log scaled in plot
        rate: bool, indicate whether to space y from 0 to 1
        
        Usage
        -----
        >>> losses = Losses("loss", "train")
        >>> for i, (y, y_hat) in enumerate(zip(prediction, target)):
        >>>     loss = mse(y_hat, y)
        >>>     losses.append(i, loss)
        >>> losses.plot()
    """
    
    def __init__(self, type: str, label: str, log=True, rate=False):
        self.label = label
        self.type = type
        self.losses = {}
    
    def append(self, iteration: int, loss: torch.Tensor) -> None:
        if loss is torch.Tensor:
            loss = loss.item()
        self.losses[iteration] = loss
    
    def plot(self, log=True, **kwargs):
        try:
            x, y = zip(*self.losses.items())
        except:
            return
        plt.plot(x, y, label=self.label, **kwargs)
        plt.xlabel("iteration")
        plt.title(self.type)
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        if log:
            plt.yscale("log")
        plt.grid(True)
#        plt.yaxis.set_ticks_position("both")
        plt.tight_layout()
        

class Accuracies(Losses):
    """ Container for accuracy measures """
    def plot(self, **kwargs):
        super(Accuracies, self).plot(log=False)
        plt.ylim(-0.1,1.1)