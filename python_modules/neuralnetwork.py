import torch

from file_system import folder_results

class NeuralNetwork(torch.nn.Module):
    """ base class with convenient procedures used by all NN"""
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.parameter_file = f"{folder_results}parameter_state_dict_{self._get_name()}.pth"
        # self.cuda() ## all NN shall run on cuda ### doesnt seem to work

    def save_model(self, file: str):
        """ save full model to file """
        torch.save(self, file)

    def save(self) -> None:
        """ save learned parameters to parameter_file """
        torch.save(self.state_dict(), self.parameter_file)

    def load(self) -> None:
        """ load learned parameters from parameter_file """
        self.load_state_dict(torch.load(self.parameter_file))

    @staticmethod
    def same_padding(kernel_size=1) -> float:
        """ return padding required to mimic 'same' padding in tensorflow """
        return (kernel_size-1) // 2

    def set_optimizer(self, optimizer, **kwargs) -> None:
        self.optimizer = optimizer(self.parameters(), **kwargs)
        
    def get_total_number_parameters(self) -> float:
        """ return total number of parameters """
        return sum([p.numel() for p in classifier.parameters()])

    def zero_grad(self):
        """ faster implementation of zero_grad """
        for p in self.parameters():
            p.grad = None
#        self.zero_grad(set_to_none=True)


def update_networks_on_loss(loss: torch.Tensor, *networks) -> None:
    if not loss:
        return
    for network in networks:
        network.zero_grad()
    loss.backward()
    for network in networks:
        network.optimizer.step()
