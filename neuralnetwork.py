from torch import load
from torch.nn.import Module

from file_system import folder_results

class NeuralNetwork(Module):
    """ base class with convenient procedures used by all NN"""
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.parameter_file = f"{folder_results}parameter_state_dict_{self._get_name()}.pth"
        # self.cuda() ## all NN shall run on cuda ### doesnt seem to work
    
    def save(self):
        torch.save(self.state_dict(), self.parameter_file)

    def load(self):
        self.load_state_dict(load(self.parameter_file))
        self.eval()
