import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from operator import mul


class NogradModule:
    """
    A wrapper class over Pytorch networks that allows users to easily take and
    update all of the weights of the underlying Pytorch network without pesky
    gradients and such. This is for optimization techniques (such as genetic
    algorithms) that are gradient-free.

    Example
    -------
    >>> model = NogradModule(nn.Sequential(
            nn.Linear(5,10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
            nn.Linear(10,2),
            nn.Softmax(dim=0)
        ))
    >>> model.size
    192
    >>> model.shape
    (torch.Size([10, 5]),
     torch.Size([10]),
     torch.Size([10, 10]),
     torch.Size([10]),
     torch.Size([2, 10]),
     torch.Size([2]))
    >>> model.values = torch.zeros(model.size)
    >>> model.values[0]
    tensor(0.)
    """
    
    def __init__(self, model):
        """
        Initialize the module with an existing Pytorch module
        
        Params
        ------
        model: nn.Module
            underlying Pytorch network
        """
        self.model = model
        
        # tuple of shapes of every parameter layer of the network
        self.shape = tuple(
            values.shape for values in self.model.state_dict().values()
        )
        
        # total number of weights in the network
        self.size = sum(reduce(mul, shape) for shape in self.shape)
        
    @property
    def values(self):
        """
        Returns a completely flattened representation of all of the weights of
        each layer of the underlying network.
        
        Returns
        -------
        values: torch.tensor
            Flattened representation of the underlying network
        """
        return torch.cat([
            v.reshape(-1) for v in self.model.state_dict().values()
        ])
    
    @values.setter
    def values(self, new_values):
        """
        Modifies the underlying network by accepting a flattened representation
        of all of the network's weights.
        
        Params
        ------
        new_values: torch.tensor
            Tensor of new values to be set to the network
            
        Preconditions
        -------------
        (verified by the method)
        
        new_values.size() == self.size
        """
        assert new_values.size()[0] == self.size, "Error"
        
        new_dict = {}  # to update the state_dict
        index = 0     
        
        # loop through every layer of the model
        for key, shape in zip(self.model.state_dict().keys(), self.shape):
            # size of the current layer
            block_size = reduce(mul, shape) 
            
            # select the corresponding block in the new_values
            block = new_values[index:index+block_size][:].reshape(shape)
            
            # push the block to the dictionary
            new_dict[key] = block
            
            # move the block forward
            index += block_size
            
        self.model.load_state_dict(new_dict)
        
    def __str__(self):
        """
        Currently just returns the __repr__ method, idk I think it's fine.
        
        Returns
        -------
        class_str: str
            User-friendly representation of the model as a string
        """
        return repr(self)
    
    def __repr__(self):
        """
        Wraps the Nograd class name around the underlying torch Module repr 
        method, also records the size of the network.
        
        Returns
        -------
        class_repr: str
            Technical representation of the model as a string
        """
        return f"Nograd({repr(self.model)}, size={self.size})"
