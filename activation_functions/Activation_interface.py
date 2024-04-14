from abc import ABC, abstractmethod

class Activation_interface(ABC):

    @abstractmethod
    def forward(self, inputs): pass

    @abstractmethod
    def make_derivatives(self, inputs): pass
    