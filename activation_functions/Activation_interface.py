'''
Andre Palacio Braga Tivo 13835534
João Pedro Gonçalves Vilela 13731070
Lucas Muniz de Lima 13728941
Lucas Toupitzen Ferracin Garcia 11804164
'''


from abc import ABC, abstractmethod

class Activation_interface(ABC):

    @abstractmethod
    def forward(self, inputs): pass

    @abstractmethod
    def make_derivatives(self, inputs): pass
    