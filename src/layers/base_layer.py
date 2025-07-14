import numpy as np
from abc import ABC, abstractmethod

class BaseLayer(ABC):
    """
    Clase base abstracta para todas las capas de la red neuronal.
    
    Esta clase define la interfaz común que deben implementar todas las capas,
    incluyendo los métodos forward y backward para la propagación hacia adelante
    y hacia atrás respectivamente.
    
    Attributes:
        params (dict): Diccionario que almacena los parámetros entrenables de la capa
        gradients (dict): Diccionario que almacena los gradientes de los parámetros
        cache (dict): Diccionario para almacenar valores necesarios durante backward pass
    """
    
    def __init__(self):
        """Inicializa la capa base con diccionarios vacíos para parámetros y gradientes."""
        self.params = {}
        self.gradients = {}
        self.cache = {}
        
    @abstractmethod
    def forward(self, inputs):
        """
        Propagación hacia adelante de la capa.
        
        Args:
            inputs: Tensor de entrada a la capa
            
        Returns:
            Tensor de salida de la capa
        """
        pass
        
    @abstractmethod
    def backward(self, grad_output):
        """
        Propagación hacia atrás de la capa.
        
        Args:
            grad_output: Gradientes que llegan desde la capa siguiente
            
        Returns:
            Gradientes que se propagan a la capa anterior
        """
        pass
        
    def get_params(self):
        """
        Obtiene los parámetros entrenables de la capa.
        
        Returns:
            dict: Diccionario con los parámetros de la capa
        """
        return self.params
        
    def get_gradients(self):
        """
        Obtiene los gradientes de los parámetros de la capa.
        
        Returns:
            dict: Diccionario con los gradientes de la capa
        """
        return self.gradients
    
    def get_output_shape(self, input_shape):
        """
        Calcula la forma de salida dada una forma de entrada.
        
        Args:
            input_shape: Tupla con las dimensiones de entrada
            
        Returns:
            tuple: Forma de salida de la capa
        """
        # Implementación por defecto - las subclases pueden sobrescribir
        return input_shape