import numpy as np
from layers.base_layer import BaseLayer

class ActivationLayer(BaseLayer):
    """
    Capa de Activación unificada que implementa múltiples funciones de activación.
    
    Las funciones de activación introducen no-linealidad en las redes neuronales,
    permitiendo que aprendan relaciones complejas. Sin funciones de activación,
    una red neuronal profunda sería equivalente a una regresión lineal.
    
    Funciones implementadas:
    - ReLU: Rectified Linear Unit - f(x) = max(0, x)
    - Sigmoid: f(x) = 1/(1 + e^(-x))  
    - Tanh: f(x) = tanh(x)
    - Softmax: f(x_i) = e^(x_i) / Σ(e^(x_j)) - para clasificación multiclase
    
    Todas las implementaciones están completamente vectorizadas para eficiencia máxima.
    
    Attributes:
        activation (str): Tipo de función de activación a aplicar
    """
    
    def __init__(self, activation='relu'):
        """
        Inicializa la capa de activación.
        
        Args:
            activation (str): Tipo de activación ('relu', 'sigmoid', 'tanh', 'softmax')
        """
        super().__init__()
        self.activation = activation
        self.activation_type = activation  # Para compatibilidad
        
    def forward(self, inputs):
        """
        Propagación hacia adelante - aplica la función de activación.
        
        Todas las implementaciones están optimizadas para estabilidad numérica
        y eficiencia computacional usando vectorización completa de NumPy.
        
        Args:
            inputs (np.ndarray): Tensor de entrada de cualquier forma
                                
        Returns:
            np.ndarray: Tensor con la función de activación aplicada
        """
        # Guardar entrada para el backward pass
        self.cache['inputs'] = inputs
        
        if self.activation == 'relu':
            # ReLU: f(x) = max(0, x)
            # Muy eficiente, ayuda con el problema de gradientes que desaparecen
            # VECTORIZACIÓN: np.maximum es completamente vectorizado
            return np.maximum(0, inputs)
            
        elif self.activation == 'sigmoid':
            # Sigmoid: f(x) = 1/(1 + e^(-x))
            # Clipping para evitar overflow numérico
            # OPTIMIZACIÓN: Limitamos valores extremos antes de exp()
            inputs_clipped = np.clip(inputs, -500, 500)
            output = 1 / (1 + np.exp(-inputs_clipped))
            self.cache['output'] = output
            return output
            
        elif self.activation == 'tanh':
            # Tanh: f(x) = tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
            # NumPy maneja automáticamente la estabilidad numérica
            # VECTORIZACIÓN: np.tanh es completamente optimizado
            output = np.tanh(inputs)
            self.cache['output'] = output
            return output
            
        elif self.activation == 'softmax':
            # Softmax: f(x_i) = e^(x_i) / Σ(e^(x_j))
            # ESTABILIDAD NUMÉRICA: Restamos el máximo para evitar overflow
            # Esta técnica es estándar y no afecta el resultado matemático
            shifted_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
            exp_inputs = np.exp(shifted_inputs)
            
            # VECTORIZACIÓN: Suma y división sobre axis=1 (dimensión de clases)
            output = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
            self.cache['output'] = output
            return output
            
        else:
            raise ValueError(f"Función de activación desconocida: {self.activation}")
    
    def backward(self, grad_output):
        """
        Propagación hacia atrás - calcula derivadas de las funciones de activación.
        
        Implementa las derivadas matemáticas exactas de cada función de activación,
        todas completamente vectorizadas para máxima eficiencia.
        
        Args:
            grad_output (np.ndarray): Gradientes de la capa siguiente
                                     
        Returns:
            np.ndarray: Gradientes multiplicados por la derivada de la activación
        """
        inputs = self.cache['inputs']
        
        if self.activation == 'relu':
            # Derivada de ReLU: f'(x) = 1 si x > 0, 0 si x <= 0
            # VECTORIZACIÓN: Comparación booleana se convierte automáticamente a 0/1
            return grad_output * (inputs > 0)
            
        elif self.activation == 'sigmoid':
            # Derivada de Sigmoid: f'(x) = f(x) * (1 - f(x))
            # Usamos la salida guardada para eficiencia (evitamos recalcular sigmoid)
            output = self.cache['output']
            # VECTORIZACIÓN: Operaciones elemento por elemento
            return grad_output * output * (1 - output)
            
        elif self.activation == 'tanh':
            # Derivada de Tanh: f'(x) = 1 - f(x)^2
            # Usamos la salida guardada para eficiencia
            output = self.cache['output']
            # VECTORIZACIÓN: Operaciones elemento por elemento
            return grad_output * (1 - output**2)
            
        elif self.activation == 'softmax':
            # Derivada de Softmax: f'(x) = f(x) * (grad - Σ(grad * f(x)))
            # Esta es la derivada exacta para softmax vectorizada
            output = self.cache['output']
            
            # VECTORIZACIÓN AVANZADA: Cálculo eficiente de la derivada softmax
            # Suma ponderada de gradientes por salidas, manteniendo dimensiones
            sum_term = np.sum(grad_output * output, axis=1, keepdims=True)
            return output * (grad_output - sum_term)
            
        else:
            raise ValueError(f"Función de activación desconocida: {self.activation}")
    
    def get_output_shape(self, input_shape):
        """
        Calcula la forma de salida de la activación.
        
        Las funciones de activación no cambian la forma de los datos,
        solo aplican transformaciones elemento por elemento.
        
        Args:
            input_shape (tuple): Forma de entrada
            
        Returns:
            tuple: Misma forma que la entrada
        """
        return input_shape


# Clases de conveniencia para compatibilidad y claridad de código
class ReLULayer(ActivationLayer):
    """
    Capa ReLU de conveniencia.
    
    ReLU (Rectified Linear Unit) es la función de activación más popular
    en redes neuronales modernas debido a su simplicidad y efectividad.
    
    Ventajas:
    - Computacionalmente eficiente
    - Ayuda a mitigar gradientes que desaparecen
    - Introduce esparsidad (muchas activaciones son exactamente 0)
    """
    
    def __init__(self):
        super().__init__('relu')


class SigmoidLayer(ActivationLayer):
    """
    Capa Sigmoid de conveniencia.
    
    Sigmoid es útil para problemas de clasificación binaria o cuando
    necesitamos salidas en el rango (0, 1).
    
    Características:
    - Salida en rango (0, 1)
    - Suave y diferenciable
    - Puede sufrir de gradientes que desaparecen en capas profundas
    """
    
    def __init__(self):
        super().__init__('sigmoid')


class TanhLayer(ActivationLayer):
    """
    Capa Tanh de conveniencia.
    
    Tanh es similar a sigmoid pero con salida en el rango (-1, 1),
    lo que puede ser útil para centrar los datos.
    
    Características:
    - Salida en rango (-1, 1)
    - Centrada en cero
    - Gradientes más fuertes que sigmoid
    """
    
    def __init__(self):
        super().__init__('tanh')


class SoftmaxLayer(ActivationLayer):
    """
    Capa Softmax de conveniencia.
    
    Softmax es la función estándar para clasificación multiclase,
    convirtiendo logits en una distribución de probabilidad.
    
    Características:
    - Salidas suman 1 (distribución de probabilidad)
    - Amplifica diferencias entre clases
    - Esencial para clasificación multiclase
    """
    
    def __init__(self):
        super().__init__('softmax')