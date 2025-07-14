import numpy as np
from layers.base_layer import BaseLayer

class DenseLayer(BaseLayer):
    """
    Capa Densa (Fully Connected) optimizada para redes neuronales.
    
    Una capa densa conecta cada neurona de entrada con cada neurona de salida,
    realizando una transformación lineal seguida de la suma de un sesgo.
    Es fundamental para la clasificación final en CNNs después de extraer
    características con capas convolucionales.
    
    Operación matemática:
    output = input @ W + b
    donde:
    - input: vector de entrada de tamaño (batch_size, input_size)
    - W: matriz de pesos de tamaño (input_size, units)
    - b: vector de sesgos de tamaño (1, units)
    - @: operación de multiplicación de matrices
    
    Attributes:
        units (int): Número de neuronas (unidades) en esta capa
    """
    
    def __init__(self, units):
        """
        Inicializa la capa densa.
        
        Args:
            units (int): Número de neuronas de salida en esta capa
        """
        super().__init__()
        self.units = units
        
    def initialize_params(self, input_size):
        """
        Inicializa los parámetros usando Xavier/Glorot Uniform initialization.
        
        Esta inicialización mantiene la varianza de las activaciones estable
        a través de las capas, ayudando al entrenamiento eficiente.
        
        La fórmula Xavier Uniform es:
        W ~ U[-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out))]
        
        Args:
            input_size (int): Número de características de entrada
        """
        # Xavier/Glorot Uniform initialization
        # Límite basado en el número de neuronas de entrada y salida
        limit = np.sqrt(6.0 / (input_size + self.units))
        
        # Pesos: matriz de transformación lineal
        self.params['W'] = np.random.uniform(-limit, limit, (input_size, self.units))
        
        # Sesgos: inicializados en cero (práctica estándar)
        self.params['b'] = np.zeros((1, self.units))
        
    def forward(self, inputs):
        """
        Propagación hacia adelante - transformación lineal.
        
        Realiza la operación: output = input @ W + b
        
        Si la entrada tiene más de 2 dimensiones (como salida de capas conv),
        la aplana automáticamente manteniendo la dimensión del batch.
        
        Args:
            inputs (np.ndarray): Tensor de entrada, puede ser:
                                - 2D: (batch_size, features) 
                                - 4D: (batch_size, channels, height, width) - se aplana
                                
        Returns:
            np.ndarray: Salida de forma (batch_size, units)
        """
        # Aplanar entrada si es necesario (para salidas de capas convolucionales)
        if len(inputs.shape) > 2:
            batch_size = inputs.shape[0]
            # Aplanar todas las dimensiones excepto la primera (batch)
            inputs_flattened = inputs.reshape(batch_size, -1)
        else:
            inputs_flattened = inputs
            
        # Inicializar parámetros automáticamente si no existen
        if 'W' not in self.params:
            self.initialize_params(inputs_flattened.shape[1])
            
        # Guardar entrada para el backward pass
        self.cache['X'] = inputs_flattened
        
        # VECTORIZACIÓN: Operación matricial eficiente
        # np.dot realiza multiplicación de matrices optimizada
        output = np.dot(inputs_flattened, self.params['W']) + self.params['b']
        
        return output
    
    def backward(self, grad_output):
        """
        Propagación hacia atrás - calcula gradientes de parámetros y entrada.
        
        Calcula tres tipos de gradientes usando regla de la cadena:
        1. ∂L/∂W = X^T @ grad_output (gradiente de pesos)
        2. ∂L/∂b = sum(grad_output, axis=0) (gradiente de sesgos)
        3. ∂L/∂X = grad_output @ W^T (gradiente de entrada)
        
        Args:
            grad_output (np.ndarray): Gradientes de la capa siguiente
                                     Forma: (batch_size, units)
                                     
        Returns:
            np.ndarray: Gradientes que se propagan a la capa anterior
                       Forma: (batch_size, input_size)
        """
        X = self.cache['X']
        batch_size = X.shape[0]
        
        # Gradiente respecto a los pesos (∂L/∂W)
        # Forma: (input_size, units)
        # Dividimos por batch_size para promediar sobre el lote
        self.gradients['W'] = np.dot(X.T, grad_output) / batch_size
        
        # Gradiente respecto a los sesgos (∂L/∂b)
        # Forma: (1, units)
        # Promediamos sobre todas las muestras del batch
        self.gradients['b'] = np.mean(grad_output, axis=0, keepdims=True)
        
        # Gradiente respecto a la entrada (∂L/∂X)
        # Forma: (batch_size, input_size)
        # VECTORIZACIÓN: Operación matricial eficiente
        grad_input = np.dot(grad_output, self.params['W'].T)
        
        return grad_input
    
    def get_output_shape(self, input_shape):
        """
        Calcula la forma de salida de la capa densa.
        
        Args:
            input_shape (tuple): Forma de entrada (puede ser multidimensional)
            
        Returns:
            tuple: Forma de salida (units,)
        """
        return (self.units,)


class FlattenLayer(BaseLayer):
    """
    Capa de Aplanado que convierte tensores multidimensionales en vectores.
    
    Esta capa es esencial para la transición entre capas convolucionales
    (que operan en 2D/3D) y capas densas (que operan en 1D).
    
    Convierte una entrada de forma (batch_size, channels, height, width)
    a (batch_size, channels * height * width).
    
    No tiene parámetros entrenables - solo reorganiza los datos.
    """
    
    def __init__(self):
        """Inicializa la capa de aplanado."""
        super().__init__()
        
    def forward(self, inputs):
        """
        Propagación hacia adelante - aplana la entrada.
        
        Convierte un tensor multidimensional en un vector manteniendo
        la primera dimensión (batch_size) intacta.
        
        Ejemplo:
        Entrada: (32, 64, 7, 7) -> Salida: (32, 3136)
        donde 3136 = 64 * 7 * 7
        
        Args:
            inputs (np.ndarray): Tensor de entrada de cualquier forma
                                (batch_size, dim1, dim2, ...)
                                
        Returns:
            np.ndarray: Tensor aplanado de forma (batch_size, features)
                       donde features = dim1 * dim2 * ...
        """
        # Guardar forma original para el backward pass
        self.cache['input_shape'] = inputs.shape
        
        batch_size = inputs.shape[0]
        
        # VECTORIZACIÓN: reshape eficiente
        # -1 indica que NumPy debe calcular automáticamente esta dimensión
        return inputs.reshape(batch_size, -1)
    
    def backward(self, grad_output):
        """
        Propagación hacia atrás - restaura la forma original.
        
        El aplanado es una operación reversible, así que simplemente
        restauramos la forma original de los gradientes.
        
        Args:
            grad_output (np.ndarray): Gradientes aplanados de la capa siguiente
                                     Forma: (batch_size, features)
                                     
        Returns:
            np.ndarray: Gradientes con forma original restaurada
                       Forma: (batch_size, dim1, dim2, ...)
        """
        input_shape = self.cache['input_shape']
        
        # VECTORIZACIÓN: reshape eficiente para restaurar forma
        return grad_output.reshape(input_shape)
    
    def get_output_shape(self, input_shape):
        """
        Calcula la forma de salida del aplanado.
        
        Args:
            input_shape (tuple): Forma de entrada (batch_size se excluye del cálculo)
            
        Returns:
            tuple: Forma de salida (total_features,)
        """
        # Calcular número total de características
        total_features = np.prod(input_shape)
        return (total_features,)