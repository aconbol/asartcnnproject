import numpy as np
from layers.base_layer import BaseLayer

class FlattenLayer(BaseLayer):
    """
    Capa de Aplanado optimizada que convierte tensores multidimensionales en vectores.
    
    Esta capa es esencial para la transición entre capas convolucionales
    (que operan en 2D/3D) y capas densas (que operan en 1D).
    
    Convierte una entrada de forma (batch_size, channels, height, width)
    a (batch_size, channels * height * width).
    
    No tiene parámetros entrenables - solo reorganiza los datos de manera eficiente.
    
    Ejemplo de transformación:
    - Entrada:  (32, 64, 7, 7)   # 32 imágenes, 64 canales, 7x7 píxeles
    - Salida:   (32, 3136)       # 32 vectores de 3136 características (64*7*7)
    """
    
    def __init__(self):
        """Inicializa la capa de aplanado sin parámetros."""
        super().__init__()
        
    def forward(self, inputs):
        """
        Propagación hacia adelante - aplana la entrada eficientemente.
        
        Convierte un tensor multidimensional en un vector manteniendo
        la primera dimensión (batch_size) intacta.
        
        La operación es completamente vectorizada usando NumPy reshape,
        lo que la hace muy eficiente computacionalmente.
        
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
        
        # VECTORIZACIÓN OPTIMIZADA: reshape es una operación O(1) en NumPy
        # -1 indica que NumPy calcula automáticamente esta dimensión
        # No hay copia de datos, solo cambio de vista
        return inputs.reshape(batch_size, -1)
    
    def backward(self, grad_output):
        """
        Propagación hacia atrás - restaura la forma original eficientemente.
        
        El aplanado es una operación completamente reversible,
        así que simplemente restauramos la forma original de los gradientes.
        Esta operación también es O(1) en términos de memoria.
        
        Args:
            grad_output (np.ndarray): Gradientes aplanados de la capa siguiente
                                     Forma: (batch_size, features)
                                     
        Returns:
            np.ndarray: Gradientes con forma original restaurada
                       Forma: (batch_size, dim1, dim2, ...)
        """
        input_shape = self.cache['input_shape']
        
        # VECTORIZACIÓN OPTIMIZADA: reshape inverso, también O(1)
        # No hay copia de datos, solo cambio de vista
        return grad_output.reshape(input_shape)
    
    def get_output_shape(self, input_shape):
        """
        Calcula la forma de salida del aplanado.
        
        Args:
            input_shape (tuple): Forma de entrada excluyendo batch_size
                                Por ejemplo: (channels, height, width)
            
        Returns:
            tuple: Forma de salida (total_features,)
                  donde total_features = channels * height * width
        """
        # Calcular número total de características
        # np.prod multiplica todos los elementos de la tupla
        total_features = np.prod(input_shape)
        return (total_features,)