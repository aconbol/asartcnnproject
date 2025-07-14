import numpy as np
from layers.base_layer import BaseLayer

class MaxPool2DLayer(BaseLayer):
    """
    Capa de Max Pooling 2D optimizada para redes neuronales convolucionales.
    
    El Max Pooling es una técnica de submuestreo que reduce la dimensionalidad
    espacial de los mapas de características mientras retiene la información
    más importante. Esto ayuda a:
    
    1. Reducir el costo computacional
    2. Controlar el sobreajuste
    3. Proporcionar invariancia a pequeñas traslaciones
    4. Extraer características dominantes
    
    Esta implementación está optimizada usando vectorización eficiente de NumPy,
    evitando bucles anidados excesivos que causan lentitud.
    
    Attributes:
        pool_size (int): Tamaño de la ventana de pooling (pool_size x pool_size)
        stride (int): Paso al deslizar la ventana de pooling
    """
    
    def __init__(self, pool_size=2, stride=2):
        """
        Inicializa la capa de Max Pooling.
        
        Args:
            pool_size (int, optional): Tamaño de la ventana de pooling. Default: 2
            stride (int, optional): Paso de la ventana. Default: 2
        """
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        
    def forward(self, inputs):
        """
        Propagación hacia adelante - aplica max pooling.
        
        Para cada ventana de pooling, selecciona el valor máximo.
        Esto preserva las características más prominentes mientras
        reduce la dimensionalidad espacial.
        
        Proceso matemático:
        - Para cada posición (i,j) en la salida:
          output[n,c,i,j] = max(input[n,c,i*stride:i*stride+pool_size,j*stride:j*stride+pool_size])
        
        Args:
            inputs (np.ndarray): Tensor de entrada de forma (batch_size, channels, height, width)
            
        Returns:
            np.ndarray: Tensor de salida de forma (batch_size, channels, out_height, out_width)
        """
        # Guardar entrada para el backward pass
        self.cache['inputs'] = inputs
        batch_size, channels, height, width = inputs.shape
        
        # Calcular dimensiones de salida
        # out_size = (input_size - pool_size) / stride + 1
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        # Inicializar tensor de salida
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        # OPTIMIZACIÓN CLAVE: Solo iterar sobre posiciones espaciales
        # Evitamos bucles sobre batch y canales usando vectorización
        for i in range(out_height):
            for j in range(out_width):
                # Calcular posiciones de la ventana
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                # Extraer ventana de pooling para todos los ejemplos y canales
                # Forma: (batch_size, channels, pool_size, pool_size)
                input_slice = inputs[:, :, h_start:h_end, w_start:w_end]
                
                # VECTORIZACIÓN: Aplicar max sobre las dimensiones espaciales
                # np.max con axis=(2,3) encuentra el máximo en la ventana pool_size x pool_size
                # para todos los ejemplos del batch y canales simultáneamente
                output[:, :, i, j] = np.max(input_slice, axis=(2, 3))
                
        return output
        
    def backward(self, grad_output):
        """
        Propagación hacia atrás - distribuye gradientes solo a posiciones máximas.
        
        En max pooling, el gradiente solo se propaga hacia atrás a las
        posiciones que contenían el valor máximo en el forward pass.
        Las demás posiciones reciben gradiente cero.
        
        Esta implementación está optimizada para evitar bucles anidados
        excesivos sobre batch y canales.
        
        Args:
            grad_output (np.ndarray): Gradientes de la capa siguiente
            
        Returns:
            np.ndarray: Gradientes que se propagan a la capa anterior
        """
        inputs = self.cache['inputs']
        batch_size, channels, height, width = inputs.shape
        
        # Inicializar gradiente de entrada
        grad_input = np.zeros_like(inputs)
        
        out_height, out_width = grad_output.shape[2], grad_output.shape[3]
        
        # OPTIMIZACIÓN CLAVE: Solo iterar sobre posiciones espaciales
        # Procesamos batch y canales simultáneamente con vectorización
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                # Extraer ventana de entrada correspondiente
                input_slice = inputs[:, :, h_start:h_end, w_start:w_end]
                
                # VECTORIZACIÓN: Crear máscara para posiciones máximas
                # Encuentra dónde están los valores máximos en cada ventana
                max_values = np.max(input_slice, axis=(2, 3), keepdims=True)
                max_mask = (input_slice == max_values)
                
                # Distribuir gradiente solo a las posiciones máximas
                # El gradiente se multiplica por la máscara y se añade a grad_input
                grad_contribution = (
                    max_mask * grad_output[:, :, i, j].reshape(batch_size, channels, 1, 1)
                )
                grad_input[:, :, h_start:h_end, w_start:w_end] += grad_contribution
                
        return grad_input
    
    def get_output_shape(self, input_shape):
        """
        Calcula la forma de salida de la capa de max pooling.
        
        Args:
            input_shape (tuple): Forma de entrada (channels, height, width)
            
        Returns:
            tuple: Forma de salida (channels, out_height, out_width)
        """
        channels, height, width = input_shape
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        return (channels, out_height, out_width)


class GlobalAveragePool2DLayer(BaseLayer):
    """
    Capa de Global Average Pooling 2D optimizada.
    
    El Global Average Pooling calcula el promedio de todos los valores
    en cada mapa de características, reduciendo cada mapa a un solo valor.
    Es útil como alternativa a capas densas antes de la clasificación final.
    
    Ventajas:
    - Reduce drasticamente el número de parámetros
    - Actúa como regularizador natural
    - Mantiene correspondencia entre mapas y clases
    """
    
    def __init__(self):
        """Inicializa la capa de Global Average Pooling."""
        super().__init__()
        
    def forward(self, inputs):
        """
        Propagación hacia adelante - calcula promedio global.
        
        Para cada mapa de características, calcula el promedio de todos
        los valores espaciales, reduciendo de (H, W) a un escalar.
        
        Args:
            inputs (np.ndarray): Tensor de entrada (batch_size, channels, height, width)
            
        Returns:
            np.ndarray: Tensor de salida (batch_size, channels)
        """
        # Guardar entrada para backward pass
        self.cache['inputs'] = inputs
        
        # VECTORIZACIÓN: Promedio sobre dimensiones espaciales
        # axis=(2, 3) corresponde a height y width
        return np.mean(inputs, axis=(2, 3))
        
    def backward(self, grad_output):
        """
        Propagación hacia atrás - distribuye gradientes uniformemente.
        
        El gradiente se distribuye uniformemente a todas las posiciones
        espaciales, dividido por el número total de elementos.
        
        Args:
            grad_output (np.ndarray): Gradientes de la capa siguiente
            
        Returns:
            np.ndarray: Gradientes distribuidos uniformemente
        """
        inputs = self.cache['inputs']
        batch_size, channels, height, width = inputs.shape
        
        # Calcular gradiente por píxel
        grad_per_pixel = grad_output / (height * width)
        
        # VECTORIZACIÓN OPTIMIZADA: Expandir dimensiones y usar broadcasting
        # En lugar de bucles, usamos reshape y broadcasting de NumPy
        grad_input = np.broadcast_to(
            grad_per_pixel.reshape(batch_size, channels, 1, 1),
            (batch_size, channels, height, width)
        ).copy()  # copy() para hacer el array escribible
        
        return grad_input
    
    def get_output_shape(self, input_shape):
        """
        Calcula la forma de salida del global average pooling.
        
        Args:
            input_shape (tuple): Forma de entrada (channels, height, width)
            
        Returns:
            tuple: Forma de salida (channels,) - solo queda la dimensión de canales
        """
        channels, height, width = input_shape
        return (channels,)