import numpy as np
from layers.base_layer import BaseLayer

class Conv2DLayer(BaseLayer):
    """
    Capa Convolucional 2D optimizada para redes neuronales convolucionales.
    
    Esta implementación optimizada combina la eficiencia de la versión rápida 
    de cnn_numpy con explicaciones académicas detalladas. Utiliza vectorización
    eficiente manteniendo claridad educativa.
    
    La convolución es una operación fundamental en CNNs que aplica filtros (kernels)
    a través de la imagen de entrada para extraer características locales como
    bordes, texturas y patrones.
    
    Attributes:
        num_filters (int): Número de filtros/kernels a aplicar
        filter_size (int): Tamaño del filtro (asumiendo filtros cuadrados)
        stride (int): Paso del filtro al deslizarse sobre la imagen
        padding (int): Cantidad de padding a añadir alrededor de la imagen
    """
    
    def __init__(self, num_filters, filter_size, stride=1, padding=0):
        """
        Inicializa la capa convolucional.
        
        Args:
            num_filters (int): Número de filtros a aplicar (profundidad de salida)
            filter_size (int): Tamaño del kernel de convolución (filter_size x filter_size)
            stride (int, optional): Paso del filtro. Default: 1
            padding (int, optional): Cantidad de padding. Default: 0
        """
        super().__init__()
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        
    def initialize_params(self, input_shape):
        """
        Inicializa los parámetros de la capa usando Xavier/Glorot initialization.
        
        Esta inicialización ayuda a mantener la varianza de las activaciones
        estable a través de las capas, evitando el problema de gradientes
        que desaparecen o explotan.
        
        Args:
            input_shape (tuple): Forma de entrada (channels, height, width)
        """
        channels, height, width = input_shape
        
        # Xavier/Glorot initialization: mantiene varianza estable
        # Factor de escala basado en el número de conexiones de entrada
        fan_in = channels * self.filter_size * self.filter_size
        xavier_scale = np.sqrt(2.0 / fan_in)
        
        # Pesos: (num_filters, input_channels, filter_height, filter_width)
        self.params['W'] = np.random.randn(
            self.num_filters, channels, self.filter_size, self.filter_size
        ) * xavier_scale
        
        # Sesgos: uno por filtro, inicializados en cero
        self.params['b'] = np.zeros((self.num_filters, 1))
        
    def forward(self, inputs):
        """
        Propagación hacia adelante - aplica la operación de convolución.
        
        La convolución se realiza deslizando cada filtro sobre la imagen de entrada,
        calculando el producto punto entre el filtro y cada región local de la imagen.
        Esta implementación está optimizada usando vectorización de NumPy.
        
        Proceso matemático:
        - Para cada posición (i,j) en la salida:
          output[n,f,i,j] = Σ(input[n,c,i*stride:i*stride+filter_size,j*stride:j*stride+filter_size] 
                              * W[f,c,:,:]) + b[f]
        
        Args:
            inputs (np.ndarray): Tensor de entrada de forma (batch_size, channels, height, width)
            
        Returns:
            np.ndarray: Tensor de salida de forma (batch_size, num_filters, out_height, out_width)
        """
        # Guardamos la entrada para el backward pass
        self.cache['inputs'] = inputs
        batch_size, channels, height, width = inputs.shape
        
        # Aplicar padding si es necesario
        # El padding añade bordes de ceros para controlar el tamaño de salida
        if self.padding > 0:
            inputs_padded = np.pad(
                inputs, 
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
                mode='constant', constant_values=0
            )
        else:
            inputs_padded = inputs
            
        # Calcular dimensiones de salida usando la fórmula estándar
        # out_size = (input_size + 2*padding - filter_size) / stride + 1
        out_height = (height + 2 * self.padding - self.filter_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.filter_size) // self.stride + 1
        
        # Inicializar tensor de salida
        output = np.zeros((batch_size, self.num_filters, out_height, out_width))
        
        # OPTIMIZACIÓN SEGURA: Convolución con vectorización parcial
        # Elimina bucle de filtros pero mantiene bucles espaciales para estabilidad
        
        for i in range(out_height):
            for j in range(out_width):
                # Calcular posiciones en la imagen padded
                h_start = i * self.stride
                h_end = h_start + self.filter_size
                w_start = j * self.stride
                w_end = w_start + self.filter_size
                
                # Extraer región de la imagen para todos los ejemplos del batch
                # Forma: (batch_size, channels, filter_size, filter_size)
                input_slice = inputs_padded[:, :, h_start:h_end, w_start:w_end]
                
                # VECTORIZACIÓN SEGURA: Todos los filtros en una operación
                # Reshape para multiplicación matricial
                input_flat = input_slice.reshape(batch_size, -1)  # (batch_size, channels*filter_size*filter_size)
                filters_flat = self.params['W'].reshape(self.num_filters, -1)  # (num_filters, channels*filter_size*filter_size)
                
                # Multiplicación matricial vectorizada + sesgo
                # (batch_size, channels*h*w) @ (channels*h*w, num_filters) = (batch_size, num_filters)
                conv_result = np.dot(input_flat, filters_flat.T)  # (batch_size, num_filters)
                output[:, :, i, j] = conv_result + self.params['b'].flatten()  # Aplanar sesgo para broadcasting
                    
        return output
        
    def backward(self, grad_output):
        """
        Propagación hacia atrás - calcula gradientes para parámetros y entrada.
        
        Esta implementación está OPTIMIZADA para evitar los bucles anidados
        excesivos que causan lentitud en otras implementaciones.
        
        Calcula tres tipos de gradientes:
        1. ∂L/∂W: Gradiente respecto a los pesos (filtros)
        2. ∂L/∂b: Gradiente respecto a los sesgos  
        3. ∂L/∂input: Gradiente respecto a la entrada (para propagar hacia atrás)
        
        Args:
            grad_output (np.ndarray): Gradientes de la capa siguiente
            
        Returns:
            np.ndarray: Gradientes que se propagan a la capa anterior
        """
        inputs = self.cache['inputs']
        batch_size, channels, height, width = inputs.shape
        
        # Inicializar gradientes
        grad_input = np.zeros_like(inputs)
        self.gradients['W'] = np.zeros_like(self.params['W'])
        self.gradients['b'] = np.zeros_like(self.params['b'])
        
        # Aplicar padding a las entradas para cálculos
        if self.padding > 0:
            inputs_padded = np.pad(
                inputs, 
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
                mode='constant'
            )
            grad_input_padded = np.pad(
                grad_input, 
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
                mode='constant'
            )
        else:
            inputs_padded = inputs
            grad_input_padded = grad_input
            
        out_height, out_width = grad_output.shape[2], grad_output.shape[3]
        
        # OPTIMIZACIÓN SEGURA: Backward pass con vectorización parcial
        # Mantiene bucles espaciales pero vectoriza operaciones de filtros
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.filter_size
                w_start = j * self.stride
                w_end = w_start + self.filter_size
                
                # Extraer slice de entrada para esta posición
                input_slice = inputs_padded[:, :, h_start:h_end, w_start:w_end]
                # Forma: (batch_size, channels, filter_size, filter_size)
                
                # Gradiente del output en esta posición para todos los filtros
                grad_out_ij = grad_output[:, :, i, j]  # (batch_size, num_filters)
                
                # VECTORIZACIÓN: Procesar todos los filtros simultáneamente
                
                # 1. Gradientes de pesos (vectorizado)
                input_flat = input_slice.reshape(batch_size, -1)  # (batch_size, channels*h*w)
                # grad_out_ij: (batch_size, num_filters)
                # Queremos: (num_filters, channels*h*w)
                grad_w_contrib = np.dot(grad_out_ij.T, input_flat)  # (num_filters, channels*h*w)
                grad_w_contrib = grad_w_contrib.reshape(self.num_filters, channels, self.filter_size, self.filter_size)
                self.gradients['W'] += grad_w_contrib
                
                # 2. Gradientes de sesgos (vectorizado)
                bias_grad = np.sum(grad_out_ij, axis=0)  # (num_filters,)
                self.gradients['b'] += bias_grad.reshape(-1, 1)  # Reshape para coincidir con forma de sesgo
                
                # 3. Gradientes de entrada (vectorizado)
                filters_flat = self.params['W'].reshape(self.num_filters, -1)  # (num_filters, channels*h*w)
                # grad_out_ij: (batch_size, num_filters)
                # Queremos: (batch_size, channels*h*w)
                grad_input_contrib = np.dot(grad_out_ij, filters_flat)  # (batch_size, channels*h*w)
                grad_input_contrib = grad_input_contrib.reshape(batch_size, channels, self.filter_size, self.filter_size)
                
                # Acumular en la posición correspondiente
                grad_input_padded[:, :, h_start:h_end, w_start:w_end] += grad_input_contrib
        
        # Ajustar tamaño del gradiente de entrada
        target_height, target_width = inputs.shape[2], inputs.shape[3]
        current_height, current_width = grad_input_padded.shape[2], grad_input_padded.shape[3]
        
        # Recortar o ajustar según sea necesario
        if current_height >= target_height and current_width >= target_width:
            h_start = (current_height - target_height) // 2
            w_start = (current_width - target_width) // 2
            grad_input = grad_input_padded[:, :, h_start:h_start+target_height, w_start:w_start+target_width]
        else:
            grad_input = grad_input_padded
            
        return grad_input
    
    def get_output_shape(self, input_shape):
        """
        Calcula la forma de salida de la capa convolucional.
        
        Args:
            input_shape (tuple): Forma de entrada (channels, height, width)
            
        Returns:
            tuple: Forma de salida (num_filters, out_height, out_width)
        """
        channels, height, width = input_shape
        out_height = (height + 2 * self.padding - self.filter_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.filter_size) // self.stride + 1
        return (self.num_filters, out_height, out_width)