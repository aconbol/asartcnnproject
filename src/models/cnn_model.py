import numpy as np
from layers.conv_layer import Conv2DLayer
from layers.pooling_layer import MaxPool2DLayer, GlobalAveragePool2DLayer
from layers.dense_layer import DenseLayer
from layers.activation_layer import ActivationLayer
from layers.flatten_layer import FlattenLayer

class CNNModel:
    """
    Modelo de Red Neuronal Convolucional optimizado para clasificación de imágenes.
    
    Esta clase implementa una CNN completa que puede ser construida de manera modular
    añadiendo capas secuencialmente. Incluye funcionalidades de compilación automática,
    propagación hacia adelante y hacia atrás, y predicción.
    
    La arquitectura se construye siguiendo el patrón:
    Input -> Conv+Activation+Pooling -> ... -> Flatten/GAP -> Dense+Activation -> Output
    
    Attributes:
        layers (list): Lista de capas que componen el modelo
        compiled (bool): Indica si el modelo ha sido compilado (parámetros inicializados)
    """
    
    def __init__(self):
        """Inicializa un modelo CNN vacío."""
        self.layers = []
        self.compiled = False
        
    def add_layer(self, layer):
        """
        Añade una capa al modelo.
        
        Las capas se procesan secuencialmente en el orden que se añaden.
        
        Args:
            layer: Instancia de cualquier capa que herede de BaseLayer
        """
        self.layers.append(layer)
        
    def compile(self, input_shape):
        """
        Compila el modelo inicializando los parámetros de todas las capas.
        
        Este proceso es crucial porque:
        1. Calcula las formas de salida de cada capa secuencialmente
        2. Inicializa los parámetros (pesos y sesgos) de capas entrenables
        3. Verifica compatibilidad dimensional entre capas consecutivas
        
        Args:
            input_shape (tuple): Forma de entrada al modelo (channels, height, width)
        """
        current_shape = input_shape
        
        for i, layer in enumerate(self.layers):
            # Inicializar parámetros si la capa los requiere
            if hasattr(layer, 'initialize_params'):
                # Para capas densas, necesitamos el tamaño aplanado
                if isinstance(layer, DenseLayer):
                    if len(current_shape) > 1:
                        # Aplanar automáticamente si viene de capas conv
                        input_size = np.prod(current_shape)
                    else:
                        input_size = current_shape[0]
                    
                    layer.initialize_params(int(input_size))
                else:
                    # Para capas conv/pool, usar la forma actual
                    layer.initialize_params(current_shape)
            
            # Calcular forma de salida para la siguiente capa
            if hasattr(layer, 'get_output_shape'):
                current_shape = layer.get_output_shape(current_shape)
            else:
                # Fallback para capas sin get_output_shape definido
                current_shape = self._calculate_output_shape(layer, current_shape)
        
        self.compiled = True
        print(f"Modelo compilado exitosamente con {len(self.layers)} capas")
        
    def _calculate_output_shape(self, layer, input_shape):
        """
        Calcula la forma de salida para capas que no implementan get_output_shape.
        
        Args:
            layer: Capa para la cual calcular la forma de salida
            input_shape: Forma de entrada a la capa
            
        Returns:
            tuple: Forma de salida de la capa
        """
        if isinstance(layer, Conv2DLayer):
            channels, height, width = input_shape
            out_height = (height + 2 * layer.padding - layer.filter_size) // layer.stride + 1
            out_width = (width + 2 * layer.padding - layer.filter_size) // layer.stride + 1
            return (layer.num_filters, out_height, out_width)
            
        elif isinstance(layer, MaxPool2DLayer):
            channels, height, width = input_shape
            out_height = (height - layer.pool_size) // layer.stride + 1
            out_width = (width - layer.pool_size) // layer.stride + 1
            return (channels, out_height, out_width)
            
        elif isinstance(layer, GlobalAveragePool2DLayer):
            channels, height, width = input_shape
            return (channels,)
            
        elif isinstance(layer, FlattenLayer):
            return (np.prod(input_shape),)
            
        elif isinstance(layer, DenseLayer):
            return (layer.units,)
            
        else:
            # Para capas de activación y otras, mantener la misma forma
            return input_shape
        
    def forward(self, inputs):
        """
        Propagación hacia adelante a través de todas las capas.
        
        Procesa la entrada secuencialmente a través de cada capa,
        donde la salida de una capa se convierte en la entrada de la siguiente.
        
        Args:
            inputs (np.ndarray): Tensor de entrada al modelo
                                Forma: (batch_size, channels, height, width)
                                
        Returns:
            np.ndarray: Salida del modelo (logits o probabilidades)
                       Forma: (batch_size, num_classes)
        """
        if not self.compiled:
            raise RuntimeError("El modelo debe ser compilado antes de usarse. Llame a model.compile(input_shape)")
        
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x
        
    def backward(self, grad_output):
        """
        Propagación hacia atrás a través de todas las capas.
        
        Calcula gradientes propagándolos desde la salida hacia la entrada,
        permitiendo que cada capa calcule sus gradientes de parámetros.
        
        Args:
            grad_output (np.ndarray): Gradientes de la función de pérdida
                                     
        Returns:
            np.ndarray: Gradientes respecto a la entrada del modelo
        """
        grad = grad_output
        # Procesar capas en orden inverso
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
        
    def predict(self, inputs):
        """
        Realiza predicciones devolviendo las clases predichas.
        
        Args:
            inputs (np.ndarray): Datos de entrada para predicción
                                
        Returns:
            np.ndarray: Clases predichas (índices de clase)
                       Forma: (batch_size,)
        """
        outputs = self.forward(inputs)
        return np.argmax(outputs, axis=1)
        
    def predict_proba(self, inputs):
        """
        Realiza predicciones devolviendo probabilidades de clase.
        
        Aplica softmax a la salida para obtener una distribución de probabilidad
        sobre las clases, incluso si el modelo no tiene softmax en la última capa.
        
        Args:
            inputs (np.ndarray): Datos de entrada para predicción
                                
        Returns:
            np.ndarray: Probabilidades de clase
                       Forma: (batch_size, num_classes)
        """
        outputs = self.forward(inputs)
        
        # Aplicar softmax para obtener probabilidades
        # Estabilidad numérica: restar el máximo
        exp_outputs = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
        probabilities = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)
        
        return probabilities
        
    def get_params(self):
        """
        Obtiene todos los parámetros del modelo.
        
        Útil para guardar el modelo entrenado o para análisis de parámetros.
        
        Returns:
            dict: Diccionario con los parámetros de todas las capas
        """
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params') and layer.params:
                params[f'layer_{i}'] = layer.get_params()
        return params
        
    def set_params(self, params):
        """
        Establece los parámetros del modelo.
        
        Útil para cargar un modelo previamente entrenado.
        
        Args:
            params (dict): Diccionario con parámetros de las capas
        """
        for i, layer in enumerate(self.layers):
            layer_key = f'layer_{i}'
            if layer_key in params and hasattr(layer, 'params'):
                layer.params = params[layer_key]
        
    def summary(self):
        """
        Imprime un resumen de la arquitectura del modelo.
        
        Muestra información detallada sobre cada capa, incluyendo
        formas de salida y número de parámetros.
        """
        print("=" * 60)
        print("RESUMEN DEL MODELO CNN")
        print("=" * 60)
        
        total_params = 0
        
        for i, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            
            # Contar parámetros
            layer_params = 0
            if hasattr(layer, 'params'):
                for param_name, param_value in layer.params.items():
                    if param_value is not None:
                        layer_params += np.prod(param_value.shape)
            
            total_params += layer_params
            
            # Información adicional según el tipo de capa
            extra_info = ""
            if isinstance(layer, Conv2DLayer):
                extra_info = f"({layer.num_filters} filtros, {layer.filter_size}x{layer.filter_size})"
            elif isinstance(layer, MaxPool2DLayer):
                extra_info = f"({layer.pool_size}x{layer.pool_size})"
            elif isinstance(layer, DenseLayer):
                extra_info = f"({layer.units} unidades)"
            elif isinstance(layer, ActivationLayer):
                extra_info = f"({layer.activation})"
            
            print(f"Capa {i+1:2d}: {layer_name:<20} {extra_info:<20} Parámetros: {layer_params:,}")
        
        print("=" * 60)
        print(f"TOTAL DE PARÁMETROS: {total_params:,}")
        print("=" * 60)


def create_retina_cnn(input_shape, num_classes=5):
    """
    Crea un modelo CNN optimizado para clasificación de retinopatía diabética.
    
    Esta arquitectura está diseñada específicamente para imágenes médicas de retina,
    con una estructura que balancea capacidad de extracción de características
    con eficiencia computacional.
    
    Arquitectura:
    - 2 bloques convolucionales ligeros (16, 32 filtros)
    - Max pooling para reducción dimensional
    - Flatten para transición a capas densas
    - Capa densa intermedia (128 unidades)
    - Capa de salida con softmax
    
    Esta arquitectura reducida es más eficiente que versiones más profundas
    mientras mantiene capacidad de aprendizaje adecuada.
    
    Args:
        input_shape (tuple): Forma de entrada (channels, height, width)
        num_classes (int, optional): Número de clases a clasificar. Default: 5
        
    Returns:
        CNNModel: Modelo CNN compilado y listo para entrenar
    """
    model = CNNModel()
    
    # ====== BLOQUE CONVOLUCIONAL 1 ======
    # Filtros: 16, Tamaño: 3x3, Padding: 1 (mantiene dimensiones)
    model.add_layer(Conv2DLayer(num_filters=16, filter_size=3, padding=1))
    model.add_layer(ActivationLayer('relu'))  # Introducir no-linealidad
    model.add_layer(MaxPool2DLayer(pool_size=2, stride=2))  # Reducir dimensiones 2x
    
    # ====== BLOQUE CONVOLUCIONAL 2 ======
    # Filtros: 32, Tamaño: 3x3, Padding: 1
    model.add_layer(Conv2DLayer(num_filters=32, filter_size=3, padding=1))
    model.add_layer(ActivationLayer('relu'))
    model.add_layer(MaxPool2DLayer(pool_size=2, stride=2))  # Reducir dimensiones 2x
    
    # ====== TRANSICIÓN A CAPAS DENSAS ======
    # Aplanar características extraídas por capas convolucionales
    model.add_layer(FlattenLayer())
    
    # ====== CAPAS DENSAS ======
    # Capa intermedia para aprendizaje de patrones complejos
    model.add_layer(DenseLayer(units=128))
    model.add_layer(ActivationLayer('relu'))
    
    # Capa de salida para clasificación
    model.add_layer(DenseLayer(units=num_classes))
    model.add_layer(ActivationLayer('softmax'))  # Probabilidades de clase
    
    # ====== COMPILACIÓN ======
    # Inicializar parámetros y verificar compatibilidad dimensional
    model.compile(input_shape)
    
    print("🧠 Modelo CNN para Retinopatía Diabética creado exitosamente")
    print(f"   📊 Entrada: {input_shape}")
    print(f"   🎯 Clases: {num_classes}")
    print(f"   ⚡ Arquitectura optimizada para velocidad y precisión")
    
    return model