import numpy as np

class LossFunction:
    """
    Clase que implementa funciones de pérdida para entrenamiento de redes neuronales.
    
    Las funciones de pérdida miden qué tan bien se desempeña el modelo comparando
    las predicciones con las etiquetas verdaderas. Cada función de pérdida incluye
    tanto el cálculo de la pérdida como su gradiente para la retropropagación.
    
    Todas las implementaciones están optimizadas para estabilidad numérica y eficiencia.
    """
    
    def __init__(self, loss_type='categorical_crossentropy'):
        """
        Inicializa la función de pérdida.
        
        Args:
            loss_type (str): Tipo de función de pérdida a usar
        """
        self.loss_type = loss_type.lower()
        
        if self.loss_type == 'categorical_crossentropy':
            self.loss_func = self.categorical_crossentropy
        elif self.loss_type == 'binary_crossentropy':
            self.loss_func = self.binary_crossentropy
        elif self.loss_type == 'mse':
            self.loss_func = self.mean_squared_error
        else:
            raise ValueError(f"Tipo de función de pérdida no soportado: {loss_type}")
    
    def __call__(self, y_true, y_pred):
        """
        Permite usar la instancia como función.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones del modelo
            
        Returns:
            tuple: (loss, gradient)
        """
        return self.loss_func(y_true, y_pred)
    
    @staticmethod
    def categorical_crossentropy(y_true, y_pred):
        """
        Entropía Cruzada Categórica para clasificación multiclase.
        
        Esta es la función de pérdida estándar para problemas de clasificación
        con múltiples clases mutuamente excluyentes (como las 5 clases de
        retinopatía diabética).
        
        Fórmula matemática:
        L = -(1/N) * Σᵢ Σⱼ yᵢⱼ * log(ŷᵢⱼ)
        
        donde:
        - N: tamaño del batch
        - yᵢⱼ: etiqueta verdadera (1 si clase j es correcta para muestra i, 0 sino)
        - ŷᵢⱼ: probabilidad predicha para clase j en muestra i
        
        Args:
            y_true (np.ndarray): Etiquetas verdaderas. Puede ser:
                                - Índices de clase: (batch_size,) con valores 0,1,2,...
                                - One-hot: (batch_size, num_classes) con 0s y 1s
            y_pred (np.ndarray): Probabilidades predichas (batch_size, num_classes)
                                Deben sumar 1 para cada muestra (salida de softmax)
        
        Returns:
            tuple: (loss, gradient)
                - loss (float): Valor escalar de la pérdida promediada sobre el batch
                - gradient (np.ndarray): Gradiente respecto a y_pred para retropropagación
        """
        batch_size = y_pred.shape[0]
        
        # Convertir índices de clase a codificación one-hot si es necesario
        if y_true.ndim == 1:
            # y_true son índices de clase: [0, 2, 1, 4, ...]
            y_true_onehot = np.zeros_like(y_pred)
            y_true_onehot[np.arange(batch_size), y_true] = 1
        else:
            # y_true ya está en formato one-hot
            y_true_onehot = y_true
            
        # ESTABILIDAD NUMÉRICA: Limitar predicciones para evitar log(0)
        # log(0) = -∞, lo que causaría NaN en los cálculos
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Calcular pérdida: -Σ(y_true * log(y_pred)) / batch_size
        # Solo las clases verdaderas (y_true_onehot = 1) contribuyen a la pérdida
        loss = -np.sum(y_true_onehot * np.log(y_pred_clipped)) / batch_size
        
        # Calcular gradiente: ∂L/∂y_pred = -(y_true / y_pred) / batch_size
        # Este gradiente se propaga hacia atrás para actualizar los pesos
        gradient = -(y_true_onehot / y_pred_clipped) / batch_size
        
        return loss, gradient
    
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """
        Error Cuadrático Medio para problemas de regresión.
        
        MSE penaliza errores grandes más que errores pequeños debido al término
        cuadrático. Es útil cuando queremos que las predicciones estén muy
        cerca de los valores verdaderos.
        
        Fórmula matemática:
        L = (1/2N) * Σᵢ (yᵢ - ŷᵢ)²
        
        Args:
            y_true (np.ndarray): Valores verdaderos
            y_pred (np.ndarray): Valores predichos (misma forma que y_true)
        
        Returns:
            tuple: (loss, gradient)
                - loss (float): MSE promediado sobre el batch
                - gradient (np.ndarray): Gradiente respecto a y_pred
        """
        batch_size = y_pred.shape[0]
        
        # Calcular diferencia entre predicción y valor verdadero
        diff = y_pred - y_true
        
        # Calcular pérdida: promedio de diferencias al cuadrado
        # Factor 1/2 para simplificar el gradiente
        loss = np.sum(diff ** 2) / (2 * batch_size)
        
        # Calcular gradiente: ∂L/∂y_pred = (y_pred - y_true) / batch_size
        gradient = diff / batch_size
        
        return loss, gradient
    
    @staticmethod
    def binary_crossentropy(y_true, y_pred):
        """
        Entropía Cruzada Binaria para clasificación binaria.
        
        Función de pérdida estándar para problemas de clasificación con solo
        dos clases (por ejemplo: tiene retinopatía / no tiene retinopatía).
        
        Fórmula matemática:
        L = -(1/N) * Σᵢ [yᵢ*log(ŷᵢ) + (1-yᵢ)*log(1-ŷᵢ)]
        
        Args:
            y_true (np.ndarray): Etiquetas verdaderas (0 o 1)
            y_pred (np.ndarray): Probabilidades predichas (entre 0 y 1)
        
        Returns:
            tuple: (loss, gradient)
                - loss (float): BCE promediado sobre el batch
                - gradient (np.ndarray): Gradiente respecto a y_pred
        """
        batch_size = y_pred.shape[0]
        
        # ESTABILIDAD NUMÉRICA: Limitar predicciones para evitar log(0)
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Calcular pérdida binaria de entropía cruzada
        # Términos: y*log(ŷ) + (1-y)*log(1-ŷ)
        loss = -np.sum(
            y_true * np.log(y_pred_clipped) + 
            (1 - y_true) * np.log(1 - y_pred_clipped)
        ) / batch_size
        
        # Calcular gradiente
        # ∂L/∂y_pred = -(y/ŷ - (1-y)/(1-ŷ)) / batch_size
        gradient = -(
            y_true / y_pred_clipped - 
            (1 - y_true) / (1 - y_pred_clipped)
        ) / batch_size
        
        return loss, gradient
    
    @staticmethod
    def sparse_categorical_crossentropy(y_true, y_pred):
        """
        Entropía Cruzada Categórica Sparse.
        
        Versión más eficiente de categorical_crossentropy cuando las etiquetas
        verdaderas están como índices de clase en lugar de one-hot encoding.
        Evita la conversión a one-hot para mejorar eficiencia de memoria.
        
        Args:
            y_true (np.ndarray): Índices de clase (batch_size,)
            y_pred (np.ndarray): Probabilidades predichas (batch_size, num_classes)
        
        Returns:
            tuple: (loss, gradient)
        """
        batch_size = y_pred.shape[0]
        
        # ESTABILIDAD NUMÉRICA: Limitar predicciones
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Calcular pérdida usando indexación avanzada de NumPy
        # Más eficiente que crear matriz one-hot
        loss = -np.sum(np.log(y_pred_clipped[np.arange(batch_size), y_true])) / batch_size
        
        # Calcular gradiente sparse
        gradient = y_pred_clipped.copy()
        gradient[np.arange(batch_size), y_true] -= 1
        gradient /= batch_size
        
        return loss, gradient
    
    @staticmethod
    def huber_loss(y_true, y_pred, delta=1.0):
        """
        Pérdida de Huber - combina MSE y MAE.
        
        Pérdida robusta que es cuadrática para errores pequeños y lineal
        para errores grandes. Útil cuando hay outliers en los datos.
        
        Args:
            y_true (np.ndarray): Valores verdaderos
            y_pred (np.ndarray): Valores predichos
            delta (float): Punto de transición entre cuadrático y lineal
        
        Returns:
            tuple: (loss, gradient)
        """
        batch_size = y_pred.shape[0]
        diff = y_pred - y_true
        abs_diff = np.abs(diff)
        
        # Pérdida cuadrática para |diff| <= delta, lineal para |diff| > delta
        quadratic = 0.5 * diff ** 2
        linear = delta * (abs_diff - 0.5 * delta)
        
        loss = np.mean(np.where(abs_diff <= delta, quadratic, linear))
        
        # Gradiente correspondiente
        gradient = np.where(
            abs_diff <= delta,
            diff,
            delta * np.sign(diff)
        ) / batch_size
        
        return loss, gradient