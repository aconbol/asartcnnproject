import numpy as np

class Optimizer:
    """
    Factory class para crear optimizadores de redes neuronales.
    
    Los optimizadores actualizan los parámetros del modelo basándose en los gradientes
    calculados durante la retropropagación. Diferentes algoritmos de optimización
    implementan estrategias distintas para acelerar la convergencia y mejorar la estabilidad.
    """
    
    def __init__(self, learning_rate=0.001, optimizer_type='adam', **kwargs):
        """
        Inicializa el optimizador según el tipo especificado.
        
        Args:
            learning_rate (float): Tasa de aprendizaje
            optimizer_type (str): Tipo de optimizador ('sgd', 'adam', 'rmsprop')
            **kwargs: Argumentos adicionales específicos del optimizador
        """
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type.lower()
        
        if self.optimizer_type == 'sgd':
            momentum = kwargs.get('momentum', 0.0)
            self.optimizer = self.SGD(learning_rate, momentum)
        elif self.optimizer_type == 'adam':
            beta1 = kwargs.get('beta1', 0.9)
            beta2 = kwargs.get('beta2', 0.999)
            epsilon = kwargs.get('epsilon', 1e-8)
            self.optimizer = self.Adam(learning_rate, beta1, beta2, epsilon)
        elif self.optimizer_type == 'rmsprop':
            beta = kwargs.get('beta', 0.9)
            epsilon = kwargs.get('epsilon', 1e-8)
            self.optimizer = self.RMSprop(learning_rate, beta, epsilon)
        else:
            raise ValueError(f"Tipo de optimizador no soportado: {optimizer_type}")
    
    def update(self, layer, layer_id):
        """
        Delega la actualización al optimizador específico.
        
        Args:
            layer: Capa con parámetros y gradientes a actualizar
            layer_id: Identificador único de la capa
        """
        return self.optimizer.update(layer, layer_id)
    
    class SGD:
        """
        Optimizador Stochastic Gradient Descent (SGD) con momentum.
        
        SGD es el algoritmo de optimización más básico que actualiza parámetros
        moviéndose en dirección opuesta al gradiente. El momentum ayuda a acelerar
        la convergencia y reduce oscilaciones.
        
        Ecuaciones:
        v_t = β * v_{t-1} + η * ∇θ
        θ_t = θ_{t-1} - v_t
        
        donde:
        - v_t: velocidad (momentum) en el tiempo t
        - β: coeficiente de momentum
        - η: tasa de aprendizaje
        - ∇θ: gradiente respecto a parámetros
        """
        
        def __init__(self, learning_rate=0.01, momentum=0.0):
            """
            Inicializa el optimizador SGD.
            
            Args:
                learning_rate (float): Tasa de aprendizaje (η)
                momentum (float): Coeficiente de momentum (β), rango [0, 1)
            """
            self.learning_rate = learning_rate
            self.momentum = momentum
            self.velocity = {}  # Almacena velocidades para cada parámetro
            
        def update(self, layer, layer_id):
            """
            Actualiza los parámetros de una capa usando SGD con momentum.
            
            Args:
                layer: Capa con parámetros y gradientes a actualizar
                layer_id: Identificador único de la capa
            """
            if hasattr(layer, 'params') and hasattr(layer, 'gradients'):
                for param_name, param in layer.params.items():
                    if param_name in layer.gradients:
                        grad = layer.gradients[param_name]
                        
                        # Clave única para este parámetro
                        velocity_key = f'{layer_id}_{param_name}'
                        
                        # Inicializar velocidad si es la primera vez
                        if velocity_key not in self.velocity:
                            self.velocity[velocity_key] = np.zeros_like(param)
                        
                        # Actualizar velocidad con momentum y gradiente
                        self.velocity[velocity_key] = (
                            self.momentum * self.velocity[velocity_key] - 
                            self.learning_rate * grad
                        )
                        
                        # Actualizar parámetros
                        layer.params[param_name] += self.velocity[velocity_key]
    
    class Adam:
        """
        Optimizador Adam (Adaptive Moment Estimation).
        
        Adam combina las ventajas de AdaGrad y RMSprop, manteniendo tanto un promedio
        móvil exponencial de gradientes (primer momento) como de gradientes al cuadrado
        (segundo momento). Es muy efectivo para la mayoría de problemas de deep learning.
        
        Ecuaciones:
        m_t = β₁ * m_{t-1} + (1 - β₁) * ∇θ
        v_t = β₂ * v_{t-1} + (1 - β₂) * (∇θ)²
        m̂_t = m_t / (1 - β₁ᵗ)  # Corrección de sesgo
        v̂_t = v_t / (1 - β₂ᵗ)  # Corrección de sesgo
        θ_t = θ_{t-1} - η * m̂_t / (√v̂_t + ε)
        
        donde:
        - m_t: primer momento (promedio de gradientes)
        - v_t: segundo momento (promedio de gradientes al cuadrado)
        - β₁, β₂: factores de decaimiento exponencial
        - ε: término pequeño para estabilidad numérica
        """
        
        def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
            """
            Inicializa el optimizador Adam.
            
            Args:
                learning_rate (float): Tasa de aprendizaje (η)
                beta1 (float): Factor de decaimiento para primer momento
                beta2 (float): Factor de decaimiento para segundo momento  
                epsilon (float): Término para estabilidad numérica
            """
            self.learning_rate = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.m = {}  # Primer momento (promedio de gradientes)
            self.v = {}  # Segundo momento (promedio de gradientes²)
            self.t = {}  # Contador de pasos por parámetro
            
        def update(self, layer, layer_id):
            """
            Actualiza los parámetros de una capa usando el algoritmo Adam.
            
            El algoritmo Adam es particularmente efectivo porque:
            1. Adapta la tasa de aprendizaje individualmente para cada parámetro
            2. Maneja gradientes sparse de manera eficiente
            3. Es robusto ante la elección de hiperparámetros
            
            Args:
                layer: Capa con parámetros y gradientes a actualizar
                layer_id: Identificador único de la capa
            """
            if hasattr(layer, 'params') and hasattr(layer, 'gradients'):
                for param_name, param in layer.params.items():
                    if param_name in layer.gradients:
                        grad = layer.gradients[param_name]
                        
                        # Clave única para este parámetro
                        param_key = f'{layer_id}_{param_name}'
                        
                        # Inicializar momentos y contador si es la primera vez
                        if param_key not in self.m:
                            self.m[param_key] = np.zeros_like(param)
                            self.v[param_key] = np.zeros_like(param)
                            self.t[param_key] = 0
                        
                        # Incrementar contador de pasos para este parámetro
                        self.t[param_key] += 1
                        
                        # Actualizar primer momento (promedio móvil de gradientes)
                        # m_t = β₁ * m_{t-1} + (1 - β₁) * ∇θ
                        self.m[param_key] = (
                            self.beta1 * self.m[param_key] + 
                            (1 - self.beta1) * grad
                        )
                        
                        # Actualizar segundo momento (promedio móvil de gradientes²)
                        # v_t = β₂ * v_{t-1} + (1 - β₂) * (∇θ)²
                        self.v[param_key] = (
                            self.beta2 * self.v[param_key] + 
                            (1 - self.beta2) * (grad ** 2)
                        )
                        
                        # Corrección de sesgo para primer momento
                        # m̂_t = m_t / (1 - β₁ᵗ)
                        m_corrected = self.m[param_key] / (1 - self.beta1 ** self.t[param_key])
                        
                        # Corrección de sesgo para segundo momento
                        # v̂_t = v_t / (1 - β₂ᵗ)
                        v_corrected = self.v[param_key] / (1 - self.beta2 ** self.t[param_key])
                        
                        # Actualizar parámetros
                        # θ_t = θ_{t-1} - η * m̂_t / (√v̂_t + ε)
                        update_step = (
                            self.learning_rate * m_corrected / 
                            (np.sqrt(v_corrected) + self.epsilon)
                        )
                        
                        layer.params[param_name] -= update_step
    
    class RMSprop:
        """
        Optimizador RMSprop (Root Mean Square Propagation).
        
        RMSprop adapta la tasa de aprendizaje dividiendo por un promedio móvil
        exponencial de los gradientes al cuadrado. Esto ayuda a resolver el problema
        de tasas de aprendizaje que decrecen demasiado rápido en AdaGrad.
        
        Ecuaciones:
        v_t = β * v_{t-1} + (1 - β) * (∇θ)²
        θ_t = θ_{t-1} - η * ∇θ / (√v_t + ε)
        """
        
        def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
            """
            Inicializa el optimizador RMSprop.
            
            Args:
                learning_rate (float): Tasa de aprendizaje
                beta (float): Factor de decaimiento para promedio móvil
                epsilon (float): Término para estabilidad numérica
            """
            self.learning_rate = learning_rate
            self.beta = beta
            self.epsilon = epsilon
            self.v = {}  # Promedio móvil de gradientes al cuadrado
            
        def update(self, layer, layer_id):
            """
            Actualiza los parámetros usando RMSprop.
            
            Args:
                layer: Capa con parámetros y gradientes a actualizar
                layer_id: Identificador único de la capa
            """
            if hasattr(layer, 'params') and hasattr(layer, 'gradients'):
                for param_name, param in layer.params.items():
                    if param_name in layer.gradients:
                        grad = layer.gradients[param_name]
                        
                        param_key = f'{layer_id}_{param_name}'
                        
                        # Inicializar acumulador si es la primera vez
                        if param_key not in self.v:
                            self.v[param_key] = np.zeros_like(param)
                        
                        # Actualizar promedio móvil de gradientes al cuadrado
                        self.v[param_key] = (
                            self.beta * self.v[param_key] + 
                            (1 - self.beta) * (grad ** 2)
                        )
                        
                        # Actualizar parámetros
                        update_step = (
                            self.learning_rate * grad / 
                            (np.sqrt(self.v[param_key]) + self.epsilon)
                        )
                        
                        layer.params[param_name] -= update_step