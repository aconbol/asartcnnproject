import numpy as np
import time
from tqdm import tqdm

class Trainer:
    """
    Clase entrenador optimizada para redes neuronales convolucionales.
    
    Coordina el proceso completo de entrenamiento incluyendo:
    - Ciclos de entrenamiento y validación por épocas
    - Propagación hacia adelante y hacia atrás
    - Actualización de parámetros usando el optimizador
    - Seguimiento de métricas y progreso
    - Evaluación del modelo en datos no vistos
    
    Esta implementación está optimizada para eficiencia y claridad académica.
    """
    
    def __init__(self, model, optimizer, loss_function, metrics=None):
        """
        Inicializa el entrenador con modelo, optimizador y función de pérdida.
        
        Args:
            model: Modelo CNN a entrenar
            optimizer: Algoritmo de optimización (SGD, Adam, etc.)
            loss_function: Función de pérdida (categorical_crossentropy, etc.)
            metrics: Objeto para calcular métricas de evaluación (opcional)
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        
        # Historial de entrenamiento para análisis y visualización
        self.history = {
            'train_loss': [],        # Pérdida en entrenamiento por época
            'train_accuracy': [],    # Precisión en entrenamiento por época
            'val_loss': [],          # Pérdida en validación por época
            'val_accuracy': []       # Precisión en validación por época
        }
        
    def calculate_accuracy(self, y_true, y_pred):
        """
        Calcula la precisión comparando predicciones con etiquetas verdaderas.
        
        Maneja tanto predicciones en formato softmax (probabilidades) como
        etiquetas en formato one-hot o índices de clase.
        
        Args:
            y_true: Etiquetas verdaderas (índices o one-hot)
            y_pred: Predicciones del modelo (probabilidades o logits)
            
        Returns:
            float: Precisión entre 0 y 1
        """
        # Convertir probabilidades a predicciones de clase
        predictions = np.argmax(y_pred, axis=1)
        
        # Convertir one-hot a índices de clase si es necesario
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
            
        # Calcular fracción de predicciones correctas
        return np.mean(predictions == y_true)
    
    def train_step(self, X, y):
        """
        Ejecuta un paso completo de entrenamiento sobre un lote.
        
        Este es el corazón del algoritmo de aprendizaje:
        1. Forward pass: calcular predicciones
        2. Calcular pérdida y gradientes
        3. Backward pass: propagar gradientes
        4. Actualizar parámetros
        
        Args:
            X (np.ndarray): Lote de imágenes de entrada
            y (np.ndarray): Etiquetas correspondientes
            
        Returns:
            tuple: (pérdida, precisión) para este lote
        """
        # PASO 1: Propagación hacia adelante
        # Las imágenes pasan por todas las capas del modelo
        y_pred = self.model.forward(X)
        
        # PASO 2: Calcular pérdida y gradientes iniciales
        # La función de pérdida mide qué tan mal están las predicciones
        loss, grad_loss = self.loss_function(y, y_pred)
        
        # PASO 3: Propagación hacia atrás
        # Los gradientes se propagan desde la salida hacia la entrada
        self.model.backward(grad_loss)
        
        # PASO 4: Actualizar parámetros
        # El optimizador ajusta pesos y sesgos basándose en los gradientes
        for i, layer in enumerate(self.model.layers):
            self.optimizer.update(layer, i)
        
        # PASO 5: Calcular métricas para monitoreo
        accuracy = self.calculate_accuracy(y, y_pred)
        
        return loss, accuracy
    
    def evaluate_step(self, X, y):
        """
        Ejecuta un paso de evaluación sin actualizar parámetros.
        
        Utilizado durante validación para medir rendimiento sin afectar
        el aprendizaje del modelo.
        
        Args:
            X (np.ndarray): Lote de imágenes de entrada
            y (np.ndarray): Etiquetas correspondientes
            
        Returns:
            tuple: (pérdida, precisión) para este lote
        """
        # Solo propagación hacia adelante (sin backward ni actualización)
        y_pred = self.model.forward(X)
        
        # Calcular pérdida (ignoramos los gradientes)
        loss, _ = self.loss_function(y, y_pred)
        
        # Calcular precisión
        accuracy = self.calculate_accuracy(y, y_pred)
        
        return loss, accuracy
    
    def train(self, train_dataset, val_dataset=None, epochs=10, verbose=True):
        """
        Ejecuta el ciclo completo de entrenamiento.
        
        Para cada época:
        1. Procesa todos los lotes de entrenamiento
        2. Actualiza parámetros del modelo
        3. Evalúa rendimiento en validación
        4. Registra métricas para análisis
        
        Args:
            train_dataset: Dataset de entrenamiento
            val_dataset: Dataset de validación (opcional)
            epochs (int): Número de épocas a entrenar
            verbose (bool): Si mostrar progreso detallado
        """
        print("Inicio del entrenamiento...")
        start_time = time.time()
        
        for epoch in range(epochs):
            if verbose:
                print(f"\nEpoch {epoch + 1}/{epochs}")
                print("-" * 30)
            
            # ========== FASE DE ENTRENAMIENTO ==========
            epoch_start = time.time()
            train_losses = []
            train_accuracies = []
            
            # Reiniciar dataset para nueva época
            train_dataset.reset()
            
            # Calcular número de lotes para la barra de progreso
            total_batches = len(train_dataset) // train_dataset.batch_size
            
            if verbose:
                pbar = tqdm(total=total_batches, desc="Training")
            
            # Procesar todos los lotes de entrenamiento
            for batch_idx in range(total_batches):
                # Obtener lote con aumento de datos
                X_batch, y_batch = train_dataset.get_batch(augment=True)
                if X_batch is None:
                    break
                
                # Ejecutar paso de entrenamiento
                loss, accuracy = self.train_step(X_batch, y_batch)
                
                # Registrar métricas del lote
                train_losses.append(loss)
                train_accuracies.append(accuracy)
                
                # Actualizar barra de progreso
                if verbose:
                    pbar.set_postfix({
                        'Loss': f'{loss:.4f}',
                        'Acc': f'{accuracy:.4f}'
                    })
                    pbar.update(1)
            
            if verbose:
                pbar.close()
            
            # Calcular métricas promedio de la época
            epoch_train_loss = np.mean(train_losses)
            epoch_train_acc = np.mean(train_accuracies)
            
            # Registrar en historial
            self.history['train_loss'].append(epoch_train_loss)
            self.history['train_accuracy'].append(epoch_train_acc)
            
            # ========== FASE DE VALIDACIÓN ==========
            if val_dataset is not None:
                val_losses = []
                val_accuracies = []
                
                # Reiniciar dataset de validación
                val_dataset.reset()
                
                val_total_batches = len(val_dataset) // val_dataset.batch_size
                
                if verbose:
                    val_pbar = tqdm(total=val_total_batches, desc="Validation")
                
                # Procesar todos los lotes de validación
                for val_batch_idx in range(val_total_batches):
                    # Obtener lote sin aumento de datos
                    X_val, y_val = val_dataset.get_batch(augment=False)
                    if X_val is None:
                        break
                    
                    # Ejecutar paso de evaluación (sin actualizar parámetros)
                    val_loss, val_accuracy = self.evaluate_step(X_val, y_val)
                    
                    # Registrar métricas del lote
                    val_losses.append(val_loss)
                    val_accuracies.append(val_accuracy)
                    
                    # Actualizar barra de progreso
                    if verbose:
                        val_pbar.set_postfix({
                            'Val Loss': f'{val_loss:.4f}',
                            'Val Acc': f'{val_accuracy:.4f}'
                        })
                        val_pbar.update(1)
                
                if verbose:
                    val_pbar.close()
                
                # Calcular métricas promedio de validación
                epoch_val_loss = np.mean(val_losses)
                epoch_val_acc = np.mean(val_accuracies)
                
                # Registrar en historial
                self.history['val_loss'].append(epoch_val_loss)
                self.history['val_accuracy'].append(epoch_val_acc)
                
                # Mostrar resumen de la época
                if verbose:
                    epoch_time = time.time() - epoch_start
                    print(f"\nTrain Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
                    print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
                    print(f"Tiempo de época: {epoch_time:.1f}s")
            else:
                # Solo entrenamiento, sin validación
                if verbose:
                    epoch_time = time.time() - epoch_start
                    print(f"\nTrain Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
                    print(f"Tiempo de época: {epoch_time:.1f}s")
        
        # Resumen final del entrenamiento
        total_time = time.time() - start_time
        if verbose:
            print(f"\n{'='*50}")
            print(f"🎉 ¡Entrenamiento completado!")
            print(f"⏱️ Tiempo total: {total_time/60:.1f} minutos")
            print(f"📊 Épocas completadas: {epochs}")
            if self.history['val_accuracy']:
                best_val_acc = max(self.history['val_accuracy'])
                print(f"🏆 Mejor precisión de validación: {best_val_acc:.4f}")
            print(f"{'='*50}")
    
    def evaluate(self, dataset):
        """
        Evalúa el modelo en un dataset completo.
        
        Útil para obtener métricas finales en conjunto de prueba o
        para generar predicciones sobre datos nuevos.
        
        Args:
            dataset: Dataset a evaluar
            
        Returns:
            dict: Diccionario con métricas y predicciones completas
        """
        print("🔍 Iniciando evaluación del modelo...")
        dataset.reset()
        
        # Listas para acumular resultados
        all_predictions = []
        all_labels = []
        losses = []
        
        # Información de progreso
        total_imgs = len(dataset)
        processed_imgs = 0
        batch_size = dataset.batch_size
        total_batches = (total_imgs + batch_size - 1) // batch_size
        
        # Barra de progreso
        pbar = tqdm(total=total_imgs, desc="Evaluación", unit="img")
        
        # Procesar todos los lotes
        while True:
            X_batch, y_batch = dataset.get_batch(augment=False)
            if X_batch is None:
                break
            
            # Actualizar contador
            processed_imgs += len(X_batch)
            pbar.update(len(X_batch))
            
            # Propagación hacia adelante
            y_pred = self.model.forward(X_batch)
            
            # Calcular pérdida
            loss, _ = self.loss_function(y_batch, y_pred)
            losses.append(loss)
            
            # Acumular predicciones y etiquetas
            all_predictions.append(y_pred)
            all_labels.append(y_batch)
        
        pbar.close()
        
        # Concatenar todos los resultados
        all_predictions = np.vstack(all_predictions)
        all_labels = np.concatenate(all_labels)
        
        # Calcular métricas finales
        final_loss = np.mean(losses)
        final_accuracy = self.calculate_accuracy(all_labels, all_predictions)
        
        # Obtener clases predichas
        predicted_classes = np.argmax(all_predictions, axis=1)
        if all_labels.ndim > 1:
            true_classes = np.argmax(all_labels, axis=1)
        else:
            true_classes = all_labels
        
        print(f"✅ Evaluación completada:")
        print(f"   📊 Muestras evaluadas: {len(true_classes):,}")
        print(f"   📉 Pérdida promedio: {final_loss:.4f}")
        print(f"   🎯 Precisión: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        
        return {
            'loss': final_loss,
            'accuracy': final_accuracy,
            'predictions': predicted_classes,
            'labels': true_classes,
            'probabilities': all_predictions
        }
    
    def get_training_history(self):
        """
        Obtiene el historial completo de entrenamiento.
        
        Útil para análisis post-entrenamiento y visualización de curvas
        de aprendizaje.
        
        Returns:
            dict: Historial con pérdidas y precisiones por época
        """
        return self.history.copy()  # Retornar copia para evitar modificaciones
    
    def save_checkpoint(self, filepath):
        """
        Guarda un checkpoint del estado actual del entrenamiento.
        
        Args:
            filepath (str): Ruta donde guardar el checkpoint
        """
        checkpoint = {
            'model_params': self.model.get_params(),
            'history': self.history,
            'optimizer_state': getattr(self.optimizer, '__dict__', {})
        }
        
        np.savez(filepath, **checkpoint)
        print(f"💾 Checkpoint guardado en: {filepath}")
    
    def load_checkpoint(self, filepath):
        """
        Carga un checkpoint previamente guardado.
        
        Args:
            filepath (str): Ruta del checkpoint a cargar
        """
        checkpoint = np.load(filepath, allow_pickle=True)
        
        # Restaurar parámetros del modelo
        if 'model_params' in checkpoint:
            self.model.set_params(checkpoint['model_params'].item())
        
        # Restaurar historial
        if 'history' in checkpoint:
            self.history = checkpoint['history'].item()
        
        print(f"📂 Checkpoint cargado desde: {filepath}")
    
    def predict(self, X):
        """
        Realiza predicciones sobre datos nuevos.
        
        Args:
            X (np.ndarray): Datos de entrada
            
        Returns:
            np.ndarray: Clases predichas
        """
        y_pred = self.model.forward(X)
        return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X):
        """
        Realiza predicciones de probabilidad sobre datos nuevos.
        
        Args:
            X (np.ndarray): Datos de entrada
            
        Returns:
            np.ndarray: Probabilidades por clase
        """
        return self.model.forward(X)
    
    def save_model(self, filepath, include_optimizer=True):
        """
        Guarda el modelo entrenado en un archivo para uso posterior.
        
        Args:
            filepath (str): Ruta donde guardar el modelo
            include_optimizer (bool): Si incluir el estado del optimizador
        """
        import json
        from datetime import datetime
        
        # Extraer todos los parámetros del modelo
        model_params = {}
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'params') and layer.params:
                for param_name, param_value in layer.params.items():
                    key = f"layer_{i}_{param_name}"
                    model_params[key] = param_value
        
        # Agregar metadata del modelo
        model_params['metadata_num_layers'] = len(self.model.layers)
        model_params['metadata_timestamp'] = datetime.now().isoformat()
        
        # Incluir estado del optimizador si se solicita
        if include_optimizer and hasattr(self.optimizer, 'optimizer'):
            opt = self.optimizer.optimizer
            if hasattr(opt, '__dict__'):
                for key, value in opt.__dict__.items():
                    if isinstance(value, (dict, np.ndarray)):
                        model_params[f'optimizer_{key}'] = value
        
        # Guardar usando numpy
        np.savez_compressed(filepath, **model_params)
        print(f"💾 Modelo guardado en: {filepath}")
    
    def save_training_history(self, filepath):
        """
        Guarda el historial de entrenamiento en formato JSON.
        
        Args:
            filepath (str): Ruta donde guardar el historial
        """
        import json
        from datetime import datetime
        
        # Convertir numpy arrays a listas para JSON
        history_json = {}
        for key, values in self.history.items():
            if isinstance(values, list) and values:
                history_json[key] = [float(v) if hasattr(v, 'item') else v for v in values]
            else:
                history_json[key] = values
        
        # Agregar metadata
        history_json['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'epochs_completed': len(history_json.get('train_loss', [])),
            'final_train_accuracy': history_json['train_accuracy'][-1] if history_json.get('train_accuracy') else None,
            'final_val_accuracy': history_json['val_accuracy'][-1] if history_json.get('val_accuracy') else None
        }
        
        # Guardar en JSON
        with open(filepath, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        print(f"📊 Historial guardado en: {filepath}")
    
    @staticmethod
    def load_model(model_instance, filepath):
        """
        Carga parámetros de modelo desde un archivo guardado.
        
        Args:
            model_instance: Instancia del modelo donde cargar los parámetros
            filepath (str): Ruta del archivo del modelo
        """
        model_data = np.load(filepath, allow_pickle=True)
        
        # Cargar parámetros por capa
        for i, layer in enumerate(model_instance.layers):
            if hasattr(layer, 'params'):
                for param_name in layer.params.keys():
                    key = f"layer_{i}_{param_name}"
                    if key in model_data:
                        layer.params[param_name] = model_data[key]
        
        print(f"📂 Modelo cargado desde: {filepath}")