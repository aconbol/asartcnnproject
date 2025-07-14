import numpy as np

class Metrics:
    """
    Clase que implementa métricas de evaluación para modelos de clasificación.
    
    Las métricas permiten evaluar qué tan bien se desempeña el modelo entrenado
    en tareas de clasificación. Incluye tanto métricas básicas como reportes
    detallados por clase, esenciales para diagnóstico médico.
    
    Todas las implementaciones están optimizadas y son autocontenidas,
    sin dependencias externas más allá de NumPy.
    """
    
    @staticmethod
    def accuracy(y_true, y_pred):
        """
        Calcula la precisión (accuracy) - fracción de predicciones correctas.
        
        Accuracy = (Verdaderos Positivos + Verdaderos Negativos) / Total
        
        Es la métrica más básica e intuitiva, pero puede ser engañosa
        en datasets desbalanceados.
        
        Args:
            y_true (np.ndarray): Etiquetas verdaderas
            y_pred (np.ndarray): Predicciones del modelo
        
        Returns:
            float: Precisión entre 0 y 1
        """
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision(y_true, y_pred, average='macro'):
        """
        Calcula la precisión (precision) por clase.
        
        Precision = Verdaderos Positivos / (Verdaderos Positivos + Falsos Positivos)
        
        Mide qué fracción de las predicciones positivas fueron correctas.
        Importante en medicina: cuando predecimos enfermedad, ¿qué tan seguros estamos?
        
        Args:
            y_true (np.ndarray): Etiquetas verdaderas
            y_pred (np.ndarray): Predicciones del modelo
            average (str): 'macro', 'micro', o None para precisión por clase
        
        Returns:
            float o list: Precisión promediada o por clase
        """
        classes = np.unique(y_true)
        precisions = []
        
        for cls in classes:
            # Verdaderos positivos: casos donde predijimos cls y era cls
            true_pos = np.sum((y_true == cls) & (y_pred == cls))
            # Todos los casos donde predijimos cls (correctos e incorrectos)
            pred_pos = np.sum(y_pred == cls)
            
            if pred_pos == 0:
                # Evitar división por cero cuando nunca predijimos esta clase
                precision = 0.0
            else:
                precision = true_pos / pred_pos
            
            precisions.append(precision)
        
        if average == 'macro':
            # Promedio simple de precisiones por clase
            return np.mean(precisions)
        elif average == 'micro':
            # Precisión global (equivale a accuracy en multiclase)
            return np.sum(y_true == y_pred) / len(y_true)
        else:
            # Retornar precisión por cada clase
            return precisions
    
    @staticmethod
    def recall(y_true, y_pred, average='macro'):
        """
        Calcula el recall (sensibilidad) por clase.
        
        Recall = Verdaderos Positivos / (Verdaderos Positivos + Falsos Negativos)
        
        Mide qué fracción de casos positivos reales fueron detectados.
        Crítico en medicina: de todos los pacientes enfermos, ¿cuántos detectamos?
        
        Args:
            y_true (np.ndarray): Etiquetas verdaderas
            y_pred (np.ndarray): Predicciones del modelo
            average (str): 'macro', 'micro', o None para recall por clase
        
        Returns:
            float o list: Recall promediado o por clase
        """
        classes = np.unique(y_true)
        recalls = []
        
        for cls in classes:
            # Verdaderos positivos: casos donde predijimos cls y era cls
            true_pos = np.sum((y_true == cls) & (y_pred == cls))
            # Todos los casos reales de cls
            actual_pos = np.sum(y_true == cls)
            
            if actual_pos == 0:
                # Evitar división por cero cuando no hay casos de esta clase
                recall = 0.0
            else:
                recall = true_pos / actual_pos
            
            recalls.append(recall)
        
        if average == 'macro':
            # Promedio simple de recalls por clase
            return np.mean(recalls)
        elif average == 'micro':
            # Recall global (equivale a accuracy en multiclase)
            return np.sum(y_true == y_pred) / len(y_true)
        else:
            # Retornar recall por cada clase
            return recalls
    
    @staticmethod
    def f1_score(y_true, y_pred, average='macro'):
        """
        Calcula el F1-score - media armónica de precisión y recall.
        
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Balancea precisión y recall en una sola métrica. Útil cuando
        necesitamos un equilibrio entre ambas (común en medicina).
        
        Args:
            y_true (np.ndarray): Etiquetas verdaderas
            y_pred (np.ndarray): Predicciones del modelo
            average (str): 'macro', 'micro', o None para F1 por clase
        
        Returns:
            float o list: F1-score promediado o por clase
        """
        # Calcular precisión y recall por clase
        precision = Metrics.precision(y_true, y_pred, average=None)
        recall = Metrics.recall(y_true, y_pred, average=None)
        
        f1_scores = []
        for p, r in zip(precision, recall):
            if p + r == 0:
                # Evitar división por cero
                f1 = 0.0
            else:
                # Media armónica de precisión y recall
                f1 = 2 * (p * r) / (p + r)
            f1_scores.append(f1)
        
        if average == 'macro':
            # Promedio simple de F1-scores por clase
            return np.mean(f1_scores)
        elif average == 'micro':
            # F1 global (equivale a accuracy en multiclase)
            return Metrics.accuracy(y_true, y_pred)
        else:
            # Retornar F1-score por cada clase
            return f1_scores
    
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """
        Calcula la matriz de confusión.
        
        La matriz de confusión muestra cómo el modelo confunde unas clases con otras.
        Filas = clases verdaderas, Columnas = clases predichas.
        Diagonal = predicciones correctas.
        
        Esencial para entender errores del modelo en diagnóstico médico:
        ¿El modelo confunde RD severa con RD moderada?
        
        Args:
            y_true (np.ndarray): Etiquetas verdaderas
            y_pred (np.ndarray): Predicciones del modelo
        
        Returns:
            np.ndarray: Matriz de confusión (n_classes x n_classes)
        """
        # Obtener todas las clases presentes en los datos
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)
        
        # Inicializar matriz de ceros
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        # Llenar matriz contando co-ocurrencias
        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                # Contar casos donde clase real=i y predicha=j
                cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
        
        return cm
    
    @staticmethod
    def classification_report(y_true, y_pred, class_names=None):
        """
        Genera un reporte detallado de clasificación.
        
        Incluye precisión, recall, F1-score y soporte (número de muestras)
        para cada clase, más métricas promediadas.
        
        Args:
            y_true (np.ndarray): Etiquetas verdaderas
            y_pred (np.ndarray): Predicciones del modelo
            class_names (list, optional): Nombres de las clases para mostrar
        
        Returns:
            dict: Reporte estructurado con métricas por clase
        """
        if class_names is None:
            classes = np.unique(y_true)
            class_names = [f"Clase {i}" for i in classes]
        
        classes = np.unique(y_true)
        report = {}
        
        # Calcular métricas por clase
        precision_scores = Metrics.precision(y_true, y_pred, average=None)
        recall_scores = Metrics.recall(y_true, y_pred, average=None)
        f1_scores = Metrics.f1_score(y_true, y_pred, average=None)
        
        for i, cls in enumerate(classes):
            # Contar muestras de esta clase
            support = np.sum(y_true == cls)
            
            report[class_names[i]] = {
                'precision': precision_scores[i],
                'recall': recall_scores[i],
                'f1-score': f1_scores[i],
                'support': support
            }
        
        # Métricas globales
        report['accuracy'] = Metrics.accuracy(y_true, y_pred)
        report['macro avg'] = {
            'precision': Metrics.precision(y_true, y_pred, average='macro'),
            'recall': Metrics.recall(y_true, y_pred, average='macro'),
            'f1-score': Metrics.f1_score(y_true, y_pred, average='macro'),
            'support': len(y_true)
        }
        
        # Promedio ponderado por soporte
        weighted_precision = np.average(precision_scores, weights=[np.sum(y_true == cls) for cls in classes])
        weighted_recall = np.average(recall_scores, weights=[np.sum(y_true == cls) for cls in classes])
        weighted_f1 = np.average(f1_scores, weights=[np.sum(y_true == cls) for cls in classes])
        
        report['weighted avg'] = {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1-score': weighted_f1,
            'support': len(y_true)
        }
        
        return report
    
    @staticmethod
    def print_classification_report(y_true, y_pred, class_names=None):
        """
        Imprime un reporte de clasificación formateado.
        
        Versión legible del reporte de clasificación, útil para análisis rápido
        del rendimiento del modelo en cada clase.
        
        Args:
            y_true (np.ndarray): Etiquetas verdaderas
            y_pred (np.ndarray): Predicciones del modelo
            class_names (list, optional): Nombres de las clases
        """
        report = Metrics.classification_report(y_true, y_pred, class_names)
        
        print("\n" + "=" * 80)
        print("📊 REPORTE DE CLASIFICACIÓN DETALLADO")
        print("=" * 80)
        
        # Encabezado
        header = f"{'Clase':<25} {'Precisión':<12} {'Recall':<12} {'F1-Score':<12} {'Soporte':<12}"
        print(header)
        print("-" * 80)
        
        # Métricas por clase
        for class_name, metrics in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                precision = metrics['precision']
                recall = metrics['recall']
                f1 = metrics['f1-score']
                support = metrics['support']
                
                row = f"{class_name:<25} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f} {support:<12}"
                print(row)
        
        print("-" * 80)
        
        # Métricas globales
        accuracy = report['accuracy']
        print(f"{'Accuracy':<25} {'':<12} {'':<12} {accuracy:<12.3f} {report['macro avg']['support']:<12}")
        
        # Promedio macro
        macro = report['macro avg']
        print(f"{'Macro Avg':<25} {macro['precision']:<12.3f} {macro['recall']:<12.3f} "
              f"{macro['f1-score']:<12.3f} {macro['support']:<12}")
        
        # Promedio ponderado
        weighted = report['weighted avg']
        print(f"{'Weighted Avg':<25} {weighted['precision']:<12.3f} {weighted['recall']:<12.3f} "
              f"{weighted['f1-score']:<12.3f} {weighted['support']:<12}")
        
        print("=" * 80)
        
        # Interpretación para diagnóstico médico
        print("\n💡 INTERPRETACIÓN PARA DIAGNÓSTICO MÉDICO:")
        print("-" * 50)
        print("• Precisión: De los casos que predijimos como positivos, ¿cuántos eran realmente positivos?")
        print("• Recall: De todos los casos positivos reales, ¿cuántos detectamos correctamente?")
        print("• F1-Score: Balance entre precisión y recall (importante para diagnóstico)")
        print("• Soporte: Número de casos reales de cada clase en el conjunto de prueba")
        print()
    
    @staticmethod
    def specificity(y_true, y_pred, average='macro'):
        """
        Calcula la especificidad (true negative rate) por clase.
        
        Specificity = Verdaderos Negativos / (Verdaderos Negativos + Falsos Positivos)
        
        Mide qué fracción de casos negativos reales fueron correctamente identificados.
        En medicina: de todos los pacientes sanos, ¿cuántos identificamos correctamente?
        
        Args:
            y_true (np.ndarray): Etiquetas verdaderas
            y_pred (np.ndarray): Predicciones del modelo
            average (str): 'macro', 'micro', o None para especificidad por clase
        
        Returns:
            float o list: Especificidad promediada o por clase
        """
        classes = np.unique(y_true)
        specificities = []
        
        for cls in classes:
            # Verdaderos negativos: casos donde no predijimos cls y no era cls
            true_neg = np.sum((y_true != cls) & (y_pred != cls))
            # Todos los casos que realmente no son cls
            actual_neg = np.sum(y_true != cls)
            
            if actual_neg == 0:
                # Evitar división por cero
                specificity = 0.0
            else:
                specificity = true_neg / actual_neg
            
            specificities.append(specificity)
        
        if average == 'macro':
            return np.mean(specificities)
        elif average == 'micro':
            # Para multiclase, micro-average de specificity es más complejo
            # Usamos la definición general
            all_tn = np.sum([np.sum((y_true != cls) & (y_pred != cls)) for cls in classes])
            all_an = np.sum([np.sum(y_true != cls) for cls in classes])
            return all_tn / all_an if all_an > 0 else 0.0
        else:
            return specificities