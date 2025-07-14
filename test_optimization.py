#!/usr/bin/env python3
"""
Script de prueba rápida para verificar optimizaciones CNN
"""
import sys
import os
import time
import numpy as np

# Agregar src al path
sys.path.append('src')

def test_optimized_conv():
    """Probar la capa convolucional optimizada"""
    print("🔄 Probando capa convolucional optimizada...")
    
    try:
        from layers.conv_layer import Conv2DLayer
        
        # Crear capa de prueba
        conv = Conv2DLayer(num_filters=16, filter_size=3, stride=1, padding=1)
        
        # Inicializar con la forma de entrada
        conv.initialize_params((3, 96, 96))  # channels, height, width
        
        # Datos de prueba
        batch_size = 8
        test_input = np.random.randn(batch_size, 3, 96, 96)
        
        print(f"📊 Input shape: {test_input.shape}")
        
        # Forward pass
        start_time = time.time()
        output = conv.forward(test_input)
        forward_time = time.time() - start_time
        
        print(f"⚡ Forward pass: {forward_time:.4f} segundos")
        print(f"📊 Output shape: {output.shape}")
        
        # Backward pass
        grad_output = np.random.randn(*output.shape)
        start_time = time.time()
        grad_input = conv.backward(grad_output)
        backward_time = time.time() - start_time
        
        print(f"⚡ Backward pass: {backward_time:.4f} segundos")
        
        total_time = forward_time + backward_time
        print(f"🏁 Tiempo total: {total_time:.4f} segundos")
        
        return total_time
        
    except Exception as e:
        print(f"❌ Error en conv layer: {e}")
        return None

def test_dataset_loading():
    """Probar la carga optimizada de datos"""
    print("\n🔄 Probando carga de datos optimizada...")
    
    try:
        from data.dataset import RetinaDataset
        
        # Verificar si existe el CSV de muestra
        csv_path = "data/sample_15_percent.csv"
        if not os.path.exists(csv_path):
            print(f"⚠️ Dataset no encontrado: {csv_path}")
            return None
            
        # Crear dataset de prueba
        dataset = RetinaDataset(csv_path, batch_size=16, target_size=(96, 96))
        
        print(f"📊 Dataset size: {len(dataset)}")
        
        # Probar carga de un lote
        start_time = time.time()
        images, labels = dataset.get_batch()
        load_time = time.time() - start_time
        
        if images is not None:
            print(f"⚡ Carga de lote: {load_time:.4f} segundos")
            print(f"📊 Batch shape: {images.shape}")
            return load_time
        else:
            print("❌ No se pudo cargar lote")
            return None
            
    except Exception as e:
        print(f"❌ Error en dataset: {e}")
        return None

def estimate_training_time():
    """Estimar tiempo total de entrenamiento"""
    print("\n📊 ESTIMACIÓN DE TIEMPO DE ENTRENAMIENTO")
    print("=" * 45)
    
    conv_time = test_optimized_conv()
    dataset_time = test_dataset_loading()
    
    if conv_time and dataset_time:
        # Configuración optimizada
        batch_size = 32
        total_samples = 6500  # 5% de ~130k
        batches_per_epoch = total_samples // batch_size
        epochs = 5
        
        # Tiempo por lote (conv + carga)
        time_per_batch = conv_time + dataset_time
        
        # Estimaciones
        time_per_epoch = time_per_batch * batches_per_epoch / 60  # minutos
        total_time = time_per_epoch * epochs / 60  # horas
        
        print(f"\n⏱️ ESTIMACIONES:")
        print(f"   🔄 Lotes por época: {batches_per_epoch}")
        print(f"   ⚡ Tiempo por lote: {time_per_batch:.3f} segundos")
        print(f"   📊 Tiempo por época: {time_per_epoch:.1f} minutos")
        print(f"   🚀 Tiempo total (5 épocas): {total_time:.1f} horas")
        
        if total_time < 1:
            print("   ✅ ¡EXCELENTE! < 1 hora")
        elif total_time < 2:
            print("   ✅ ¡BUENO! 1-2 horas")
        elif total_time < 5:
            print("   ⚠️ Aceptable: 2-5 horas")
        else:
            print("   ❌ Aún lento: > 5 horas")
    else:
        print("❌ No se pudo estimar tiempo - errores en componentes")

if __name__ == "__main__":
    print("🚀 PRUEBA DE OPTIMIZACIONES CNN")
    print("=" * 35)
    
    estimate_training_time()