#!/usr/bin/env python3
"""
Test rápido para verificar que la capa convolucional corregida funciona
"""
import sys
import numpy as np

sys.path.append('src')

def test_conv_layer():
    print("🔄 Probando capa convolucional corregida...")
    
    try:
        from layers.conv_layer import Conv2DLayer
        
        # Crear capa de prueba
        conv = Conv2DLayer(num_filters=16, filter_size=3, stride=1, padding=1)
        
        # Inicializar con forma de entrada
        conv.initialize_params((3, 96, 96))
        
        # Datos de prueba
        batch_size = 4
        test_input = np.random.randn(batch_size, 3, 96, 96)
        
        print(f"📊 Input shape: {test_input.shape}")
        
        # Forward pass
        print("⚡ Ejecutando forward pass...")
        output = conv.forward(test_input)
        print(f"✅ Forward exitoso! Output shape: {output.shape}")
        
        # Backward pass
        print("⚡ Ejecutando backward pass...")
        grad_output = np.random.randn(*output.shape)
        grad_input = conv.backward(grad_output)
        print(f"✅ Backward exitoso! Grad input shape: {grad_input.shape}")
        
        print("🎉 ¡Capa convolucional funciona correctamente!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 TEST DE CORRECCIÓN DE CAPA CONVOLUCIONAL")
    print("=" * 44)
    
    if test_conv_layer():
        print("\n✅ CORRECCIÓN EXITOSA - LISTO PARA ENTRENAMIENTO")
    else:
        print("\n❌ AÚN HAY ERRORES - NECESITA MÁS CORRECCIÓN")