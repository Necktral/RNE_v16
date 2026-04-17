# scripts/generate.py
import torch
import torch.nn as nn

from configs import hnet_760m_config as aeon_config
from aeon.models.hnet import HNet
from aeon.utils.device import get_device, get_dtype

def generate():
    device = get_device()
    dtype = get_dtype()
    vocab_size = 256

    # 1. Re-instanciar la arquitectura del modelo
    embedding = nn.Embedding(vocab_size, aeon_config.D_MODEL).to(device)
    model = HNet(d_model=aeon_config.D_MODEL, n_layers=aeon_config.N_LAYERS).to(device, dtype=dtype)
    output_head = nn.Linear(aeon_config.D_MODEL, vocab_size).to(device)

    # 2. Cargar los pesos entrenados desde el checkpoint
    try:
        checkpoint = torch.load(aeon_config.SAVE_PATH, map_location=device)
        embedding.load_state_dict(checkpoint['embedding_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        output_head.load_state_dict(checkpoint['output_head_state_dict'])
        print("Checkpoint del modelo cargado exitosamente.")
    except FileNotFoundError:
        print(f"Error: No se encontró el checkpoint en {aeon_config.SAVE_PATH}. Por favor, ejecuta el script de entrenamiento primero.")
        return

    # Poner los modelos en modo de evaluación
    embedding.eval()
    model.eval()
    output_head.eval()

    # 3. Iniciar la generación
    print("\n--- INICIO DE LA GENERACIÓN DE TEXTO ---")
    # Empezamos con un carácter de nueva línea (byte=10)
    # El modelo espera una secuencia, así que le damos forma (batch_size, seq_len) -> (1, 1)
    input_bytes = torch.tensor([[10]], dtype=torch.long, device=device)

    generated_bytes = []

    # Generamos 300 caracteres (bytes)
    with torch.no_grad(): # No necesitamos calcular gradientes
        for _ in range(300):
            with torch.autocast(device_type=device, dtype=dtype, enabled=(dtype == torch.float16)):
                x_emb = embedding(input_bytes).to(dtype)
                logits, _ = model(x_emb) # La ratio_loss no es necesaria en la inferencia
                
                # Nos enfocamos solo en el último paso de tiempo para la predicción
                last_step_logits = logits[:, -1, :] 
                final_logits = output_head(last_step_logits.to(torch.float32))

            # Convertimos los logits a probabilidades y muestreamos el siguiente byte
            probs = torch.nn.functional.softmax(final_logits, dim=-1)
            next_byte = torch.multinomial(probs, num_samples=1)
            
            # Añadimos el nuevo byte a nuestra secuencia de entrada para el siguiente paso
            input_bytes = torch.cat((input_bytes, next_byte), dim=1)
            
            generated_bytes.append(next_byte.item())

    # 4. Decodificar y mostrar el resultado
    generated_text = "".join([chr(b) for b in generated_bytes])
    print(generated_text)
    print("\n--- FIN DE LA GENERACIÓN ---")


if __name__ == "__main__":
    generate()