import torch
print("GPU Disponible:", torch.cuda.is_available())
print("Nombre:", torch.cuda.get_device_name(0))
