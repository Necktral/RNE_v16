# -*- coding: utf-8 -*-
"""
Script para graficar métricas de sensores (sensor_data.json):
- cpu_load, gpu_power, memory, temperature, thermal_margin, vram_usage, entropy

Uso:
    python scripts/plot_sensor_data.py --input logs/sensor_data.json [--output sensor_data.png] [--show]
"""
import argparse
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Graficar métricas de sensores")
parser.add_argument('--input', required=True, help='Archivo JSON de datos de sensores')
parser.add_argument('--output', help='Archivo de salida para la figura (opcional)')
parser.add_argument('--show', action='store_true', help='Mostrar la figura en pantalla')

def main():
    args = parser.parse_args()
    import os
    if not os.path.isfile(args.input):
        print(f"[ERROR] El archivo '{args.input}' no existe. No se puede graficar.")
        return
    with open(args.input, encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('El archivo debe ser una lista de dicts')
    fields = ['cpu_load', 'gpu_power', 'memory', 'temperature', 'thermal_margin', 'vram_usage', 'entropy']
    t = [d['timestamp'] for d in data]
    plt.figure(figsize=(12, 10))
    # Downsampling si hay demasiados datos
    max_points = 10000
    if len(data) > max_points:
        print(f"[INFO] Downsampling: mostrando solo {max_points} de {len(data)} puntos para graficar.")
        step = len(data) // max_points
        data = data[::step]
        t = [d['timestamp'] for d in data]
    for i, field in enumerate(fields, 1):
        plt.subplot(len(fields), 1, i)
        plt.plot(t, [d[field] for d in data], label=field)
        plt.ylabel(field)
        if i == 1:
            plt.title('Evolución de métricas de sensores')
        if i == len(fields):
            plt.xlabel('timestamp')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.2)
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f'Figura guardada en {args.output}')
    else:
        # Guardar automáticamente en la misma carpeta que el input
        import pathlib
        out_path = pathlib.Path(args.input).with_suffix('.png')
        plt.savefig(out_path, dpi=150)
        print(f'Figura guardada en {out_path}')
    # Mostrar la figura solo si hay suficientes datos
    if len(data) > 10:
        plt.show()

if __name__ == "__main__":
    main()
