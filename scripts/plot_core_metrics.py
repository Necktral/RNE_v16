# -*- coding: utf-8 -*-
"""
Script para graficar la evolución de VFE y η_bayes a partir de los archivos de métricas generados por la validación AEON.
Soporta tanto core_fase0.json como metrics_log.json.

Uso:
    python scripts/plot_core_metrics.py --input validation_output/core_fase0.json
    python scripts/plot_core_metrics.py --input validation_output/metrics_log.json
Opcional:
    --output <archivo.png>  (guarda la figura)
    --show                 (muestra la figura en pantalla)
"""
import argparse
import json
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Graficar métricas AEON (VFE, η_bayes vs tiempo)")
parser.add_argument('--input', required=True, help='Ruta al archivo JSON de métricas')
parser.add_argument('--output', help='Archivo de salida para la figura (opcional)')
parser.add_argument('--show', action='store_true', help='Mostrar la figura en pantalla')

def main():
    args = parser.parse_args()
    if not os.path.isfile(args.input):
        print(f"[ERROR] El archivo '{args.input}' no existe. No se puede graficar.")
        return
    with open(args.input, encoding='utf-8') as f:
        data = json.load(f)
    # Detectar formato (core_fase0.json: lista de dicts con 'timestamp', metrics_log.json: lista de dicts con 't' y 'label')
    if isinstance(data, list) and 'timestamp' in data[0]:
        t = [d['timestamp'] for d in data]
        vfe = [d['vfe'] for d in data]
        eta = [d['eta_bayes'] for d in data]
        label = None
    elif isinstance(data, list) and 't' in data[0]:
        t = list(range(len(data)))
        vfe = [d['vfe'] for d in data]
        eta = [d['eta_bayes'] for d in data]
        label = [d.get('label','') for d in data]
    else:
        raise ValueError('Formato de archivo no reconocido')
    # Downsampling si hay demasiados datos
    max_points = 10000
    if len(data) > max_points:
        print(f"[INFO] Downsampling: mostrando solo {max_points} de {len(data)} puntos para graficar.")
        step = len(data) // max_points
        data = data[::step]
        if isinstance(data[0], dict) and 'timestamp' in data[0]:
            t = [d['timestamp'] for d in data]
            vfe = [d['vfe'] for d in data]
            eta = [d['eta_bayes'] for d in data]
            label = None
        elif isinstance(data[0], dict) and 't' in data[0]:
            t = list(range(len(data)))
            vfe = [d['vfe'] for d in data]
            eta = [d['eta_bayes'] for d in data]
            label = [d.get('label','') for d in data]
    plt.figure(figsize=(10,6))
    plt.plot(t, vfe, label='VFE', color='tab:blue')
    plt.plot(t, eta, label='η_bayes', color='tab:orange')
    plt.xlabel('Tiempo' if label is None else 'Ciclo')
    plt.ylabel('Valor')
    plt.title('Evolución de VFE y η_bayes')
    plt.legend()
    if label:
        for i, l in enumerate(label):
            if l and (l.endswith('-START') or l.endswith('-END')):
                plt.axvline(i, color='gray', linestyle='--', alpha=0.3)
                plt.text(i, max(vfe[i],eta[i]), l, rotation=90, fontsize=7, va='bottom', ha='right', alpha=0.7)
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f'Figura guardada en {args.output}')
    else:
        import pathlib
        out_path = pathlib.Path(args.input).with_suffix('.png')
        plt.savefig(out_path, dpi=150)
        print(f'Figura guardada en {out_path}')
    # Mostrar la figura solo si hay suficientes datos
    if len(data) > 10:
        plt.show()

if __name__ == "__main__":
    main()
