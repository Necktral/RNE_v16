# fase0_cert.py
"""
Automatiza la batería de pruebas Fase 0 Consolidada para AEON FENIX-Δ.
Ejecuta pruebas clave, verifica logs y criterios básicos de éxito.
"""
import subprocess
import sys
import os
import time

def run(cmd, timeout=120, check=True, capture_output=True, shell=True):
    print(f"\n[RUN] {cmd}")
    try:
        result = subprocess.run(cmd, check=check, capture_output=capture_output, shell=shell, text=True, timeout=timeout)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return result
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] El comando excedió {timeout} segundos y fue abortado.")
        return None

def check_file_exists(path):
    if not os.path.exists(path):
        print(f"[FAIL] No existe: {path}")
        return False
    print(f"[OK] Existe: {path}")
    return True

def main():
    # 1. Unit + Integration
    print("\n=== 1. Unit + Integration ===")
    run("pytest -v --maxfail=3 --disable-warnings > test_output.log 2>&1")
    check_file_exists("test_output.log")

    # 2. Smoke-run corto
    print("\n=== 2. Smoke-run corto ===")
    run("python run_aeon.py --cycles 10 --batch 4 --logdir runs/test_smoke --val_interval 2 --noise 0.0", timeout=120)
    check_file_exists("runs/test_smoke")

    # 3. Epoch-stress (memoria)
    print("\n=== 3. Epoch-stress (memoria) ===")
    run("python run_aeon.py --cycles 20 --batch 128 --logdir runs/test_stress --val_interval 5 --noise 0.0", timeout=120)
    check_file_exists("runs/test_stress")

    # 4. Warm-up→Poda real (requiere soporte en Orchestrator)
    print("\n=== 4. Warm-up→Poda real ===")
    run("python run_aeon.py --cycles 50 --batch 16 --katana_warmup 10 --logdir runs/test_poda --val_interval 10 --noise 0.0", timeout=180)
    check_file_exists("runs/test_poda")

    # 5. Val-generalización
    print("\n=== 5. Val-generalización ===")
    run("python run_aeon.py --cycles 30 --batch 8 --logdir runs/test_val --val_interval 5 --noise 0.0", timeout=120)
    check_file_exists("runs/test_val")

    print("\n[INFO] Pruebas automáticas básicas completadas. Revisa logs y TensorBoard para criterios avanzados.")
    print("Puedes extender este script para pruebas 6-10 (reproducibilidad, NaNs, VRAM, etc.) según tus necesidades.")

if __name__ == "__main__":
    main()
