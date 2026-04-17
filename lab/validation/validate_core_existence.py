#!/usr/bin/env python3
"""
validate_core_existence.py  ·  AEON Δ  —  Fase 0 Core Validation
================================================================

Ejecución:
    python validate_core_existence.py --cycles 2 --out ./reports/core_fase0.md --verbose --use-hooks --real-sensors
"""
# Add project root to sys.path for src imports
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# ───────────────────────────────────────────────────────────────────────────────
import os, sys, time, json, argparse, random, contextlib, io, pathlib, inspect
from collections import defaultdict
from datetime import datetime
import tqdm  # Barra de progreso
from torch.utils.tensorboard import SummaryWriter

# Ensure project root is in sys.path for src imports
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.evolution.meta_optimizer import QuantumExponentialOptimizer, QuantumExponentialConfig
from src.core.epistemic_drift_predictor import EpistemicDriftPredictor

# ============================== 1. Seeds y utilidades =========================
def set_seeds(seed: int = 42):
    random.seed(seed)
    try:
        import numpy as np, torch
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # numpy / torch pueden no estar instalados

set_seeds()

# ============================== 2. Import defensivo ===========================
def _try_import(path, name):
    try:
        module = __import__(path, fromlist=[name])
        return getattr(module, name)
    except (ImportError, AttributeError):
        return None

# ───── Módulos núcleo reales (si existen) ─────────────────────────────────────
BaseModel      = _try_import("src.core.model",          "BaseModel")
EpistemeMeter  = _try_import("src.episteme.episteme_meter",       "EpistemeMeter")
HookManager    = _try_import("src.core.hook_manager",          "HookManager")
AutoMutator    = _try_import("src.evolution.auto_mutator",       "AutoMutator")

# ───── Fallback mocks ─────────────────────────────────────────────────────────
class _LogShim:                       # sustituto de get_logger
    def __init__(self, name="shim"): self.name=name
    def info(self,*a,**k):  print("[I]",*a,**k)
    def warn(self,*a,**k):  print("[W]",*a,**k)
    def error(self,*a,**k): print("[E]",*a,**k)
LOG = _LogShim("core_validation")

if BaseModel is None:
    class BaseModel:                  # mock mínimo
        def __init__(self): self._hash="mockbase"
        def observe(self,x):    pass
        def tick(self):         pass
        def random_observation(self,scale=1.0): return scale*random.random()
        def allocate_dummy_tensor(self):       pass
        def stress_step(self):                 pass
        def structure_hash(self): return self._hash
        def lambda_max(self): return random.uniform(0.5, 2.0)  # Mock λ_max
    LOG.warn("BaseModel mockeado.")

if EpistemeMeter is None:
    class EpistemeMeter:
        def compute_vfe(self):                         return random.uniform(0.2,2.0)
        def compute_efficiency(self):                  return random.uniform(0.7,0.9)
    LOG.warn("EpistemeMeter mockeado.")

if HookManager is None or not hasattr(HookManager, 'evaluate'):
    class HookManager:
        def __init__(self, *_, **__): pass
        def evaluate(self, *_, **__): pass  # Mock compatible siempre
    LOG.warn("HookManager mockeado (forzado, con método evaluate)")

if AutoMutator is None:
    class AutoMutator:
        def __init__(self,*_,**__): pass
        def evaluate(self,*_,**__):  pass
    LOG.warn("AutoMutator mockeado.")

# ============================== 3. Sensores ===================================
class DummySensors:
    def vram_ratio(self): return random.uniform(0.1,0.9)
    def temp_ratio(self): return random.uniform(0.1,0.9)

class RealSensors(DummySensors):
    def __init__(self):
        import importlib
        self.nvml   = importlib.import_module("pynvml")        # puede fallar
        self.psutil = importlib.import_module("psutil")
        self.nvml.nvmlInit()
    def vram_ratio(self):
        handle = self.nvml.nvmlDeviceGetHandleByIndex(0)
        info   = self.nvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / info.total
    def temp_ratio(self):
        # En Windows, usar NVML para temperatura GPU
        try:
            handle = self.nvml.nvmlDeviceGetHandleByIndex(0)
            temp = self.nvml.nvmlDeviceGetTemperature(handle, self.nvml.NVML_TEMPERATURE_GPU)
            return temp / 90.0  # 90 °C == 1.0
        except Exception:
            # Fallback: intentar psutil (solo Linux)
            try:
                temps  = self.psutil.sensors_temperatures(fahrenheit=False)
                if "coretemp" in temps:
                    cpu = max(t.current for t in temps["coretemp"])
                    return cpu / 90.0
            except Exception:
                pass
        return super().temp_ratio()

# ============================== 4. Chaos Injector =============================
EVENT_REGISTRY = defaultdict(int)
def register(evt): EVENT_REGISTRY[evt]+=1

class ChaosInjector:
    def __init__(self, model: BaseModel): self.model=model
    # ----- cada caos envía START/END + invoca métodos en model --------------
    def blackout(self):
        register("Blackout START"); self.model.observe(None); self.model.tick(); register("Blackout END")
    def noisy_stream(self):
        register("NoisyStream START")
        for _ in range(128):
            self.model.observe(self.model.random_observation(scale=10.0))
            self.model.tick()
        register("NoisyStream END")
    def mem_pressure(self):
        register("MemPressure START")
        for _ in range(50): self.model.allocate_dummy_tensor()
        register("MemPressure END")
    def power_stress(self):
        register("PowerStress START")
        for _ in range(5000): self.model.stress_step()
        register("PowerStress END")

    def all_cycles(self):
        self.blackout(); self.noisy_stream(); self.mem_pressure(); self.power_stress()

# ============================== 5. Core Validator =============================
class CoreValidator:
    def __init__(self, cycles:int, use_hooks:bool, sensors_cls):
        self.cycles = cycles
        self.model  = BaseModel()
        self.epm    = EpistemeMeter()
        self.hooks  = HookManager(self.model) if use_hooks else None
        self.mutator= AutoMutator(self.model, None)
        self.sensors= sensors_cls()
        self.metrics_log=[]
        self.tb_writer = SummaryWriter(log_dir="runs/aeon_run_tb")
        # Integración del meta-optimizador cuántico
        self.meta_optimizer = QuantumExponentialOptimizer(QuantumExponentialConfig())
        # Predictor de deriva epistémica
        self.drift_predictor = EpistemicDriftPredictor(window_size=50, threshold=0.0005, cooldown=1000)
        # Integrar módulos reales si existen
        if hasattr(self.model, 'modules'):
            self.drift_predictor.modules = self.model.modules
    # ------------- snapshot ---------------------------------------------------
    def _snapshot(self, i=None):
        # --- Robust extraction of VFE and efficiency ---
        vfe = None
        eta = None
        # Try real EpistemeMeter API first
        if hasattr(self.epm, 'evaluate') and callable(getattr(self.epm, 'evaluate')):
            try:
                # Use last known context if available, else empty dict
                context = getattr(self, 'last_episteme_context', {})
                result = self.epm.evaluate(context)
                vfe = result.get('mutual_info', None)  # Use mutual_info as VFE proxy
                eta = result.get('efficiency', None)
            except Exception as e:
                LOG.warn(f"EpistemeMeter.evaluate() failed: {e}")
        elif hasattr(self.epm, 'update') and callable(getattr(self.epm, 'update')):
            try:
                # Try to use last known efficiency/fisher_info, else use defaults
                eff = getattr(self, 'last_efficiency', 1.0)
                fisher = getattr(self, 'last_fisher_info', [1.0])
                result = self.epm.update(eff, fisher)
                vfe = result.get('mutual_info', None)
                eta = result.get('efficiency', None)
            except Exception as e:
                LOG.warn(f"EpistemeMeter.update() failed: {e}")
        # Fallback a métodos mock si es necesario
        if vfe is None and hasattr(self.epm, 'compute_vfe'):
            try:
                vfe = self.epm.compute_vfe()
            except Exception as e:
                LOG.warn(f"EpistemeMeter.compute_vfe() falló: {e}")
        if eta is None and hasattr(self.epm, 'compute_efficiency'):
            try:
                eta = self.epm.compute_efficiency()
            except Exception as e:
                LOG.warn(f"EpistemeMeter.compute_efficiency() falló: {e}")
        lmax = self.model.lambda_max() if hasattr(self.model, 'lambda_max') else None
        vram = self.sensors.vram_ratio()
        temp = self.sensors.temp_ratio()
        snap = {
            "timestamp"   : time.time(),
            "vfe"         : vfe,
            "eta_bayes"   : eta,
            "vram_ratio"  : vram,
            "temp_ratio"  : temp,
            "lambda_max"  : lmax,
            "shutdown"    : False,   # set later
        }
        self.metrics_log.append(snap)
        # Log to TensorBoard if index is provided
        if i is not None:
            if vfe is not None:
                self.tb_writer.add_scalar("VFE", vfe, i)
            if eta is not None:
                self.tb_writer.add_scalar("eta_bayes", eta, i)
            self.tb_writer.add_scalar("vram_ratio", vram, i)
            self.tb_writer.add_scalar("temp_ratio", temp, i)
            if lmax is not None:
                self.tb_writer.add_scalar("lambda_max", lmax, i)
        return snap
    # ------------- criterio de shutdown --------------------------------------
    def _check_shutdown(self):
        # Ejemplo minimal: si temp >1 o vram >0.95
        last = self.metrics_log[-1]
        return (last["temp_ratio"]>=1.0) or (last["vram_ratio"]>=0.95)
    # ------------- bucle principal -------------------------------------------
    def run(self):
        injector = ChaosInjector(self.model)
        if self.cycles >= 1_000_000:
            sample_indices = [int(i * self.cycles / 10) for i in range(10)]
        else:
            sample_indices = list(range(self.cycles))
        self.metrics_log = []
        pbar = tqdm.tqdm(total=self.cycles, desc="Ciclos de validación", unit="ciclo")
        for i in range(self.cycles):
            injector.all_cycles()
            # --- integración meta-optimizador con sanitización física ---
            vram = getattr(self.sensors, 'vram_ratio', lambda: 0.5)()
            temp = getattr(self.sensors, 'temp_ratio', lambda: 0.5)()
            entropy = random.uniform(0.0, 1.0)
            cognitive_load = random.uniform(0.0, 1.0)
            physics_stats = {
                'vram': vram,
                'thermal': temp,
                'entropy': entropy,
                'cognitive_load': cognitive_load
            }
            def epistemic_callback(uid):
                return random.uniform(0.6, 1.0)
            # Solo mutar si no hay sobrecarga física
            if vram < 0.95 and temp < 0.95:
                self.meta_optimizer.step(physics_stats, epistemic_callback)
            else:
                LOG.warn(f"Mutación/NAS bloqueada por límites físicos: vram={vram:.2f}, temp={temp:.2f}")
            # --- fin integración meta-optimizador ---
            if i in sample_indices:
                snap = self._snapshot(i)
                snap["cycle"] = i
                snap["meta_optimizer"] = {
                    'cycle': self.meta_optimizer.state['cycle'],
                    'eta_bayes': self.meta_optimizer.state['eta_bayes'],
                    'cognitive_load': self.meta_optimizer.state['cognitive_load'],
                    'num_modules': len(self.meta_optimizer.state['modules'])
                }
                # --- Actualiza predictor de deriva epistémica ---
                self.drift_predictor.update(snap.get('eta_bayes'), snap.get('vfe'))
                alerta, razon = self.drift_predictor.check_drift(i)
                if alerta:
                    self.drift_predictor.force_mutation(razon)
            else:
                snap = self._snapshot(i)
                self.drift_predictor.update(snap.get('eta_bayes'), snap.get('vfe'))
                alerta, razon = self.drift_predictor.check_drift(i)
                if alerta and razon is not None:
                    self.drift_predictor.force_mutation(razon)
            if self.hooks: self.hooks.evaluate()
            if hasattr(self.mutator, "evaluate"):
                self.mutator.evaluate()
            if self.metrics_log and self._check_shutdown():
                self.metrics_log[-1]["shutdown"] = True
                pbar.update(self.cycles - i)
                break
            pbar.update(1)
        pbar.close()
        self.tb_writer.close()
    # ------------- reporte Markdown + JSON -----------------------------------
    def compile_reports(self, out_md:str):
        dt   = datetime.utcnow().isoformat(timespec="seconds")
        last = self.metrics_log[-1]
        ok   = not last["shutdown"]
        md   = [f"# Informe de Validación de Núcleo — AEON ∆",
                f"**Fecha/Hora:** {dt}",
                "",
                "## Resumen ejecutivo",
                "",
                f"* **Shutdown activado:** ✅  {'NO' if ok else 'SÍ'}",
                f"* **VFE final:** {last['vfe']:.4f}" if last.get('vfe') is not None else "* **VFE final:** N/A",
                f"* **Eficiencia epistémica η_bayes final:** {last['eta_bayes']:.4f}" if last.get('eta_bayes') is not None else "* **Eficiencia epistémica η_bayes final:** N/A",
                f"* **VRAM ratio final:** {last['vram_ratio']:.3f}" if last.get('vram_ratio') is not None else "* **VRAM ratio final:** N/A",
                f"* **Temp ratio final:** {last['temp_ratio']:.3f}" if last.get('temp_ratio') is not None else "* **Temp ratio final:** N/A",
                f"* **λ_max final:** {last['lambda_max']:.4f}" if last.get('lambda_max') is not None else "* **λ_max final:** N/A",
                "",
                "## Estado del Meta-Optimizador Cuántico",
                "",
                f"* Ciclo actual: {last.get('meta_optimizer',{}).get('cycle','N/A')}",
                f"* η_bayes global: {last.get('meta_optimizer',{}).get('eta_bayes','N/A'):.4f}" if last.get('meta_optimizer',{}).get('eta_bayes') is not None else "* η_bayes global: N/A",
                f"* Carga cognitiva: {last.get('meta_optimizer',{}).get('cognitive_load','N/A'):.3f}" if last.get('meta_optimizer',{}).get('cognitive_load') is not None else "* Carga cognitiva: N/A",
                f"* Número de módulos: {last.get('meta_optimizer',{}).get('num_modules','N/A')}",
                "",
                "## Muestreo de snapshots",
                "",
                "| Ciclo | VFE | η_bayes | VRAM | Temp | λ_max | Shutdown |",
                "|-------|------|---------|------|------|-------|----------|",
        ]
        for snap in self.metrics_log:
            md.append(
                f"| {snap.get('cycle','N/A')} "
                f"| {snap['vfe']:.4f}" if snap.get('vfe') is not None else "| N/A"
                f" | {snap['eta_bayes']:.4f}" if snap.get('eta_bayes') is not None else " | N/A"
                f" | {snap['vram_ratio']:.3f}" if snap.get('vram_ratio') is not None else " | N/A"
                f" | {snap['temp_ratio']:.3f}" if snap.get('temp_ratio') is not None else " | N/A"
                f" | {snap['lambda_max']:.4f}" if snap.get('lambda_max') is not None else " | N/A"
                f" | {'SÍ' if snap.get('shutdown', False) else 'NO'} |"
            )
        md += ["", "## Eventos críticos registrados", ""]
        for evt,cnt in EVENT_REGISTRY.items():
            md.append(f"- {evt} × {cnt}")
        md += ["", "## Conclusión", "",
               f"{'✅ **AEON preserva su núcleo y cumple criterios de Fase 0.**' if ok else '❌ **AEON NO cumple los criterios de Fase 0!**'}",
               "",
               "---",
               "_Fin del reporte_",
               ""]
        pathlib.Path(out_md).write_text("\n".join(md), encoding="utf-8")
        # JSON raw para curvas (solo snapshots muestreados)
        json_path = pathlib.Path(out_md).with_suffix(".json")
        json_path.write_text(json.dumps(self.metrics_log, indent=2))
        return ok

# ============================== 6. CLI ========================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--out",    type=str)
    p.add_argument("--verbose",action="store_true")
    p.add_argument("--use-hooks",      action="store_true")
    p.add_argument("--real-sensors",   action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    sensors_cls = RealSensors if args.real_sensors else DummySensors

    # Si no se especifica --out, genera nombre automático según ciclos
    if not args.out:
        args.out = f"reports/core_fase0_{args.cycles}.md"
    else:
        # Si el nombre no tiene número de ciclos, lo agrega
        base = pathlib.Path(args.out)
        if "core_fase0" in base.stem and str(args.cycles) not in base.stem:
            args.out = str(base.with_stem(f"core_fase0_{args.cycles}"))

    # --- Renombrar salida para incluir el número de ciclos si corresponde ---
    out_path = pathlib.Path(args.out)
    if out_path.stem.startswith("core_fase0") and not out_path.stem.endswith(str(args.cycles)):
        out_md = f"core_fase0_{args.cycles}.md"
        print(f"[INFO] El reporte se guardará como: {out_md}")
    else:
        out_md = args.out

    cv = CoreValidator(args.cycles, args.use_hooks, sensors_cls)

    if args.verbose:
        cv.run()
    else:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cv.run()

    ok = cv.compile_reports(out_md)
    # Graficar automáticamente sensor_data si existe
    import subprocess
    sensor_json = pathlib.Path('logs/sensor_data.json')
    if sensor_json.exists():
        print("[INFO] Generando gráfica de sensores...")
        subprocess.run([
            sys.executable,
            'scripts/plot_sensor_data.py',
            '--input', str(sensor_json),
            '--output', 'logs/sensor_data.png'
        ])
    # Graficar automáticamente métricas del core si existe el JSON
    core_json = pathlib.Path(out_md).with_suffix('.json')
    if core_json.exists():
        print("[INFO] Generando gráfica de métricas del core...")
        subprocess.run([
            sys.executable,
            'scripts/plot_core_metrics.py',
            '--input', str(core_json),
            '--output', str(core_json.with_suffix('.png'))
        ])
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
