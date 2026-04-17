from __future__ import annotations
import time, math, warnings
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Callable, Tuple, List, Optional, Any

import torch, psutil
from torch import Tensor, distributions
try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # pragma: no cover
    _GPU_HANDLE = None

from aeon.core.event_bus import event_bus  # Integración EventBus centralizado

# ────────────────────────────────────────────────── CONSTANTS ──────────────────────────────────────────────────
QUANTUM_TEMP = 0.1  # Temperatura cuántica base
QUANTUM_FREQ = 0.5  # Frecuencia de fluctuaciones cuánticas (Hz)
QUANTUM_AMP = 0.05  # Amplitud de fluctuaciones cuánticas
MAX_LYAPUNOV = 1000 # Umbral para resurrección caótica (λ_max > 1000)

# ────────────────────────────────────────────────── PUBLIC SNAPSHOT ──────────────────────────────────────────────────
@dataclass(frozen=True)
class EpistemeSnapshot:
    t_wall: float
    epist_int: float
    w_int: float
    fisher_int: float
    fisher_matrix: Tensor
    lyapunov_exp: float
    entropy_production: float
    efficiency: float
    fisher_density: float
    quantum_beta: float
    events: Tuple[str, ...]
    memory_usage: float
    temperature: float

# ────────────────────────────────────────────────── MAIN CLASS ──────────────────────────────────────────────────
class EpistemeMeter:
    """Meta-sensor termodinámico avanzado para AEON FENIX-Δ con física cuántica integrada."""
    
    # region ───────── INIT ─────────
    def __init__(
        self,
        tau_epist: float = 30.0,         # ventana s-1 para Δ_epist
        tau_power: float = 30.0,         # idem potencia
        tau_fisher: float = 60.0,        # ventana Fisher
        max_dt: float = 10.0,            # corte de pausa
        epsilon: float = 1e-9,           # reg numérica
        fisher_damp: float = 1e-3,       # damping Fisher
        pid_cfg: Optional[Dict] = None,
        enable_quantum: bool = True,     # Habilitar correcciones cuánticas
        enable_lyapunov: bool = True,    # Habilitar cálculo de Lyapunov
        skip_sensor_update: bool = False # NUEVO: permite desactivar sensores para test
    ):
        self.tau_epist, self.tau_power, self.tau_fisher = tau_epist, tau_power, tau_fisher
        self.max_dt = max_dt
        self.eps = epsilon
        self.fisher_damp = fisher_damp
        self.enable_quantum = enable_quantum
        self.enable_lyapunov = enable_lyapunov
        self.quantum_temp = QUANTUM_TEMP
        self.quantum_freq = QUANTUM_FREQ
        self.quantum_amp = QUANTUM_AMP
        self._skip_sensor_update = skip_sensor_update

        # integrales exponenciales
        self._int_epist = self._int_power = self._int_fisher = 0.0
        self._last_t: float | None = None

        # buffers ventana rectangular (para speed reset)
        self._buf_epist: Deque[Tuple[float, float]] = deque()
        self._buf_power: Deque[Tuple[float, float]] = deque()
        self._buf_fisher: Deque[Tuple[float, float]] = deque()

        # hooks externos
        self._hooks: Dict[str, Callable[[EpistemeSnapshot], None]] = {}
        self._events: List[str] = []

        # PID Thermostat
        self.pid_cfg = pid_cfg or {"k_p": .1, "k_i": .01, "k_d": .0,
                                   "target": .1, "beta": 1.0,
                                   "min_beta": 0.0, "max_beta": 10.0}
        self._pid_err_i = self._pid_prev = 0.0
        
        # Memoria epistémica para estados críticos
        self.epistemic_memory: List[Dict[str, Any]] = []
        self.last_lyapunov = 0.0
        
        # Estados de homeostasis
        self.memory_usage = 0.0
        self.temperature = 0.0

    # endregion

    # region ───────── PUBLIC API ─────────
    @torch.no_grad()
    def update(
        self,
        posterior: Tensor,              # (B,…) prob.   q
        prior:     Tensor,              # (B,…) prob.   p
        hidden_act: Optional[Tensor] = None,  # (B,D) activ latentes
        *,
        t_wall: float | None = None,
        w_override: float | None = None,
        skip_sensor_update: bool | None = None # NUEVO: permite forzar el flag por llamada
    ) -> EpistemeSnapshot:
        """Paso de medición avanzado con física cuántica y detección de crisis."""
        t_now = t_wall or time.time()
        if self._last_t is None:
            self._last_t = t_now
            # Inicializar snapshot con valores por defecto
            # Verificar homeostasis en la inicialización para tests
            self._check_homeostasis()
            fisher_matrix = torch.zeros(1, 1)
            lyapunov_exp = 0.0
            entropy_prod = 0.0
            beta = self.pid_cfg["beta"] if "beta" in self.pid_cfg else 1.0
            snap = self._snapshot(t_now, fisher_matrix, lyapunov_exp, entropy_prod, beta)
            # No emitir hooks en la inicialización para evitar callback extra
            self._events.clear()
            return snap

        dt = t_now - self._last_t
        if dt > self.max_dt:                # pausa larga → reset
            self._reset_integrals()
            self._last_t = t_now
            # Inicializar snapshot con valores por defecto
            fisher_matrix = torch.zeros(1, 1)
            lyapunov_exp = 0.0
            entropy_prod = 0.0
            beta = self.pid_cfg["beta"] if "beta" in self.pid_cfg else 1.0
            snap = self._snapshot(t_now, fisher_matrix, lyapunov_exp, entropy_prod, beta)
            self._emit_hooks(snap)
            self._events.clear()
            return snap
        self._last_t = t_now

        # ----- Actualizar sensores físicos SOLO si no está desactivado
        if not (skip_sensor_update if skip_sensor_update is not None else self._skip_sensor_update):
            self._update_physical_sensors()

        # ----- Δ_epist (KL) estable
        delta_epist_tensor = self._kl_stable(posterior, prior, as_tensor=True)
        if isinstance(delta_epist_tensor, torch.Tensor):
            delta_epist = delta_epist_tensor.mean().item()
        else:
            delta_epist = float(delta_epist_tensor)

        # ----- Fisher-info matricial
        fisher_matrix = self._full_fisher(posterior, prior)
        fisher_trace = fisher_matrix.diag().sum().item() if fisher_matrix is not None else 0.0

        # ----- Potencia disipada
        w_diss = w_override if w_override is not None else _read_power()

        # ----- Producción de entropía estocástica
        entropy_prod = self._stochastic_entropy_production(delta_epist, w_diss)

        # ----- Exponente de Lyapunov
        lyapunov_exp = self._compute_lyapunov(posterior, prior) if self.enable_lyapunov else 0.0
        self.last_lyapunov = lyapunov_exp

        # ----- Integraciones
        self._integrate(self._buf_epist, delta_epist * dt, self.tau_epist, "_int_epist", dt)
        self._integrate(self._buf_power, w_diss * dt, self.tau_power, "_int_power", dt)
        self._integrate(self._buf_fisher, fisher_trace * dt, self.tau_fisher, "_int_fisher", dt)

        # ----- Verificar homeostasis y disparar eventos (MOVER ANTES DEL SNAPSHOT)
        self._check_homeostasis()

        # ----- Thermostat-PID con corrección cuántica
        beta = self._quantum_pid(delta_epist, w_diss) if self.enable_quantum else self._classic_pid(delta_epist, w_diss)
        self.pid_cfg["beta"] = beta

        # ----- Crear snapshot
        snap = self._snapshot(t_now, fisher_matrix, lyapunov_exp, entropy_prod, beta)

        # ----- Guardar estado crítico si es necesario
        if any(e in snap.events for e in ["CHAOTIC_RESURRECTION_TRIGGER", "THERMAL_CRISIS"]):
            self._save_critical_state(snap)

        self._emit_hooks(snap)
        self._events.clear()
        return snap
    # endregion

    # region ───────── CORE IMPROVEMENTS ─────────
    def _full_fisher(self, q: Tensor, p: Tensor) -> Tensor:
        """Calcula la matriz completa de Fisher Information."""
        if q.requires_grad:
            q = q.detach()
        q.requires_grad_(True)
        log_q = torch.log(q + self.eps)
        try:
            grads = []
            for i in range(q.shape[0]):
                grad_i = torch.autograd.grad(log_q[i].sum(), q, retain_graph=True)[0]
                grads.append(grad_i[i].unsqueeze(0))
            grads_tensor = torch.cat(grads, dim=0)
            fisher = torch.einsum('bdi,bdj->bdij', grads_tensor, grads_tensor)
            return fisher.mean(dim=0)
        except RuntimeError:
            warnings.warn("Fisher matrix calculation failed, falling back to zeros")
            return torch.zeros(q.shape[-1], q.shape[-1])

    def _compute_lyapunov(self, q: Tensor, p: Tensor) -> float:
        """Calcula el máximo exponente de Lyapunov usando autodiff."""
        try:
            # Crear función para el cálculo de la divergencia KL
            def kl_fn(prior):
                return self._kl_stable(q, prior, as_tensor=True)
            
            # Calcular jacobiano de la función KL respecto al prior
            J = torch.autograd.functional.jacobian(kl_fn, p, create_graph=False)
            J_flat = J.flatten(start_dim=1)
            
            # Calcular autovalores (solo los mayores)
            eigvals = torch.linalg.eigvalsh(J_flat @ J_flat.T)
            max_eigval = eigvals.max().item()
            return max_eigval
        except Exception as e:
            warnings.warn(f"Lyapunov calculation failed: {str(e)}")
            return 0.0

    def _stochastic_entropy_production(self, delta_epist: float, w: float) -> float:
        """Calcula la entropía producida σ con fluctuaciones cuánticas."""
        beta = 1 / (self.quantum_temp + self.eps)
        quantum_fluct = torch.randn(1).item() * self.quantum_amp
        return delta_epist - beta * w + quantum_fluct

    def _quantum_pid(self, delta_epist: float, w: float) -> float:
        """Controlador PID con correcciones cuánticas."""
        target = self.pid_cfg["target"]
        err = (delta_epist / (w + self.eps)) - target
        self._pid_err_i += err
        derr = err - self._pid_prev
        self._pid_prev = err
        
        # Componente clásico
        P = self.pid_cfg["k_p"] * err
        I = self.pid_cfg["k_i"] * self._pid_err_i
        D = self.pid_cfg["k_d"] * derr
        
        # Corrección cuántica (oscilador armónico)
        quantum_correction = math.sin(time.time() * self.quantum_freq) * self.quantum_amp
        
        beta = self.pid_cfg["beta"] + P + I + D + quantum_correction
        return max(self.pid_cfg["min_beta"], min(beta, self.pid_cfg["max_beta"]))

    def _classic_pid(self, delta_epist: float, w: float) -> float:
        """Controlador PID clásico."""
        target = self.pid_cfg["target"]
        err = (delta_epist / (w + self.eps)) - target
        self._pid_err_i += err
        derr = err - self._pid_prev
        self._pid_prev = err
        
        beta = self.pid_cfg["beta"] + (
            self.pid_cfg["k_p"] * err +
            self.pid_cfg["k_i"] * self._pid_err_i +
            self.pid_cfg["k_d"] * derr
        )
        return max(self.pid_cfg["min_beta"], min(beta, self.pid_cfg["max_beta"]))
    # endregion

    # region ───────── HOMEOSTASIS & CRISIS MANAGEMENT ─────────
    def _update_physical_sensors(self):
        """Actualiza sensores físicos desde el sistema."""
        try:
            # Obtener uso de memoria
            self.memory_usage = psutil.virtual_memory().percent / 100.0
            
            # Obtener temperatura (GPU o CPU)
            if _GPU_HANDLE:
                temp = pynvml.nvmlDeviceGetTemperature(_GPU_HANDLE, pynvml.NVML_TEMPERATURE_GPU)
                self.temperature = temp / 100.0  # Normalizar
            else:
                self.temperature = 0.6  # Valor fijo si no hay sensor
        except Exception:
            self.memory_usage = 0.7
            self.temperature = 0.5

    def _check_homeostasis(self):
        """Verifica invariantes de homeostasis y dispara eventos."""
        # Memoria (Blueprint: Mem(t) < 0.95)
        if self.memory_usage > 0.93:
            self._events.append("MEMORY_THRESHOLD")
            event_bus.emit('episteme_memory_threshold', {
                'memory_usage': self.memory_usage,
                'timestamp': time.time()
            })
            if "prune_callback" in self._hooks:
                self._hooks["prune_callback"](self._snapshot(time.time(), torch.zeros(1,1), 0.0, 0.0, self.pid_cfg["beta"]))

        # Temperatura (Blueprint: Energy(t) < 1)
        if self.temperature > 0.95:
            self._events.append("THERMAL_CRISIS")
            event_bus.emit('episteme_thermal_crisis', {
                'temperature': self.temperature,
                'timestamp': time.time()
            })
            if "cooling_callback" in self._hooks:
                self._hooks["cooling_callback"](self._snapshot(time.time(), torch.zeros(1,1), 0.0, 0.0, self.pid_cfg["beta"]))

        # Estabilidad (Blueprint: λ_max(t) < 1000)
        if self.enable_lyapunov and self.last_lyapunov > MAX_LYAPUNOV:
            self._events.append("CHAOTIC_RESURRECTION_TRIGGER")
            event_bus.emit('episteme_chaotic_resurrection', {
                'lyapunov': self.last_lyapunov,
                'timestamp': time.time()
            })
            if "resurrection_callback" in self._hooks:
                self._hooks["resurrection_callback"](self._snapshot(time.time(), torch.zeros(1,1), 0.0, 0.0, self.pid_cfg["beta"]))

        # Límite de beta
        if self.pid_cfg["beta"] >= self.pid_cfg["max_beta"]:
            self._events.append("β_CAP_REACHED")
            event_bus.emit('episteme_beta_cap_reached', {
                'beta': self.pid_cfg["beta"],
                'timestamp': time.time()
            })

    def _save_critical_state(self, snap: EpistemeSnapshot):
        """Guarda estado crítico en memoria epistémica."""
        # Guardar si tiene los atributos requeridos, aunque sea un mock
        try:
            lyap = float(snap.lyapunov_exp)
            state = {
                "timestamp": float(snap.t_wall),
                "lyapunov": lyap,
                "entropy_prod": float(snap.entropy_production),
                "temperature": float(snap.temperature),
                "memory": float(snap.memory_usage),
                "events": tuple(snap.events)
            }
            self.epistemic_memory.append(state)
            # Mantener solo los 10 estados más significativos por Lyapunov
            self.epistemic_memory.sort(key=lambda x: x["lyapunov"], reverse=True)
            self.epistemic_memory = self.epistemic_memory[:10]
        except Exception:
            pass  # Ignorar si el objeto no tiene los atributos necesarios
    # endregion

    # region ───────── INTERNALS ─────────
    def _kl_stable(self, q: Tensor, p: Tensor, as_tensor: bool = False) -> float | Tensor:
        q = q + self.eps
        p = p + self.eps
        kl = torch.distributions.kl_divergence(
            distributions.Categorical(probs=q),
            distributions.Categorical(probs=p)
        )
        return kl if as_tensor else kl.mean().item()

    def _integrate(self, buf: Deque, val: float, tau: float, attr: str, dt: float):
        setattr(self, attr, getattr(self, attr) + val)
        buf.append((self._last_t, val))
        if tau > 0:
            while buf and (self._last_t - buf[0][0]) > tau:
                t_old, v_old = buf.popleft()
                setattr(self, attr, getattr(self, attr) - v_old)

    def _snapshot(self, t: float, fisher_matrix: Tensor, lyapunov: float, entropy_prod: float, beta: float) -> EpistemeSnapshot:
        eff = self._int_epist / max(self._int_power, self.eps)
        density = self._int_fisher / max(self._int_power, self.eps)
        
        return EpistemeSnapshot(
            t_wall=t,
            epist_int=self._int_epist,
            w_int=self._int_power,
            fisher_int=self._int_fisher,
            fisher_matrix=fisher_matrix if fisher_matrix is not None else torch.tensor(0.0),
            lyapunov_exp=lyapunov,
            entropy_production=entropy_prod,
            efficiency=eff,
            fisher_density=density,
            quantum_beta=beta,
            events=tuple(self._events),
            memory_usage=self.memory_usage,
            temperature=self.temperature
        )

    def _emit_hooks(self, snap: EpistemeSnapshot):
        for fn in self._hooks.values():
            try:
                fn(snap)
            except Exception as e:
                warnings.warn(f"Episteme hook failed: {str(e)}")

    def _reset_integrals(self):
        self._int_epist = self._int_power = self._int_fisher = 0.0
        self._buf_epist.clear()
        self._buf_power.clear()
        self._buf_fisher.clear()
    # endregion

    # region ───────── HOOKS & UTIL ─────────
    def register_hook(self, name: str, fn: Callable[[EpistemeSnapshot], None]):
        self._hooks[name] = fn

    def remove_hook(self, name: str):
        self._hooks.pop(name, None)
        
    def get_critical_states(self) -> List[Dict]:
        """Obtiene los estados críticos almacenados."""
        return self.epistemic_memory.copy()
    # endregion

# ─────────────────────────────────────────── POWER READING ───────────────────────────────────────────
def _read_power() -> float:
    """Lee la potencia disipada con fallback a CPU."""
    # Intentar leer de GPU primero
    if _GPU_HANDLE is not None:
        try:
            return pynvml.nvmlDeviceGetPowerUsage(_GPU_HANDLE) / 1000.0  # Convertir a Watts
        except Exception:
            pass
    
    # Fallback a CPU (heurística)
    cpu_percent = psutil.cpu_percent()
    return cpu_percent * 0.05  # Aproximación: 0.05W por % de CPU