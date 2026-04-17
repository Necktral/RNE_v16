from typing import Callable, Dict, List, Tuple, Optional
from collections import defaultdict, deque
import os
import time
import signal
import warnings
import sys
from src.utils.influx_logger import InfluxLogger

class Hook:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, context: Dict[str, any]):
        raise NotImplementedError("Debe implementar __call__()")

class CompositeHook(Hook):
    def __init__(self, name: str, hooks: List[Hook]):
        super().__init__(name)
        self.hooks = hooks

    def __call__(self, context: Dict[str, any]):
        for hook in self.hooks:
            result = hook(context)
            if result == "abort":
                return "abort"

class TimeoutException(Exception):
    pass

def timeout(seconds: float):
    def decorator(fn):
        if not hasattr(sys, 'platform') or sys.platform.startswith('win'):
            # No-op en Windows
            return fn
        def handler(signum, frame):
            raise TimeoutException("Timeout en ejecución del hook")
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, handler)
            signal.setitimer(signal.ITIMER_REAL, seconds)
            try:
                return fn(*args, **kwargs)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
        return wrapper
    return decorator

class HookManager:
    def __init__(self, action_map: Optional[Dict[str, Callable]] = None, verbose: bool = True, influx_logger=None):
        self.action_map = action_map or {}
        self.dependencies = defaultdict(list)
        self.execution_stats = defaultdict(lambda: {"count": 0, "last_duration": 0.0, "history": deque(maxlen=20)})
        self.execution_policy = {}
        self.hook_timeouts = {}
        self.verbose = verbose
        self.influx_logger = influx_logger or InfluxLogger(
            url=os.environ.get("INFLUXDB_URL", "http://localhost:8181"),
            token=os.environ.get("INFLUXDB_TOKEN", "<TOKEN>"),
            database=os.environ.get("INFLUXDB_BUCKET", "aeon_metrics")
        )

    def register_action(self, name: str, fn: Callable, requires: Optional[List[str]] = None, policy: str = "abort", timeout_sec: Optional[float] = None):
        # Telemetría: log de registro de acción
        try:
            self.influx_logger.log_event(
                name="hook_register",
                tags={"hook_name": name, "policy": policy, "timeout_sec": str(timeout_sec)}
            )
        except Exception as e:
            pass
        self.action_map[name] = fn
        self.dependencies[name] = requires or []
        self.execution_policy[name] = policy
        if timeout_sec:
            self.hook_timeouts[name] = timeout_sec

    def resolve_execution_order(self, actions: List[Tuple[str, int]]) -> List[str]:
        seen = set()
        order = []

        def visit(hook: str):
            if hook in seen:
                return
            for dep in self.dependencies.get(hook, []):
                visit(dep)
            seen.add(hook)
            order.append(hook)

        for action, _ in actions:
            visit(action)

        return order

    def run(self, actions: List[Tuple[str, int]], context: Optional[dict] = None):
        if context is None:
            context = {}
        ordered = self.resolve_execution_order(actions)

        for action_name in ordered:
            fn = self.action_map.get(action_name)
            if not fn:
                warnings.warn(f"Acción desconocida: {action_name}")
                continue

            timeout_sec = self.hook_timeouts.get(action_name)
            wrapped_fn = timeout(timeout_sec)(fn) if timeout_sec else fn
            start = time.perf_counter()

            try:
                wrapped_fn(context)
                duration = time.perf_counter() - start
                stats = self.execution_stats[action_name]
                stats["count"] += 1
                stats["last_duration"] = duration
                stats["history"].append(duration)
                if self.verbose:
                    print(f"[HOOK] {action_name} ejecutado en {duration:.3f}s")
            except TimeoutException:
                warnings.warn(f"Timeout ejecutando {action_name}")
                if self.execution_policy.get(action_name) == "abort":
                    break
            except Exception as e:
                warnings.warn(f"Error ejecutando {action_name}: {str(e)}")
                if self.execution_policy.get(action_name) == "abort":
                    break
