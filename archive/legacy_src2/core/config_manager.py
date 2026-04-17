import os
import time
import structlog
from pathlib import Path
from ruamel.yaml import YAML
import hashlib
import sqlite3
import ast
from typing import Any, Dict, List

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def yaml_dump(obj) -> bytes:
    yaml = YAML()
    from io import StringIO
    buf = StringIO()
    yaml.dump(obj, buf)
    return buf.getvalue().encode('utf-8')

class ConfigManager:
    def version_and_write(self):
        # Dummy stub for version_and_write
        pass

    def rollback(self):
        # Dummy stub for rollback
        pass

    def evaluate_policies(self, cycle, stats):
        # Dummy stub for evaluate_policies
        # Returns (dirty, mutations)
        return False, []
    def __init__(self, cfg_path: str):
        self.cfg_path = Path(cfg_path).resolve()
        self.yaml = YAML()
        self.cfg = self._load_and_verify()
        self.mutations: List[Dict[str, Any]] = []
        self.dirty = False
        self.meta_path = self.cfg_path.parent
        self.logger = structlog.get_logger("ConfigManager")

    def _load_and_verify(self):
        with open(self.cfg_path, 'r', encoding='utf-8') as f:
            cfg = self.yaml.load(f)
        for block_name, block in cfg.items():
            digest = block.get("checksum")
            block_copy = dict(block)
            block_copy.pop("checksum", None)
            actual = sha256(yaml_dump(block_copy))
            if digest != actual:
                raise ValueError(f"Checksum mismatch: {block_name}\nEsperado: {digest}\nActual:   {actual}")
        return cfg

    def evaluate_policies(self, cycle: int, stats: Dict[str, float]):
        runtime_vars = {"cycle": cycle, **stats}
        for section_name, section in self.cfg.items():
            for rule in section.get("mutate_policy", []):
                if all(self._eval_trigger(cond, runtime_vars) for cond in rule["trigger"]):
                    old_section = dict(section)
                    self._exec_action(section, rule["action"], runtime_vars)
                    self.dirty = True
                    self.mutations.append({
                        "ts": time.time(),
                        "cycle": cycle,
                        "section": section_name,
                        "rule_json": str(rule),
                        "old_value": str(old_section),
                        "new_value": str(section),
                        "checksum_post": None # se actualiza tras versionado
                    })
        return self.dirty, self.mutations

    def _eval_trigger(self, expr: str, vars: Dict[str, Any]) -> bool:
        # Mini-DSL seguro: solo operadores lógicos y variables permitidas
        allowed_names = set(vars.keys())
        node = ast.parse(expr, mode='eval')
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id not in allowed_names:
                raise ValueError(f"Variable no permitida: {n.id}")
            if isinstance(n, (ast.Call, ast.Import, ast.ImportFrom, ast.Attribute)):
                raise ValueError("Operación no permitida en trigger")
        return eval(compile(node, '<trigger>', 'eval'), {}, vars)

    def _exec_action(self, section: dict, action: str, vars: Dict[str, Any]):
        # Solo permite asignaciones aritméticas simples
        node = ast.parse(action, mode='exec')
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id not in section:
                raise ValueError(f"Variable no permitida en action: {n.id}")
            if isinstance(n, (ast.Call, ast.Import, ast.ImportFrom, ast.Attribute)):
                raise ValueError("Operación no permitida en action")
        local = dict(section)
        exec(compile(node, '<action>', 'exec'), {}, local)
        section.update(local)

    def version_and_save(self):
        from shutil import copy2
        import sys, errno
        from datetime import datetime
        if not self.dirty:
            return None
        # Actualiza meta.version y meta.updated
        self.cfg["meta"]["version"] = float(self.cfg["meta"].get("version", 0)) + 0.01
        self.cfg["meta"]["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # Recalcula checksums
        for block_name, block in self.cfg.items():
            block_copy = dict(block)
            block_copy.pop("checksum", None)
            block["checksum"] = sha256(yaml_dump(block_copy))
        # Guarda snapshot atómico
        fname = f"config_{datetime.utcnow():%Y%m%d_%H%M%S}.yaml"
        final_path = self.meta_path / fname
        with open(final_path, 'w', encoding='utf-8') as f:
            self.yaml.dump(self.cfg, f)
        # Actualiza symlink/copia atómica
        symlink = self.meta_path / "config_current.yaml"
        try:
            if symlink.exists() or symlink.is_symlink():
                symlink.unlink()
            symlink.symlink_to(final_path.name)
        except OSError as e:
            # WinError 1314 (Windows) o EPERM/EACCES (POSIX)
            if (hasattr(e, 'winerror') and e.winerror == 1314) or (e.errno in {errno.EPERM, errno.EACCES}):
                tmp = symlink.with_suffix('.tmp')
                copy2(final_path, tmp)
                os.replace(tmp, symlink)
            else:
                raise
        # Actualiza checksum_post en mutaciones
        for mut in self.mutations:
            mut["checksum_post"] = self.cfg[mut["section"]]["checksum"]
        self.logger.info("Config versioned", path=str(final_path))
        self.dirty = False
        return final_path

    def yield_mutations(self):
        for mut in self.mutations:
            yield mut
        self.mutations.clear()

    @property
    def current_cfg(self):
        return self.cfg

# SQLiteRepo para registrar mutaciones
class SQLiteRepo:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_table()
    def _init_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS mutations (
                ts REAL, cycle INT, section TEXT, rule_json TEXT,
                old_value TEXT, new_value TEXT, checksum_post TEXT
            )""")
    def insert_mutation(self, mut: Dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            INSERT INTO mutations (ts, cycle, section, rule_json, old_value, new_value, checksum_post)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                mut["ts"], mut["cycle"], mut["section"], mut["rule_json"],
                mut["old_value"], mut["new_value"], mut["checksum_post"]
            ))

# Ejemplo de integración en main loop
if __name__ == "__main__":
    cfg_manager = ConfigManager("config/config.yaml")
    repo = SQLiteRepo("data/metacognition.db")
    total_steps = 1000
    for cycle in range(total_steps):
        # Simula métricas runtime
        stats = {"cycle": cycle, "energy": 0.3, "ewma_success": 0.5, "surprise_spike": 1.2, "loss_plateau": 100}
        cfg_manager.evaluate_policies(cycle, stats)
        path = cfg_manager.version_and_save()
        for mut in cfg_manager.yield_mutations():
            repo.insert_mutation(mut)
            cfg_manager.logger.info("cfg_mutation", **mut)
        cfg = cfg_manager.current_cfg
        # ... resto del ciclo ...
