# persistence.py (versión mejorada)
import os
import json
import time
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable
from threading import Lock
from datetime import datetime, timedelta
from contextlib import contextmanager
import numpy as np

# Configuración avanzada de logging
logger = logging.getLogger("Persistence::StateManager")
logger.setLevel(logging.INFO)

class StatePreserver:
    _lock = Lock()  # Bloqueo para operaciones seguras en multihilo
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el sistema de preservación de estado con validación de datos.
        
        Args:
            config: Diccionario con parámetros como 'backup_dir' y 'max_backups'.
        """
        self.backup_dir = Path(config.get("backup_dir", "backups"))
        self.max_backups = config.get("max_backups", 10)
        self.compression = config.get("compression", False)  # Nuevo: compresión de respaldos
        self.retention_days = config.get("retention_days", 7)  # Nuevo: retención por tiempo
        
        # Validar y crear directorio de respaldo
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[Persistence] Carpeta de respaldo: {self.backup_dir}")
            self._cleanup_old_backups()  # Limpiar archivos antiguos al iniciar
        except Exception as e:
            logger.error(f"[Persistence] No se pudo crear el directorio de respaldo: {e}")
            raise

    @contextmanager
    def atomic_write(self, file_path: Path):
        """Context manager para escritura atómica de archivos."""
        temp_file = file_path.with_suffix(file_path.suffix + ".tmp")
        try:
            with open(temp_file, "w") as f:
                yield f
            # Reemplazar archivo original solo si la escritura fue exitosa
            shutil.move(str(temp_file), str(file_path))
        except Exception as e:
            logger.error(f"[Persistence] Error en escritura atómica: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise

    @contextmanager
    def context(self, tag: str = "generic"):
        """
        Context manager para modificar y guardar automáticamente el estado bajo un tag.
        Al salir del contexto, guarda el estado actualizado.
        """
        state = self.load_latest_state(tag) or {}
        try:
            yield state
            self.save_state(state, tag=tag)
        except Exception as e:
            logger.error(f"[Persistence] Error en context manager: {e}")
            raise

    def save_state(self, data: Dict[str, Any], tag: str = "generic") -> bool:
        """
        Guarda un estado en disco con manejo atómico y validación de datos.
        
        Args:
            data: Diccionario con el estado a guardar.
            tag: Etiqueta para categorizar el estado (ej.: 'homeostasis', 'model').
        Returns:
            bool: True si se guardó correctamente.
        """
        try:
            # Validar datos antes de guardar
            if not isinstance(data, dict):
                logger.error("[Persistence] Datos inválidos: debe ser un diccionario")
                return False
                
            timestamp = int(time.time())
            file = self.backup_dir / f"{tag}_{timestamp}.json"
            
            with self._lock, self.atomic_write(file) as f:
                json.dump(data, f, indent=2, default=self._json_serializer)
                
            logger.info(f"[Persistence] Estado guardado en {file}")
            self._cleanup_old_backups()  # Limpiar después de guardar
            return True
            
        except Exception as e:
            logger.error(f"[Persistence] Error al guardar estado: {e}", exc_info=True)
            return False

    def load_latest_state(self, tag: str = "generic") -> Dict[str, Any]:
        """
        Carga el estado más reciente con validación de integridad.
        
        Args:
            tag: Etiqueta del estado a cargar.
        Returns:
            Dict con el estado o diccionario vacío si falla.
        """
        try:
            pattern = f"{tag}_*.json"
            backups = sorted(self.backup_dir.glob(pattern), reverse=True)
            
            if not backups:
                logger.warning(f"[Persistence] No se encontró respaldo para '{tag}'.")
                return {}
                
            latest = backups[0]
            
            with self._lock, open(latest, "r") as f:
                data = json.load(f)
                
            # Validar estructura básica del estado
            if not isinstance(data, dict):
                logger.error(f"[Persistence] Formato inválido en {latest}")
                return {}
                
            logger.info(f"[Persistence] Cargado estado desde {latest}")
            return data
            
        except Exception as e:
            logger.error(f"[Persistence] Error al cargar: {e}", exc_info=True)
            return {}

    def _json_serializer(self, obj: Any) -> Union[str, float, int, None]:
        """Serializador personalizado para tipos no estándar."""
        if isinstance(obj, (datetime, np.datetime64)):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        raise TypeError(f"Tipo {type(obj)} no serializable")

    def _cleanup_old_backups(self) -> None:
        """Limpia respaldos antiguos por cantidad y edad."""
        try:
            all_backups = list(self.backup_dir.glob("*.json"))
            
            # Eliminar por cantidad
            if len(all_backups) > self.max_backups:
                sorted_by_time = sorted(all_backups, key=lambda x: x.stat().st_mtime)
                for old in sorted_by_time[:-self.max_backups]:
                    old.unlink()
                    logger.info(f"[Persistence] Respaldo eliminado: {old}")
                    
            # Eliminar por retención en días
            cutoff = datetime.now() - timedelta(days=self.retention_days)
            for file in all_backups:
                if datetime.fromtimestamp(file.stat().st_mtime) < cutoff:
                    file.unlink()
                    logger.info(f"[Persistence] Respaldo antiguo eliminado: {file}")
                    
        except Exception as e:
            logger.error(f"[Persistence] Error en limpieza: {e}")

    def save_critical_state(self, state_data: Dict[str, Any]) -> bool:
        """
        Guarda un estado crítico con prioridad máxima.
        
        Args:
            state_data: Datos del estado crítico.
        Returns:
            bool: Resultado de la operación.
        """
        return self.save_state(state_data, tag="critical")

    def save_full_state(self, get_snapshot_func: Callable[[], Dict[str, Any]]) -> bool:
        """
        Guarda el estado completo del sistema sin dependencia circular.
        
        Args:
            get_snapshot_func: Función que devuelve el snapshot del sistema.
        Returns:
            bool: Resultado de la operación.
        """
        try:
            snapshot = get_snapshot_func()
            return self.save_state(snapshot, tag="full")
        except Exception as e:
            logger.error(f"[Persistence] Error en save_full_state: {e}")
            return False

    def verify_backup_integrity(self, file_path: Path) -> bool:
        """
        Verifica la integridad de un archivo de respaldo.
        
        Args:
            file_path: Ruta al archivo de respaldo.
        Returns:
            bool: True si es válido.
        """
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            return isinstance(data, dict) and bool(data)
        except Exception:
            return False

    def create_compressed_backup(self, source_dir: Path, output_zip: Path) -> bool:
        """
        Crea un respaldo comprimido del directorio de estados.
        
        Args:
            source_dir: Directorio a comprimir.
            output_zip: Ruta de salida del archivo ZIP.
        Returns:
            bool: Resultado de la operación.
        """
        if not self.compression:
            return True
            
        try:
            shutil.make_archive(str(output_zip), 'zip', source_dir)
            logger.info(f"[Persistence] Respaldo comprimido creado: {output_zip}")
            return True
        except Exception as e:
            logger.error(f"[Persistence] Error en compresión: {e}")
            return False