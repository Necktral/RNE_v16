
# --- Clase moderna para InfluxDB OSS 3.x y Cloud ---
import time
from typing import Optional, Dict
try:
    from influxdb_client_3 import InfluxDBClient3, Point as Point3
    _INFLUXDB_CLIENT3_AVAILABLE = True
except ImportError:
    _INFLUXDB_CLIENT3_AVAILABLE = False

class InfluxLogger:
    """
    Logger OO para InfluxDB OSS 3.x/Cloud usando influxdb_client_3 (si está disponible).
    Compatible con tests de tipo unittest/pytest.
    """
    def __init__(self, url: str, token: str, database: str):
        if not _INFLUXDB_CLIENT3_AVAILABLE:
            raise ImportError("influxdb_client_3 no está instalado")
        self.client = InfluxDBClient3(host=url, token=token, database=database)

    def _current_ns_timestamp(self) -> int:
        return int(time.time() * 1e9)

    def log_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, timestamp: Optional[int] = None):
        if timestamp is None:
            timestamp = self._current_ns_timestamp()
        point = Point3(name).field("value", value).time(timestamp)
        if tags:
            for key, val in tags.items():
                point.tag(key, val)
        self.client.write(point)

    def log_event(self, name: str, tags: Optional[Dict[str, str]] = None, timestamp: Optional[int] = None):
        if timestamp is None:
            timestamp = self._current_ns_timestamp()
        point = Point3("event").tag("name", name).time(timestamp)
        if tags:
            for key, val in tags.items():
                point.tag(key, val)
        self.client.write(point)

    def close(self):
        self.client.close()
"""
Módulo utilitario para logging de métricas y eventos en InfluxDB.
Requiere: pip install influxdb-client
"""
import os
import logging
from typing import Optional, Dict, Any
try:
    from influxdb_client.client.influxdb_client import InfluxDBClient
    from influxdb_client.client.write.point import Point
    from influxdb_client.client.write_api import WriteOptions, SYNCHRONOUS
    _INFLUXDB_CLIENT_AVAILABLE = True
except ImportError:
    _INFLUXDB_CLIENT_AVAILABLE = False
import requests
import time
import os
import logging


INFLUXDB_URL = os.environ.get("INFLUXDB_URL", "http://localhost:8181")
INFLUXDB_TOKEN = os.environ.get("INFLUXDB_TOKEN", "<TOKEN>")
INFLUXDB_ORG = os.environ.get("INFLUXDB_ORG", "<ORG>")
INFLUXDB_BUCKET = os.environ.get("INFLUXDB_BUCKET", "aeon_metrics")

_logger = logging.getLogger("influx_logger")

_client = None
_write_api = None

def _init_client():
    global _client, _write_api
    if not _INFLUXDB_CLIENT_AVAILABLE:
        return False
    if _client is None or _write_api is None:
        try:
            _client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
            _write_api = _client.write_api(write_options=SYNCHRONOUS)
        except Exception as e:
            _logger.error(f"Error inicializando InfluxDBClient: {e}")
            _client = None
            _write_api = None
            return False
    return True

def _send_line_protocol(measurement: str, fields: dict, tags: Optional[Dict[str, Any]] = None, timestamp=None):
    tags = tags or {}
    tag_str = ",".join(f"{k}={v}" for k, v in tags.items())
    field_str = ",".join(f"{k}={v}" for k, v in fields.items())
    timestamp_ns = int(timestamp) if timestamp else int(time.time() * 1e9)
    line = f"{measurement}"
    if tag_str:
        line += f",{tag_str}"
    line += f" {field_str} {timestamp_ns}"
    headers = {
        "Authorization": f"Bearer {INFLUXDB_TOKEN}",
        "Content-Type": "text/plain"
    }
    try:
        resp = requests.post(
            f"{INFLUXDB_URL}/api/v3/write_lp?db={INFLUXDB_BUCKET}",
            headers=headers,
            data=line,
            timeout=3.0
        )
        resp.raise_for_status()
    except Exception as e:
        _logger.error(f"[HTTP Fallback] Error enviando métrica a InfluxDB: {e}")



def log_metric(measurement: str, fields: dict, tags: Optional[Dict[str, Any]] = None, timestamp=None):
    """
    Envía una métrica a InfluxDB. Usa el cliente oficial si está disponible, si no, fallback a HTTP/line protocol.
    :param measurement: Nombre de la métrica (measurement)
    :param fields: Diccionario de campos numéricos
    :param tags: Diccionario de tags (opcional)
    :param timestamp: Timestamp en segundos o nanosegundos (opcional)
    """
    if _INFLUXDB_CLIENT_AVAILABLE and _init_client() and _write_api is not None:
        point = Point(measurement)
        if tags:
            for k, v in tags.items():
                point = point.tag(k, v)
        for k, v in fields.items():
            point = point.field(k, v)
        if timestamp:
            point = point.time(timestamp)
        try:
            _write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
            return
        except Exception as e:
            _logger.error(f"[influxdb-client] Error enviando métrica a InfluxDB: {e}")
    # Fallback HTTP
    _send_line_protocol(measurement, fields, tags, timestamp)


def log_event(event_type: str, details: dict, tags: Optional[Dict[str, Any]] = None, timestamp=None):
    """
    Envía un evento a InfluxDB como measurement 'aeon_event'.
    :param event_type: Tipo de evento
    :param details: Diccionario de detalles del evento
    :param tags: Diccionario de tags (opcional)
    :param timestamp: Timestamp en segundos o nanosegundos (opcional)
    """
    fields = {"event_type": event_type}
    fields.update(details)
    log_metric("aeon_event", fields, tags, timestamp)
