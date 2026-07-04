import streamlit as st
import threading
import queue
import time
from datetime import datetime
import json
import os

# Importa el EventBus global de AEON FENIX-Δ
try:
    from runtime.core.event_bus import event_bus
except ImportError:
    event_bus = None  # Para pruebas sin AEON

# Ruta del log persistente
EVENT_LOG_PATH = os.environ.get("AEON_EVENT_LOG", "aeon_event_log.jsonl")

# Lista de eventos críticos a monitorear
CRITICAL_EVENTS = [
    'crisis', 'homeostasis_violation', 'quantum_mutation_triggered',
    'quantum_mutation_applied', 'module_spawned', 'module_pruned',
    'meta_optimizer_step', 'quantum_reorganization', 'thermal_throttling',
    'cognitive_challenge', 'chaotic_resurrection', 'beta_cap',
    'quarantine', 'recovery', 'quantum_pruning',
]

def read_events_from_log(max_events=200):
    events = []
    if os.path.exists(EVENT_LOG_PATH):
        with open(EVENT_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                    events.append(ev)
                except Exception:
                    continue
    return events[-max_events:]

# Streamlit UI
st.set_page_config(page_title="AEON FENIX-Δ Dashboard", layout="wide")
st.title("🦾 AEON FENIX-Δ – Event Monitoring Dashboard")
st.markdown("""
**Estado en tiempo real de eventos críticos, módulos y métricas del sistema AGI.**
""")

# Sidebar: Filtros y controles
event_filter = st.sidebar.multiselect(
    "Filtrar eventos", CRITICAL_EVENTS, default=CRITICAL_EVENTS
)
max_events = st.sidebar.slider("Máx. eventos recientes", 10, 200, 50)

# Main: Panel de eventos recientes
events = read_events_from_log(max_events)

if events:
    st.subheader(f"Eventos recientes ({len(events)})")
    for ev in reversed(events):
        if ev['event'] in event_filter:
            with st.expander(f"[{ev.get('timestamp', '-')}] {ev['event']}", expanded=False):
                st.json(ev['payload'])
else:
    st.info("Esperando eventos...")

# Panel de métricas clave (ejemplo: ciclo, carga cognitiva, módulos activos)
if events:
    last_step = next((e for e in reversed(events) if e['event'] == 'meta_optimizer_step'), None)
    if last_step:
        st.metric("Ciclo actual", last_step['payload'].get('cycle', '-'))
        st.metric("Carga cognitiva", last_step['payload'].get('cognitive_load', '-'))
    modules_spawned = [e for e in events if e['event'] == 'module_spawned']
    modules_pruned = [e for e in events if e['event'] == 'module_pruned']
    st.metric("Módulos activos (estimado)", max(0, len(modules_spawned) - len(modules_pruned)))

# Panel de logs críticos
def render_log_panel(events):
    crisis_events = [e for e in events if e['event'] == 'crisis']
    if crisis_events:
        st.error(f"⚠️ Crisis detectada: {crisis_events[-1]['payload']}")
    violations = [e for e in events if e['event'] == 'homeostasis_violation']
    if violations:
        st.warning(f"Violaciones homeostáticas: {len(violations)} (última: ciclo {violations[-1]['payload'].get('cycle', '-')})")
    mutations = [e for e in events if e['event'] == 'quantum_mutation_applied']
    if mutations:
        st.info(f"Mutaciones cuánticas: {len(mutations)} (última: {mutations[-1]['payload'].get('reason', '-')})")

render_log_panel(events)

st.markdown("---")
st.caption("AEON FENIX-Δ Monitoring Dashboard | Streamlit | 2025")
