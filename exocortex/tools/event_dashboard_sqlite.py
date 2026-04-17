import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
from streamlit_extras.badges import badge
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
from src.core.event_log_sqlite import EventLogSQLite

CRITICAL_EVENTS = [
    'crisis', 'homeostasis_violation', 'quantum_mutation_triggered',
    'quantum_mutation_applied', 'module_spawned', 'module_pruned',
    'meta_optimizer_step', 'quantum_reorganization', 'thermal_throttling',
    'cognitive_challenge', 'chaotic_resurrection', 'beta_cap',
    'quarantine', 'recovery', 'quantum_pruning',
]

default_db_path = "aeon_event_log.db"
db_path = os.environ.get("AEON_EVENT_DB", default_db_path)
event_log = EventLogSQLite(db_path)

st.set_page_config(page_title="AEON FENIX-Δ Dashboard (SQLite)", layout="wide", page_icon="🦾")
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #232526 0%, #414345 100%);
        color: #F8F8F2;
    }
    .stApp {
        background: linear-gradient(135deg, #232526 0%, #414345 100%);
    }
    .st-bb, .st-c6, .st-cg, .st-cj, .st-cq {
        background: #232526 !important;
        color: #F8F8F2 !important;
    }
    .stMetric {
        background: #282a36;
        border-radius: 8px;
        padding: 8px 16px;
        margin: 4px;
        color: #50fa7b;
    }
    .stExpanderHeader {
        font-weight: bold;
        color: #8be9fd;
    }
</style>
""", unsafe_allow_html=True)

st.title("🦾 AEON FENIX-Δ – Event Monitoring Dashboard (SQLite)")
st.markdown("""
<div style='font-size:1.2em; color:#8be9fd;'>
<b>Estado en tiempo real de eventos críticos, módulos y métricas del sistema AGI.</b>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2721/2721297.png", width=80)
    st.markdown("<h3 style='color:#50fa7b;'>Filtros y controles</h3>", unsafe_allow_html=True)
    event_filter = st.multiselect(
        "Filtrar eventos", CRITICAL_EVENTS, default=CRITICAL_EVENTS
    )
    max_events = st.slider("Máx. eventos recientes", 10, 200, 50)
    st.markdown("<hr>", unsafe_allow_html=True)
    badge(type="github", name="neckt-dev/aeon_fenix_delta")

# Main: Panel de eventos recientes
events = event_log.get_events(limit=max_events, event_types=event_filter)

if events:
    st.markdown(f"📝 <span style='color:#ffb86c'>Eventos recientes</span> <span style='font-size:0.8em;color:#bd93f9'>({len(events)})</span>", unsafe_allow_html=True)
    for ev in reversed(events):
        if ev['event'] in event_filter:
            color = "#50fa7b" if "cognitive" in ev['event'] else ("#ff5555" if "crisis" in ev['event'] else "#8be9fd")
            with stylable_container(key=f"exp_{ev['event']}_{ev['timestamp']}", css_styles=f"background: #232526; border-left: 6px solid {color}; border-radius: 8px; margin-bottom: 8px;"):
                with st.expander(f"[{ev.get('timestamp', '-')}] {ev['event']}", expanded=False):
                    st.json(ev['payload'])
else:
    st.info("Esperando eventos...")

# Panel de métricas clave
def metric_card(label, value, icon=None, color="#50fa7b"):
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:8px;background:#282a36;border-radius:8px;padding:8px 16px;margin:4px;'>
        <span style='font-size:1.3em;'>{icon or ''}</span>
        <span style='color:{color};font-weight:bold;font-size:1.1em'>{label}:</span>
        <span style='color:#f8f8f2;font-size:1.1em'>{value}</span>
    </div>
    """, unsafe_allow_html=True)

if events:
    last_step = next((e for e in reversed(events) if e['event'] == 'meta_optimizer_step'), None)
    if last_step:
        metric_card("Ciclo actual", last_step['payload'].get('cycle', '-'), icon="🔄")
        metric_card("Carga cognitiva", last_step['payload'].get('cognitive_load', '-'), icon="🧠", color="#8be9fd")
    modules_spawned = [e for e in events if e['event'] == 'module_spawned']
    modules_pruned = [e for e in events if e['event'] == 'module_pruned']
    metric_card("Módulos activos (estimado)", max(0, len(modules_spawned) - len(modules_pruned)), icon="🧩", color="#bd93f9")

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

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("AEON FENIX-Δ Monitoring Dashboard | <span style='color:#50fa7b'>Streamlit + SQLite</span> | 2025", unsafe_allow_html=True)
