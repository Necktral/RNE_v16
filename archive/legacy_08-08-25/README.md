# AEON-G: Sistema de Cognición Artificial Modular

**Autor:** Equipo AEON-G
**Fecha:** Agosto 2025

---

## Descripción General
AEON-G es una plataforma de cognición artificial modular, diseñada para experimentación avanzada en neurociencia computacional, aprendizaje profundo y sistemas adaptativos. El sistema integra componentes evolutivos, homeostáticos, de planificación y metacognición, permitiendo la simulación y entrenamiento de agentes inteligentes en entornos complejos.

## Arquitectura del Proyecto
```
├── aeon/
│   ├── components/        # Capas y bloques funcionales (chunking, SSM, etc.)
│   ├── core/              # Núcleo de gestión, eventos, entrenamiento y configuración
│   ├── data/              # Cargadores, normalizadores y utilidades de datos
│   ├── models/            # Modelos principales (HNet, etc.)
│   ├── systems/           # Subsistemas: evolución, homeostasis, cognición, etc.
│   ├── utils/             # Utilidades generales y logging
├── orchestrator/          # Orquestador principal y ciclo de entrenamiento
├── scripts/               # Scripts de preparación y entrenamiento
├── configs/               # Configuraciones de modelos y experimentos
├── notebooks/             # Cuadernos de pruebas y exploración
├── tests/                 # Pruebas unitarias
```

## Módulos Principales
- **aeon/components/**: Implementa capas especializadas como DynamicChunkingLayer y SSMBlock.
- **aeon/core/**: Incluye el gestor de configuración, bus de eventos, ciclo de entrenamiento y modelos probabilísticos.
- **aeon/data/**: Proporciona cargadores y normalizadores para distintos formatos de datos.
- **aeon/models/**: Contiene arquitecturas de red como HNet.
- **aeon/systems/**: Agrupa subsistemas evolutivos, cognitivos y de homeostasis.
- **aeon/utils/**: Herramientas de logging, métricas y utilidades generales.
- **orchestrator/**: Controla el flujo principal, integración de módulos y entrenamiento.

## Guía Rápida de Uso
1. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepara los datos:
   ```bash
   python scripts/prepare_data.py
   ```
3. Ejecuta el entrenamiento:
   ```bash
   python orchestrator/run.py --config configs/hnet_760m_config.py
   ```

## Documentación Técnica
Cada módulo y clase incluye docstrings detallados en español técnico. Consulta los archivos fuente para explicaciones de parámetros, retornos y ejemplos de uso.

## ULM (Universal Language Model)
El archivo `ULM_AEON-G.md` contiene una referencia rápida de todas las clases, funciones y flujos principales del sistema.

---

**Contacto:** aeon-g@proyecto.org
