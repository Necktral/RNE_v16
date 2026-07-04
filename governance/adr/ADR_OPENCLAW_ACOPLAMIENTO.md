---
title: ADR_OPENCLAW_ACOPLAMIENTO
status: normative
version: 1.0.0
date: 2026-03-17
owner: Wis
type: architecture-decision-record
subject: Acoplamiento de OpenClaw como exocorteza operativa de RNFE
---

# ADR — Acoplamiento de OpenClaw como exocorteza operativa

## 1. Contexto

Existe dentro del corpus una base externa relevante: `openclaw-main.zip`. La inspección del ZIP muestra un runtime de gran tamaño y madurez operativa, con fuerte densidad en `src/`, `extensions/`, `apps/`, `docs/`, `ui/` y `skills/`, y con predominio de TypeScript, seguido por Swift y Kotlin. La arquitectura observada es claramente de plataforma operativa y no de núcleo cognitivo.

La decisión que este ADR resuelve es: **cómo reutilizar OpenClaw sin destruir la ontología de RNFE**.

## 2. Decisión

OpenClaw será tratado como **exocorteza operativa / shell de ejecución / superficie de producto**.

No será tratado como:

- núcleo cognitivo,
- fuente de identidad del organismo,
- sustituto de SMG/LOT-F/C-GWM,
- mecanismo de herencia,
- definición de memoria viva.

## 3. Justificación

### 3.1 Qué aporta OpenClaw realmente

OpenClaw resuelve bien problemas de:

- gateway y control plane;
- canales y sesiones;
- plugins y skills;
- nodos periféricos/dispositivos;
- sandboxing operativo;
- apps y superficie multicanal;
- integración de herramientas.

### 3.2 Qué no resuelve

No resuelve por sí mismo:

- signos internos estables;
- formalización LOT-F;
- world model causal;
- continuidad identitaria;
- herencia certificada;
- linajes;
- criterio de vida del organismo.

### 3.3 Riesgo de mala integración

Si se injerta OpenClaw como “cerebro”, RNFE degenera en un asistente multicanal con sesiones, no en un organismo cibernético con viabilidad, identidad y herencia.

## 4. Arquitectura de frontera

### 4.1 Dentro del núcleo RNFE

Permanece exclusivamente en RNFE:

- F–M–S
- SMG
- LOT-F
- C-GWM
- MFM/VFD
- OMG/certificados
- Hctrl/MRO/Edge
- scheduler de razonamientos
- agentes cognitivos
- S-I-E
- lineages/meta

### 4.2 En la exocorteza OpenClaw

Se permite reutilizar o inspirarse en OpenClaw para:

- gateway
- transporte de eventos
- routing de canales
- nodes periféricos
- skills/plugins
- sandboxing
- apps móviles/desktop/web
- ejecución de herramientas
- superficie de producto y observabilidad periférica

## 5. Principio de no contaminación ontológica

La exocorteza puede exponer al organismo, pero no definirlo.

Esto implica:

1. la memoria de OpenClaw no sustituye MFM/OMG;
2. las sesiones de OpenClaw no sustituyen episodios certificados;
3. bootstrap files/context injection no sustituyen identidad operativa;
4. multi-agent routing no sustituye ecología de razón ni herencia certificada.

## 6. Patrón de acoplamiento aprobado

### 6.1 Patrón

`OpenClaw Gateway/Nodes/Apps  <->  RNFE Adapter Layer  <->  RNFE Core`

### 6.2 Adapter Layer obligatoria

La capa adaptadora debe desacoplar:

- protocolos de eventos;
- artefactos de memoria;
- identidad de agente vs identidad del organismo;
- tools/plugins vs acciones internas RNFE;
- canales externos vs contextos internos certificados.

### 6.3 Contratos requeridos

La capa adaptadora debe definir al menos:

- `event.schema`
- `tool_request.schema`
- `tool_result.schema`
- `session_bridge.schema`
- `episode_export.schema`
- `telemetry_bridge.schema`
- `safety_policy.schema`

## 7. Modo de adopción por etapas

### Etapa O1 — Observación e ingeniería inversa

- mapear módulos de gateway, nodes, plugins, skills, sandbox y apps;
- identificar piezas reutilizables sin contaminación del núcleo.

### Etapa O2 — Prototipo de adapter

- puente mínimo de eventos;
- tool execution desacoplada;
- telemetry bridge;
- session-to-episode bridge de solo lectura.

### Etapa O3 — Integración controlada

- OpenClaw consume salidas de RNFE y ejecuta acciones periféricas;
- RNFE no depende vitalmente de OpenClaw para su coherencia interna.

### Etapa O4 — Producto

- montar una o más superficies de valor sobre la exocorteza;
- mantener capacidad de reemplazar OpenClaw sin reescribir el núcleo RNFE.

## 8. Reglas de seguridad

1. tools externas corren sandboxed;
2. la exocorteza no puede escribir directo sobre memoria viva del núcleo;
3. propuestas estructurales originadas desde fuera entran siempre como `proposal`, nunca como `commit`;
4. OpenClaw puede detonar acciones, pero la aceptación persistente sigue bajo S-I-E.

## 9. Ventajas esperadas

- acelerar superficie operativa sin distraer recursos del núcleo;
- obtener canales, apps y toolchain más rápido;
- desacoplar producto de cognición profunda;
- reducir costo de ingeniería periférica.

## 10. Costos y riesgos

### Costos

- mantener adapter layer;
- mapear protocolos;
- evitar acoplamiento accidental;
- gobernar seguridad de tools/plugins.

### Riesgos

- que la interfaz domine la ontología;
- que sesiones/context injection suplanten episodios certificados;
- que la facilidad de plugins incentive arquitectura superficial;
- que el proyecto derive hacia “assistant platform” y abandone el nacimiento cognitivo.

## 11. Decisión final

**Se aprueba reutilizar OpenClaw solo como exocorteza operativa y nunca como fundamento del organismo.**

Cualquier intento de mover funciones de identidad, memoria viva, herencia, world model o razonamiento nuclear hacia OpenClaw requerirá un ADR nuevo y una justificación extraordinaria.

## 12. Consecuencia práctica inmediata

La siguiente acción correcta no es migrar RNFE a OpenClaw. La siguiente acción correcta es:

1. terminar el núcleo vivo mínimo RNFE;
2. levantar la adapter layer mínima;
3. usar OpenClaw como shell multicanal y entorno de tools cuando el núcleo ya pueda sostener identidad y certificados.
