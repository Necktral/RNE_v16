---
date: 2026-07-12
description: "Estado vivo del conectoma neural-simbólico, su capa tecnológica y los cinco agentes no autoritativos que auditan integración, riesgo y sinergia."
status: active
tags:
  - rnfe
  - neural
  - connectome
---

# RNFE v16 — Conectoma neural-simbólico

## Contexto

La campaña continúa bajo [[North Star]] y A-M0: primero se construye la
conectómica del organismo; Mamba2, H-Net, N6 entrenado y otras tecnologías se
integrarán después mediante puertos explícitos.

## Implementación

- Rama: `integration/connectomic-organism-v1`.
- Base: `main@b56562fb8cb852eac00466c0b10361c01b24cb93`.
- Contratos: topología `rnfe-connectome-v1`, actividad
  `rnfe-connectome-activity-v1`, plasticidad propuesta
  `rnfe-connectome-plasticity-v1` y checkpoint `rnfe-connectome-checkpoint-v1`.
- La actividad sólo nace de candidatos hasheados y `ConsumerReceipt` validados.
- CAU, CTF y C-GWM aparecen como autoridades causales separadas.
- La plasticidad es acotada, deduplicada y no aplicable; no muta el grafo.
- `off` no añade conectómica a la traza viva.
- No se modificaron `runtime/world`, razonamiento, memoria, certificación ni
  autoevolución; se conserva el editor único del sustrato.

## Integración pendiente coordinada

El coordinador expone exportación/restauración del ledger conectómico. Su conexión
al checkpoint soberano corresponde a una ventana con el dueño del sustrato. La
traza conectómica ya queda anclada transitivamente por el
`symbiosis_trace_hash` de la transición vital.

## Capa tecnológica — 2026-07-12

La rama `integration/neural-tech-connectomic-v1` compone sin conflictos los
commits de Claude Code `6c3fad3` (Mamba2 lazy/Turing) y `6ced3b6` (compatibilidad
H-Net/FlashAttention) con el conectoma. Sus 47 pruebas de engine pasaron.

- H-Net: upstream exacto `3ae01de...`, MIT vendorizada; se corrigió la frontera
  nativa a `split_offset`. Loader/trainer N5 listos, artefacto aún ausente porque
  el gate físico rechazó entrenar con otro proceso ocupando 6.38 GiB VRAM.
- Mamba2/N3: SSD-minimal CPU entrenado en laboratorio, 2 631 parámetros,
  pérdida 0.673→0.561; ejecución shadow end-to-end verificada.
- N1: MLP lab de 580 parámetros; ECE 0.316 obliga abstención/no promoción.
- N4: message-passing tipado lab de 64 parámetros; no aprende topología.
- N6: consume propuestas plásticas elegibles, excluye sus propias aristas y no
  dispone de `apply_fn`.

Los artefactos lab locales viven bajo `rnfe_artifacts/neural/` y están ignorados
por Git. Véase `governance/adr/ADR_NEURAL_TRAINED_BINDINGS.md`.

Ver [[RNFE v16 Project Memory]] y
`governance/adr/ADR_NEURAL_CONNECTOME.md`.

## Capa de cinco agentes — 2026-07-14

La rama `codex/neural-agent-suite-v1` añade orquestación, conectómica, comunicación
latente, adversarial y simbiosis/sinergia sobre la evidencia existente. El ciclo es
determinista, hasheado, no autoritativo y se expone en certificación; no duplica el
scheduler ni modifica el conectoma.

El ciclo y todos sus reportes declaran `experimental=true`. La comunicación latente
materializa por separado medición, clasificación, análisis y deliberación; por sí
sola no entrena ni se aplica. Sus límites son una envolvente de seguridad, no un
setpoint cognitivo aprendido.

La auditoría externa entregada por el humano corrigió dos afirmaciones fuertes:

- N4 ahora reemplaza su propuesta preliminar por una ligada a la intervención
  comprometida antes de emitir el recibo causal.
- La plasticidad sólo aprende de veredictos `accepted` o negativos explícitos;
  observación, comparación, abstención e indisponibilidad son neutrales.
- La comunicación latente rápida **no usa Mamba2**. Mamba2 es una alternativa
  experimental para memoria y creatividad horizontal en `SHADOW`, no el canal
  canónico de coordinación.
- La misión integra familias de razonamiento + entrenamiento neuronal + guía de un
  7B local para comprensión y mejora cognitiva de un ser cibernético. El mismo 7B
  tiene dos contratos que no deben mezclarse: razonador caro tier 3 y maestro
  post-experiencia. Ambos guían y critican; no sustituyen la deliberación ni la
  autoridad del organismo.

Véase `governance/adr/ADR_NEURAL_AGENT_SUITE.md` y
[[RNFE v16 Project Memory]].

### Primera extensión especializada

El agente `metacognitive_epistemic` conecta META/PROB/CAU/CTF con la capa de
agentes. Mide cobertura y certeza comprometida, clasifica conflicto o ausencia y
delibera propuestas no autoritativas. La ganancia permanece explícitamente
`unmeasured_pre_outcome`; no convierte una posterior en comprensión demostrada.

El agente `memory_consolidation` conecta la recuperación MFM con N3/N5 y separa
memoria trazable, duplicada, sin procedencia o cross-scenario. Nunca escribe memoria:
propone cuarentena, deduplicación o paso posterior por el gate certificado.

El agente `pedagogical_teacher` observa el ciclo final y compara el daño posterior
con el golpe que originó la lección. El maestro dejó de fabricar su recomendación como
éxito certificado; una preferencia del 7B sigue siendo hipótesis hasta repetirse con
mejora medible.

La primera ronda de diez tareas queda conectada mediante el bundle especializado:
metacognición, memoria, pedagogía, inmunidad, currículo, sensoriomotor/world-model,
metabolismo/MSRC, desarrollo/linaje, creatividad horizontal y social/exocórtex.
El currículo compara control, 7B local y docente Codex; sin pares por semilla no
declara eficiencia. Mamba2 sigue siendo alternativa shadow para experimentos, no
comunicación rápida canónica.

Ensayo físico inicial del 7B en RTX 2070 Max-Q: 6.54 s de proceso y 31.8 tokens/s
de generación. Cumplió JSON pero falló los tres criterios semánticos docentes
(`avoid`, `prefer`, contenido de la lección), por lo que no se admite como maestro
autónomo. Ver `docs/analysis/21_TEACHER_7B_EFFICIENCY.md`.

Campaña avanzada lanzada: 3 escenarios × 3 seeds × 3 variantes, temperatura 0.25
y horizonte 3. La guía estructurada mejora reward y severidad frente al control en
los tres escenarios, pero 7B sólo logra 66.7% de calidad semántica. Codex alcanza
100%; ambos causan el mismo cambio porque convergen en el mismo `avoid/prefer`.
Dictamen: 7B permanece alumno supervisado; cero promoción o entrenamiento todavía.

Ola held-out ejecutada: 180 trials, 60 inferencias 7B y 1,080 episodios, horizonte
5. El exportador curricular construyó tres registros Codex→7B pero los rechazó:
ninguna acción fija domina al control en las dos perturbaciones. El siguiente
contrato debe enseñar abstención y solicitud de medición/foresight cuando reward y
severidad entran en tradeoff. No se entrena con promedios que oculten regresiones.
