---
title: ADR_NEURAL_AGENT_SUITE
status: experimental
date: 2026-07-15
---

# Capa de agentes neurales RNFE v1

## Decisión

RNFE incorpora cinco agentes explícitos sobre la frontera simbiótica existente:
orquestación, conectómica, comunicación latente, adversarial y simbiosis/sinergia.
Son observadores y coordinadores deterministas; no reemplazan `N0`, el scheduler,
la certificación, el conectoma ni los adaptadores N1–N6.

Todo el ciclo declara `experimental=true`. La comunicación latente materializa las
cuatro etapas `measure → classify → analyze → deliberate` como artefactos separados;
enumerarlas no basta. Deliberar una propuesta no autoriza aplicarla ni usarla como
dato de entrenamiento.

El ciclo se ejecuta sobre `OrganTrace`, `ConsumerReceipt` y el snapshot conectómico
ya validados. Su salida se adjunta como metadata aditiva al bloque neural de
certificación bajo `rnfe-neural-agent-cycle-v1`.

La versión v1 corre antes del certificado y reward finales. Por diseño sólo afirma
integridad, conectividad y deliberación experimental del episodio; no afirma ganancia
cognitiva ni eficacia docente. Esa atribución requiere un consumidor longitudinal
post-outcome separado.

## Extensiones especializadas

El ciclo de cinco roles permanece cerrado bajo `rnfe-neural-agent-cycle-v1`. Los
agentes posteriores usan `rnfe-neural-agent-extensions-v1`: comparten identidad y
contrato de reporte, pero no alteran la cardinalidad ni el hash del ciclo base.

La primera extensión es `metacognitive_epistemic`. Consume directamente la salida
viva de META (`sequence`, `sequence_validation`, estado PROB/CAU/CTF), además de
trazas y recibos neuronales. Clasifica únicamente estados observables:
`unmeasured`, `measured_conflicted` o `measured_consistent`; no usa umbrales de salud
soldados. `prob_lcb` se registra como certeza comprometida, mientras la ganancia
epistémica queda `unmeasured_pre_outcome`. Puede proponer PROB, crítica adversarial o
consulta tier 3, pero no autoriza invocación ni cambia el scheduler.

La segunda extensión es `memory_consolidation`. Consume hits recuperados, candidatos
N3/N5 y recibos. Mide procedencia, duplicación y cruces de escenario; una memoria sin
ID o sin compatibilidad se propone para cuarentena, nunca para promoción. Las salidas
son candidatas auditables y conservan `writes_memory=false`,
`promotion_authorized=false`; MFM/SMG y certificación mantienen autoridad.

La tercera extensión, `pedagogical_teacher`, corre post-outcome en
`finalize_episode`. Enlaza lección, situación, sesgo aplicado, certificado, reward y
severidad respecto del golpe origen. Una sola mejora queda como observación, nunca
como efecto causal probado; un resultado igual o peor propone cuarentena curricular.
El 7B ya no escribe su alternativa preferida como éxito certificado: sólo refuerza
la herida observada y conserva la preferencia como propuesta. Sus cicatrices
sintéticas quedan marcadas y se excluyen de nuevas reflexiones para impedir un bucle
de auto-enseñanza.

Las extensiones siguientes completan la primera ronda de diez tareas:
`model_data_immune`, `curriculum_learning`, `sensorimotor_world_model`,
`interoceptive_homeostatic`, `metabolic_budget`, `development_lineage`, `horizontal_creativity` y
`social_exocortex`. Todas producen el mismo ciclo explícito
`measure → classify → analyze → deliberate`, tienen autoridad `none` y preservan
los gates soberanos existentes.

`interoceptive_homeostatic` no duplica `metabolic_budget`: el primero integra el
estado interno del organismo y rechaza defaults como mediciones; el segundo resume
el sobre de cómputo. Ambos conservan autoridad en MSRC, N0 y el kernel de viabilidad.

`curriculum_learning` formaliza la evaluación solicitada del 7B. Una comparación
válida requiere la misma situación y semilla con variantes `no_teacher`,
`local_7b` y `codex_frontier`; registra fuente, reducción de severidad y latencia.
Codex puede aportar lecciones estructuradas como docente externo, pero ni esas
lecciones ni las del 7B se convierten en entrenamiento hasta demostrar mejora
repetida held-out. El 7B se considera alumno/candidato, no maestro eficaz por
supuesto.

## Orden e invariantes

1. Conectómica contrasta topología declarada, actividad y consumo real.
2. Adversarial valida identidad causal, hashes, replay y techos de autoridad.
3. Comunicación latente propone ganancia acotada sólo con evidencia informativa.
4. Simbiosis calcula cobertura exacta e identifica órganos aislados.
5. Orquestación resume el ciclo y propaga bloqueo/degradación sin decidir acciones.

- Exactamente cinco roles únicos por ciclo.
- Hash canónico determinista por reporte y por ciclo.
- Autoridad máxima `evidence_only`; orquestación, conectómica, latente y adversarial
  usan `none`.
- Toda modulación usa límites `[0.75, 1.25]` como **envolvente de seguridad**, no
  como setpoint cognitivo. El setpoint permanece `unlearned`. Cada propuesta declara
  `apply_authorized=false` y no toca el grafo ni pesos vivos.
- La comunicación latente v1 no usa Mamba2. Mamba2 queda como backend opcional
  `SHADOW` para memoria temporal o experimentos de creatividad horizontal, nunca
  como canal canónico ni como autoridad.
- La misión cognitiva combina la ecología formal de familias de razonamiento,
  entrenamiento neuronal y guía de un 7B local. Se distinguen dos usos del mismo
  recurso: razonador caro `tier_3_external` y maestro post-experiencia. Ambos sólo
  producen evidencia tipada; el organismo conserva medición, deliberación y
  autoridad. Esta capa v1 no ejecuta entrenamiento ni invoca al 7B.
- Ausencia de confianza, incertidumbre o recibo informativo produce abstención.
- Una inconsistencia de identidad, hash o autoridad pone la fuente en cuarentena y
  bloquea la afirmación de sinergia.

## Coherencia temporal N4

N4 puede producir una propuesta preliminar antes del razonamiento para conservar
compatibilidad diagnóstica, pero esa propuesta no recibe recibo. Después de resolver
el posible override, `bind_committed_action` reemplaza N4 en la traza soberana,
recalcula con la `causal_attestation` final y emite el único recibo causal con
`temporal_binding=committed_action`.

## Plasticidad conectómica

La plasticidad usa clases explícitas: sólo `accepted` suma evidencia positiva;
`rejected`, `invalid` y `persistence_degraded` suman negativa. Las clases
observacionales o de abstención son neutrales. Esto evita convertir ausencia de
refutación en éxito y mantiene N6 sin propuesta cuando no hay evidencia informativa.

## Promoción y rollback

La capa permanece experimental hasta demostrar estabilidad multiescenario y costo
acotado. El rollback consiste en retirar `runtime/neural/agents/` y la clave aditiva
`neural_agents`; ninguna ruta autoritativa depende de estos reportes.
