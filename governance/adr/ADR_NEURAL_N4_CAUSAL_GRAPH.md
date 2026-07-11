---
title: ADR_NEURAL_N4_CAUSAL_GRAPH
status: experimental
version: 2.0.0
date: 2026-07-11
owner: Codex
---

# ADR — N4 Grafo neural causal tipado

## Contexto y principio rector

N4 es un órgano de propuesta bajo A-M0. Debe enriquecer la evidencia causal sin
degradar la coherencia global de A-M4 ni crear una falsa herencia bajo A-M8. CAU,
CTF, C-GWM, LOT-F, DED, la constitución y la certificación conservan autoridad.
Este ADR no habilita un hook vivo ni modifica firmas causales canónicas.

## Decisión

El contrato `n4-causal-graph-v1` representa snapshots causales de solo lectura.
Todos los nodos incluyen id estable, tipo, vector de características, procedencia,
identidad de escenario, timestamp opcional y versión. Tipos permitidos:

- `world_variable`, `observation`, `intervention`, `sign`;
- `evidence`, `memory`, `goal`, `constraint`.

Todas las aristas incluyen id estable, origen, destino, tipo, fuerza firmada,
confianza, procedencia, marca canónica y versión. Tipos permitidos:

- `causal_positive`, `causal_negative`, `temporal`, `support`;
- `contradiction`, `counterfactual`, `semantic`, `morphism`.

Las relaciones entre escenarios solo son válidas mediante una arista `morphism`.
Tipos desconocidos, ids duplicados, aristas colgantes, NaN/Inf, confianza inválida,
causalidad sin signo, procedencia ausente, identidad de escenario inconsistente o
versión incompatible fallan cerrado.

## Salida y representación del efecto

El contrato de salida es `n4-causal-proposal-v1`. Para cada relación predicha
incluye efecto esperado firmado en `[-1, 1]`, magnitud absoluta, incertidumbre,
confianza, estimación acotada del siguiente estado, ids de soporte, identidad/hash
del modelo, versión del grafo e indicación OOD/evidencia insuficiente.

El efecto usa dirección explícita y magnitud acotada por `tanh`; no usa un escalar
sigmoid como efecto causal. La incertidumbre procede de una cabeza independiente y
no se identifica con la magnitud. El artefacto v1 exige cuatro salidas mínimas:
estado, modulación de efecto, confianza e incertidumbre.

## Desacuerdo con la firma canónica

N4 informa uno de estos estados estructurados:

- `aligned`;
- `weak_disagreement`;
- `direction_conflict`;
- `missing_canonical_edge`;
- `unsupported_prediction`;
- `insufficient_evidence`.

Un conflicto de dirección permanece explícito y reduce la confianza de admisión;
nunca se promedia con la firma canónica. OOD o evidencia insuficiente requieren el
fallback de las autoridades causales deterministas.

## Backend de referencia y procedencia

`CausalMessagePassingBackend` es un backend CPU compacto para validar contratos.
Cada salida declara `classification`, `trained`, `frozen` y `experimental`, además
del hash de artefacto y manifiesto. El artefacto incluido en tests es `reference`,
no entrenado, congelado y experimental; no se presenta como GNN entrenada.

El contrato v0 permanece únicamente como adaptador de fixtures LAB, marcado
`legacy_reference_v0`; se rechaza en frontera LIVE y no constituye integración.

## Autoridad y admisión

La salida estricta no contiene mutaciones, acciones ni intervención seleccionada.
La admisión puede rechazar una forma inválida, registrar shadow, marcar desacuerdo
y reducir confianza. No puede:

- alterar C-GWM o una firma causal canónica;
- reemplazar CAU o CTF;
- elegir una intervención;
- autorizar una acción, cierre o certificado;
- modificar memoria, estado del organismo o scheduler.

Aunque el runtime genérico disponga de modo `provisional`, N4 permanece limitado a
laboratorio/shadow por este ADR hasta una ratificación posterior. Sin causalidad
enlazada, el runtime degrada provisional a shadow automáticamente.

## Presupuesto físico

El perfil de referencia fija límites duros de 512 nodos, 4096 aristas, 16 384
valores de características y cuatro pasos de message passing. Cada artefacto puede
reducirlos, nunca ampliarlos. El backend es CPU y reporta RAM estimada y VRAM cero.
El benchmark exige asignación Python pico menor o igual a 64 MiB para sus grafos
sintéticos; N0 conserva además los gates físicos globales.

## Evidencia y promoción

`runtime/neural/lab/n4_benchmark.py` cubre, con al menos tres semillas: efecto
positivo, efecto negativo, cadena, collider, confusor ambiguo, contradicción, firma
canónica ausente, morfismo entre escenarios, desacuerdo factual/contrafactual y OOD.
Reporta precisión del signo, error de magnitud/siguiente estado, precisión/recall
de desacuerdo, calibración, rechazo de grafos malformados, latencia, RAM, VRAM y
fallback.

La promoción futura requiere además los gates generales N0/A-M0: cero violaciones
de autoridad, intervalo bootstrap positivo, no regresión global mayor a un punto,
ECE menor o igual a 0.10 cuando exista probabilidad, presupuesto físico satisfecho
y `OrganismImpactReport` sin daño significativo. Este backend de referencia no
cumple por sí solo evidencia de entrenamiento ni de promoción.

## Rollback

El rollback es poner N4 en `off` o retirar su manifiesto/registro. No existen
migraciones, mutaciones de grafo ni estado neural persistente que revertir. En
shadow, el fallback determinista sigue siendo la salida efectiva y las propuestas
solo dejan trazas neuronales versionadas.
