# ADR: trayectoria dinámica canónica del organismo

## Estado

Implementado en `integration/dynamic-organism-v1` como contrato interno aditivo. No
modifica el canon, el registro de regímenes ni el esquema de storage.

## Decisión

Cada episodio sano materializa dos resúmenes acotados `organism-dynamic-state-v1`
para Ω_t y Ω_t+1. Conservan hashes/referencias del mundo, organismo, memoria, órganos
neuronales, política, recursos, homeostasis, linaje y régimen; no copian blobs ni
convierten mediciones ausentes en valores favorables.

Después de transición factual, certificación, recompensa, experiencia y finalización
de los consumidores neuronales, `DynamicLifeChain` construye
`organism-life-transition-v1`. Su ID no depende del reloj y su hash enlaza el estado
anterior, el posterior y el hash de la transición previa. Sólo tras persistir mediante
el ledger existente como `organism.life_transition.committed` avanza el head y se
cierra `neural-symbiosis-trace-v2`.

Una escritura fallida no avanza la cadena y devuelve `status=incomplete` con reason
code. No se presenta como vida sana.

## Régimen y continuidad

El runner consulta `runtime.organism.regime_model` sin duplicar ni modificar su
registro. Registra compatibilidad, distancia estructural y factibilidad de transporte.

El checkpoint soberano conserva, dentro de su bloque neural aditivo, N3 y el head
dinámico: organismo, linaje, último estado/hash/ID, índice, epoch y versiones. Un nuevo
`run_id` es válido; otro organismo o linaje falla cerrado. Un checkpoint N3 legacy
restaura N3 y abre una epoch explícita con
`legacy_checkpoint_without_dynamic_chain`, sin inventar historia.

## Autoridad

La cadena describe lo ocurrido. No elige intervención, no certifica, no escribe MFM,
no cambia scheduler/MSRC y no concede autoridad a N1–N6.
