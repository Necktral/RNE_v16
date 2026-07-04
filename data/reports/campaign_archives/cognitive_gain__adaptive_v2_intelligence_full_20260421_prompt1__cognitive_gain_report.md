# Ganancia Cognitiva v2

## 1. Dictamen primario

`no hay ganancia cognitiva suficiente`

## 2. Evidencia por régimen

| Régimen | Clase | Mejor perfil | Mejor baseline fijo | ΔIVC-R vs core | ΔPrecisión | ΔViability | ΔIoC proxy | success_rate | cierre estable | opcionales activas |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| homogeneous_safe | sin ganancia | core_only | core_only | -0.0040 | 0.0000 | 0.0000 | -0.0020 | 1.0000 | sí | - |
| heterogeneous_elevated | sin ganancia | core_only | core_only | -0.0349 | 0.0000 | 0.0000 | -0.0157 | 1.0000 | sí | HEUR |
| heterogeneous_warning | sin ganancia | core_only | core_only | -0.0323 | 0.0000 | 0.0000 | -0.0145 | 1.0000 | sí | HEUR, FAL_GUARD |
| viability_edge | sin ganancia | core_only | core_only | -0.0297 | 0.0000 | 0.0000 | -0.0134 | 1.0000 | sí | DIA_ADV, FAL_GUARD |
| vram_favorable | sin ganancia | core_only | core_only | -0.0031 | 0.0000 | 0.0000 | -0.0014 | 1.0000 | sí | HEUR |

## 3. Evidencia por perfil

La lectura principal se hace contra `core_only`; la comparación contra el mejor baseline fijo por régimen se deja en `regime_cognitive_verdicts.json` para aislar si el valor viene de adaptividad o de un overlay fijo.

## 4. Familias que parecen aportar

- `HEUR`: activo en 3 regímenes, aparece en 0 regímenes positivos y alinea con el mejor baseline fijo en 0.
- `DIA_ADV`: activo en 1 regímenes, aparece en 0 regímenes positivos y alinea con el mejor baseline fijo en 0.
- `FAL_GUARD`: activo en 2 regímenes, aparece en 0 regímenes positivos y alinea con el mejor baseline fijo en 0.

## 5. Coste y guardas

Los costos operativos se dejan como contexto. Ningún dictamen se sostiene sin `success_rate`, `closure_break_rate` y `backbone_floor_satisfied_rate` estables.

## 6. Riesgos residuales

- La señal primaria depende de bootstrap sobre episodios, no de tests pareados externos.
- `ioc_proxy_gain` es analítico y no sustituye a `ivc_r`.
- La sensibilidad con `reasoning_max_steps=10` se reporta, pero no gobierna el dictamen principal.