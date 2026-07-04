# Ganancia cognitiva por tipo de razonamiento — conflict-actuated-v1

Generado: 2026-06-13T18:17:52  
Diseño: 7 perfiles × 1 regímenes × 4 bloques × 6 episodios. Protocolo primario `steps10` (presupuesto 10 = tope duro: cada familia puede expresarse; la recompensa canon internaliza el coste extra) + protocolo de sensibilidad `natural` (presupuesto 6 por defecto). Bootstrap 500.

## 1. Dictamen primario

`hay ganancia cognitiva fuerte en al menos un tipo de razonamiento: IND, OPT, PLAN`

## 2. Matriz régimen × perfil (protocolo primario)

### causal_counterfactual_conflict

| Perfil | IVC-R | ΔIVC-R | IoC | IoC* | Ω | Recompensa r | ΔRecompensa | Coste | Éxito | Cierre | Clase |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| núcleo solo | 0.0484 | — | 0.8880 | 0.8880 | 0.0000 | -0.0590 | — | 5.9 | 1.00 | sí | baseline |
| núcleo + HEUR | 0.0485 | +0.0001 | 0.8880 | 0.8880 | 0.0000 | -0.0650 | -0.0060 | 6.5 | 1.00 | sí | ganancia cognitiva condicionada |
| núcleo + DIA_ADV | 0.0483 | -0.0001 | 0.8880 | 0.8880 | 0.0000 | -0.0700 | -0.0110 | 7.0 | 1.00 | sí | sin ganancia |
| núcleo + IND | 0.1155 | +0.0671 | 0.8859 | 0.8836 | 0.0079 | -0.0670 | -0.0080 | 6.7 | 1.00 | sí | ganancia cognitiva fuerte |
| núcleo + PLAN | 0.4526 | +0.4042 | 0.8880 | 0.8880 | 0.0000 | -0.0700 | -0.0110 | 7.0 | 1.00 | sí | ganancia cognitiva fuerte |
| núcleo + OPT | 0.4498 | +0.4014 | 0.8880 | 0.8880 | 0.0000 | -0.0690 | -0.0100 | 6.9 | 1.00 | sí | ganancia cognitiva fuerte |
| núcleo + PLAN + OPT | 0.4435 | +0.3950 | 0.8880 | 0.8880 | 0.0000 | -0.0800 | -0.0210 | 8.0 | 1.00 | sí | ganancia cognitiva fuerte |

## 3. Aislamiento por familia (core+X vs core_only)

| Familia | Rol global | Regímenes positivos | Regímenes dañinos | Δr̄ (recompensa) | Evidencia por régimen |
|---|---|---|---|---:|---|
| HEUR | neutral | — | — | -0.0060 | causal_counterfactual_conflict: neutral (ΔIVC-R +0.0001, Δr -0.0060, CI r [-0.0060,-0.0060]) |
| DIA_ADV | neutral | — | — | -0.0110 | causal_counterfactual_conflict: neutral (ΔIVC-R -0.0001, Δr -0.0110, CI r [-0.0110,-0.0110]) |
| FAL_GUARD | sin evidencia | — | — | +0.0000 | — |
| IND | aporta | causal_counterfactual_conflict | — | -0.0080 | causal_counterfactual_conflict: aporta (ΔIVC-R +0.0671, Δr -0.0080, CI r [-0.0125,-0.0040]) |
| PLAN | aporta | causal_counterfactual_conflict | — | -0.0110 | causal_counterfactual_conflict: aporta (ΔIVC-R +0.4042, Δr -0.0110, CI r [-0.0110,-0.0110]) |
| OPT | aporta | causal_counterfactual_conflict | — | -0.0100 | causal_counterfactual_conflict: aporta (ΔIVC-R +0.4014, Δr -0.0100, CI r [-0.0100,-0.0100]) |

## 4. Sinergia deliberativa (PLAN+OPT)

- **causal_counterfactual_conflict**: ΔIVC-R PLAN +0.4042, OPT +0.4014, combo +0.3950 ⇒ sinergia -0.0092 (redundante); Δr combo -0.0210.

## 5. Núcleo ABD/ANA/CAU/CTF/DED/PROB — contrafactual intra-episodio

El núcleo no es ablacionable (los floors de cierre lo protegen); su aporte se mide con el contrafactual intra-episodio `family_delta_ivc_r` sobre la celda `core_only`.

| Régimen | ABD | ANA | CAU | CTF | DED | PROB |
|---|---:|---:|---:|---:|---:|---:|
| causal_counterfactual_conflict | +0.01080 | +0.00513 | +0.00440 | +0.00990 | +0.01045 | +0.00773 |

## 6. Economía del razonamiento (recompensa canon r = ΔIoC* − λE·coste)

La recompensa semi-Markov internaliza el coste: Δr vs core es ganancia neta en la escala del canon. Ranking global (media de Δr sobre regímenes):

1. **FAL_GUARD** — Δr̄ +0.0000 (rol: sin evidencia)
2. **HEUR** — Δr̄ -0.0060 (rol: neutral)
3. **IND** — Δr̄ -0.0080 (rol: aporta)
4. **OPT** — Δr̄ -0.0100 (rol: aporta)
5. **DIA_ADV** — Δr̄ -0.0110 (rol: neutral)
6. **PLAN** — Δr̄ -0.0110 (rol: aporta)

## 7. Coherencia multi-contexto (Ω / IoC*)

- **HEUR**: ΔΩ̄ +0.0000, ΔIoC*̄ +0.0000 (no reduce obstrucción).
- **DIA_ADV**: ΔΩ̄ +0.0000, ΔIoC*̄ +0.0000 (no reduce obstrucción).
- **IND**: ΔΩ̄ +0.0079, ΔIoC*̄ -0.0044 (no reduce obstrucción).
- **PLAN**: ΔΩ̄ +0.0000, ΔIoC*̄ +0.0000 (no reduce obstrucción).
- **OPT**: ΔΩ̄ +0.0000, ΔIoC*̄ +0.0000 (no reduce obstrucción).

## 8. Sensibilidad al presupuesto natural (6 pasos)

Bajo el presupuesto por defecto, la inserción legacy de overlays expulsa a DED (el validador Z3) y rompe el cierre — el artefacto que contaminó el dictamen de abril. Tasas de éxito por perfil (primario `steps10` vs `natural`):

| Régimen | Perfil | Éxito steps10 | Éxito natural | Cierre roto natural | r natural |
|---|---|---:|---:|---:|---:|
| causal_counterfactual_conflict | núcleo + PLAN | 1.00 | 1.00 | 1.00 | -0.0983 |
| causal_counterfactual_conflict | núcleo + PLAN + OPT | 1.00 | 1.00 | 1.00 | -0.0983 |

## 9. Comparación con la campaña de abril 2026

- Abril (`adaptive_v2_intelligence_full_20260421_prompt1`): `no hay ganancia cognitiva suficiente` — medía solo HEUR/DIA_ADV/FAL_GUARD, sin IND/PLAN/OPT reales ni IoC*/Ω/recompensa.
- Dos artefactos corregidos desde abril: (a) el perfil de cierre `adaptive_min` no reconocía PLAN/OPT como opcionales legítimas (rechazo automático de toda secuencia deliberativa); (b) el protocolo primario de abril (presupuesto 6) expulsaba a DED al activarse cualquier overlay.
- Ahora: `hay ganancia cognitiva fuerte en al menos un tipo de razonamiento: IND, OPT, PLAN`.
- Clases por régimen en abril: homogeneous_safe: sin ganancia; heterogeneous_elevated: sin ganancia; heterogeneous_warning: sin ganancia; viability_edge: sin ganancia; vram_favorable: sin ganancia.

## 10. Guardas y riesgos residuales

- Ningún dictamen se sostiene sin `success_rate`, `closure_break_rate` y `backbone_floor_satisfied_rate` estables (columna Cierre de la matriz §2).
- La señal primaria sigue siendo bootstrap sobre episodios; la recompensa canon añade la dimensión de coste pero su Dₜ (disipación física) es 0 hasta R4.
- Los contrafactuales intra-episodio (`family_delta_ivc_r`) son proxies aditivos, no ablaciones reales del núcleo.
- PLAN/OPT operan sobre el modelo de efectos declarado de la firma causal; su valor depende de la fidelidad de esa firma al mundo.
- `full_family_exploration` no cabe ni en el tope duro de 10 pasos (6 núcleo + 6 opcionales): su celda mide al perfil bajo desbordamiento real de presupuesto, no el valor ideal de sus familias.
