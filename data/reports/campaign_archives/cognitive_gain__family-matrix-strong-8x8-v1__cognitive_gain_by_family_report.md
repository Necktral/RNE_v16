# Ganancia cognitiva por tipo de razonamiento — family-matrix-strong-8x8-v1

Generado: 2026-06-13T17:40:52  
Diseño: 10 perfiles × 5 regímenes × 8 bloques × 8 episodios. Protocolo primario `steps10` (presupuesto 10 = tope duro: cada familia puede expresarse; la recompensa canon internaliza el coste extra) + protocolo de sensibilidad `natural` (presupuesto 6 por defecto). Bootstrap 1000.

## 1. Dictamen primario

`no hay ganancia cognitiva suficiente en ningún tipo de razonamiento`

## 2. Matriz régimen × perfil (protocolo primario)

### homogeneous_safe

| Perfil | IVC-R | ΔIVC-R | IoC | IoC* | Ω | Recompensa r | ΔRecompensa | Coste | Éxito | Cierre | Clase |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| núcleo solo | 0.0493 | — | 0.8880 | 0.8880 | 0.0000 | -0.0590 | — | 5.9 | 1.00 | sí | baseline |
| núcleo + HEUR | 0.0449 | -0.0044 | 0.8880 | 0.8880 | 0.0000 | -0.0650 | -0.0060 | 6.5 | 1.00 | sí | sin ganancia |
| núcleo + DIA_ADV | 0.0469 | -0.0024 | 0.8880 | 0.8880 | 0.0000 | -0.0700 | -0.0110 | 7.0 | 1.00 | sí | sin ganancia |
| núcleo + FAL_GUARD | 0.0434 | -0.0059 | 0.8880 | 0.8880 | 0.0000 | -0.0680 | -0.0090 | 6.8 | 1.00 | sí | sin ganancia |
| núcleo + IND | 0.0441 | -0.0051 | 0.8880 | 0.8880 | 0.0000 | -0.0670 | -0.0080 | 6.7 | 1.00 | sí | sin ganancia |
| núcleo + PLAN | 0.0425 | -0.0068 | 0.8880 | 0.8880 | 0.0000 | -0.0700 | -0.0110 | 7.0 | 1.00 | sí | sin ganancia |
| núcleo + OPT | 0.0445 | -0.0048 | 0.8880 | 0.8880 | 0.0000 | -0.0690 | -0.0100 | 6.9 | 1.00 | sí | sin ganancia |
| núcleo + PLAN + OPT | 0.0442 | -0.0051 | 0.8880 | 0.8880 | 0.0000 | -0.0800 | -0.0210 | 8.0 | 1.00 | sí | sin ganancia |
| adaptativo v2 | 0.0433 | -0.0059 | 0.8880 | 0.8880 | 0.0000 | -0.0590 | +0.0000 | 5.9 | 1.00 | sí | sin ganancia |
| exploración total | 0.0433 | -0.0059 | 0.8880 | 0.8880 | 0.0000 | -0.0950 | -0.0360 | 9.5 | 1.00 | no | sin ganancia |

### heterogeneous_elevated

| Perfil | IVC-R | ΔIVC-R | IoC | IoC* | Ω | Recompensa r | ΔRecompensa | Coste | Éxito | Cierre | Clase |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| núcleo solo | 0.4840 | — | 0.8880 | 0.8880 | 0.0000 | -0.0590 | — | 5.9 | 1.00 | sí | baseline |
| núcleo + HEUR | 0.4454 | -0.0386 | 0.8880 | 0.8880 | 0.0000 | -0.0650 | -0.0060 | 6.5 | 1.00 | sí | sin ganancia |
| núcleo + DIA_ADV | 0.4632 | -0.0209 | 0.8880 | 0.8880 | 0.0000 | -0.0700 | -0.0110 | 7.0 | 1.00 | sí | sin ganancia |
| núcleo + FAL_GUARD | 0.4394 | -0.0446 | 0.8880 | 0.8880 | 0.0000 | -0.0680 | -0.0090 | 6.8 | 1.00 | sí | sin ganancia |
| núcleo + IND | 0.4431 | -0.0409 | 0.8880 | 0.8880 | 0.0000 | -0.0670 | -0.0080 | 6.7 | 1.00 | sí | sin ganancia |
| núcleo + PLAN | 0.4388 | -0.0453 | 0.8880 | 0.8880 | 0.0000 | -0.0700 | -0.0110 | 7.0 | 1.00 | sí | sin ganancia |
| núcleo + OPT | 0.4425 | -0.0416 | 0.8880 | 0.8880 | 0.0000 | -0.0690 | -0.0100 | 6.9 | 1.00 | sí | sin ganancia |
| núcleo + PLAN + OPT | 0.4392 | -0.0448 | 0.8880 | 0.8880 | 0.0000 | -0.0800 | -0.0210 | 8.0 | 1.00 | sí | sin ganancia |
| adaptativo v2 | 0.4347 | -0.0494 | 0.8880 | 0.8880 | 0.0000 | -0.0650 | -0.0060 | 6.5 | 1.00 | sí | sin ganancia |
| exploración total | 0.4324 | -0.0517 | 0.8880 | 0.8880 | 0.0000 | -0.0950 | -0.0360 | 9.5 | 1.00 | no | sin ganancia |

### heterogeneous_warning

| Perfil | IVC-R | ΔIVC-R | IoC | IoC* | Ω | Recompensa r | ΔRecompensa | Coste | Éxito | Cierre | Clase |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| núcleo solo | 0.4507 | — | 0.8880 | 0.8880 | 0.0000 | -0.0590 | — | 5.9 | 1.00 | sí | baseline |
| núcleo + HEUR | 0.4217 | -0.0290 | 0.8880 | 0.8880 | 0.0000 | -0.0650 | -0.0060 | 6.5 | 1.00 | sí | sin ganancia |
| núcleo + DIA_ADV | 0.4166 | -0.0341 | 0.8880 | 0.8880 | 0.0000 | -0.0700 | -0.0110 | 7.0 | 1.00 | sí | sin ganancia |
| núcleo + FAL_GUARD | 0.4120 | -0.0387 | 0.8880 | 0.8880 | 0.0000 | -0.0680 | -0.0090 | 6.8 | 1.00 | sí | sin ganancia |
| núcleo + IND | 0.4161 | -0.0346 | 0.8880 | 0.8880 | 0.0000 | -0.0670 | -0.0080 | 6.7 | 1.00 | sí | sin ganancia |
| núcleo + PLAN | 0.4136 | -0.0370 | 0.8880 | 0.8880 | 0.0000 | -0.0700 | -0.0110 | 7.0 | 1.00 | sí | sin ganancia |
| núcleo + OPT | 0.4171 | -0.0336 | 0.8880 | 0.8880 | 0.0000 | -0.0690 | -0.0100 | 6.9 | 1.00 | sí | sin ganancia |
| núcleo + PLAN + OPT | 0.4121 | -0.0386 | 0.8880 | 0.8880 | 0.0000 | -0.0800 | -0.0210 | 8.0 | 1.00 | sí | sin ganancia |
| adaptativo v2 | 0.4071 | -0.0435 | 0.8880 | 0.8880 | 0.0000 | -0.0740 | -0.0150 | 7.4 | 1.00 | sí | sin ganancia |
| exploración total | 0.4071 | -0.0436 | 0.8880 | 0.8880 | 0.0000 | -0.0950 | -0.0360 | 9.5 | 1.00 | no | sin ganancia |

### viability_edge

| Perfil | IVC-R | ΔIVC-R | IoC | IoC* | Ω | Recompensa r | ΔRecompensa | Coste | Éxito | Cierre | Clase |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| núcleo solo | 0.3957 | — | 0.8880 | 0.8880 | 0.0000 | -0.0590 | — | 5.9 | 1.00 | sí | baseline |
| núcleo + HEUR | 0.3790 | -0.0166 | 0.8880 | 0.8880 | 0.0000 | -0.0650 | -0.0060 | 6.5 | 1.00 | sí | sin ganancia |
| núcleo + DIA_ADV | 0.3534 | -0.0423 | 0.8880 | 0.8880 | 0.0000 | -0.0700 | -0.0110 | 7.0 | 1.00 | sí | sin ganancia |
| núcleo + FAL_GUARD | 0.3641 | -0.0316 | 0.8880 | 0.8880 | 0.0000 | -0.0680 | -0.0090 | 6.8 | 1.00 | sí | sin ganancia |
| núcleo + IND | 0.3631 | -0.0326 | 0.8880 | 0.8880 | 0.0000 | -0.0670 | -0.0080 | 6.7 | 1.00 | sí | sin ganancia |
| núcleo + PLAN | 0.3696 | -0.0260 | 0.8880 | 0.8880 | 0.0000 | -0.0700 | -0.0110 | 7.0 | 1.00 | sí | sin ganancia |
| núcleo + OPT | 0.3668 | -0.0289 | 0.8880 | 0.8880 | 0.0000 | -0.0690 | -0.0100 | 6.9 | 1.00 | sí | sin ganancia |
| núcleo + PLAN + OPT | 0.3597 | -0.0360 | 0.8880 | 0.8880 | 0.0000 | -0.0800 | -0.0210 | 8.0 | 1.00 | sí | sin ganancia |
| adaptativo v2 | 0.3606 | -0.0351 | 0.8880 | 0.8880 | 0.0000 | -0.0790 | -0.0200 | 7.9 | 1.00 | sí | sin ganancia |
| exploración total | 0.3596 | -0.0361 | 0.8880 | 0.8880 | 0.0000 | -0.0950 | -0.0360 | 9.5 | 1.00 | no | sin ganancia |

### vram_favorable

| Perfil | IVC-R | ΔIVC-R | IoC | IoC* | Ω | Recompensa r | ΔRecompensa | Coste | Éxito | Cierre | Clase |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| núcleo solo | 0.0439 | — | 0.8880 | 0.8880 | 0.0000 | -0.0590 | — | 5.9 | 1.00 | sí | baseline |
| núcleo + HEUR | 0.0459 | +0.0021 | 0.8880 | 0.8880 | 0.0000 | -0.0650 | -0.0060 | 6.5 | 1.00 | sí | ganancia cognitiva condicionada |
| núcleo + DIA_ADV | 0.0419 | -0.0019 | 0.8880 | 0.8880 | 0.0000 | -0.0700 | -0.0110 | 7.0 | 1.00 | sí | sin ganancia |
| núcleo + FAL_GUARD | 0.0439 | +0.0001 | 0.8880 | 0.8880 | 0.0000 | -0.0680 | -0.0090 | 6.8 | 1.00 | sí | ganancia cognitiva condicionada |
| núcleo + IND | 0.0412 | -0.0026 | 0.8880 | 0.8880 | 0.0000 | -0.0670 | -0.0080 | 6.7 | 1.00 | sí | sin ganancia |
| núcleo + PLAN | 0.0437 | -0.0001 | 0.8880 | 0.8880 | 0.0000 | -0.0700 | -0.0110 | 7.0 | 1.00 | sí | sin ganancia |
| núcleo + OPT | 0.0434 | -0.0005 | 0.8880 | 0.8880 | 0.0000 | -0.0690 | -0.0100 | 6.9 | 1.00 | sí | sin ganancia |
| núcleo + PLAN + OPT | 0.0426 | -0.0013 | 0.8880 | 0.8880 | 0.0000 | -0.0800 | -0.0210 | 8.0 | 1.00 | sí | sin ganancia |
| adaptativo v2 | 0.0427 | -0.0011 | 0.8880 | 0.8880 | 0.0000 | -0.0650 | -0.0060 | 6.5 | 1.00 | sí | sin ganancia |
| exploración total | 0.0424 | -0.0015 | 0.8880 | 0.8880 | 0.0000 | -0.0950 | -0.0360 | 9.5 | 1.00 | no | sin ganancia |

## 3. Aislamiento por familia (core+X vs core_only)

| Familia | Rol global | Regímenes positivos | Regímenes dañinos | Δr̄ (recompensa) | Evidencia por régimen |
|---|---|---|---|---:|---|
| HEUR | neutral | — | heterogeneous_elevated, heterogeneous_warning, viability_edge | -0.0060 | homogeneous_safe: neutral (ΔIVC-R -0.0044, Δr -0.0060, CI r [-0.0060,-0.0060]); heterogeneous_elevated: perjudica (ΔIVC-R -0.0386, Δr -0.0060, CI r [-0.0060,-0.0060]); heterogeneous_warning: perjudica (ΔIVC-R -0.0290, Δr -0.0060, CI r [-0.0060,-0.0060]); viability_edge: perjudica (ΔIVC-R -0.0166, Δr -0.0060, CI r [-0.0060,-0.0060]); vram_favorable: neutral (ΔIVC-R +0.0021, Δr -0.0060, CI r [-0.0060,-0.0060]) |
| DIA_ADV | neutral | — | heterogeneous_elevated, heterogeneous_warning, viability_edge | -0.0110 | homogeneous_safe: neutral (ΔIVC-R -0.0024, Δr -0.0110, CI r [-0.0110,-0.0110]); heterogeneous_elevated: perjudica (ΔIVC-R -0.0209, Δr -0.0110, CI r [-0.0110,-0.0110]); heterogeneous_warning: perjudica (ΔIVC-R -0.0341, Δr -0.0110, CI r [-0.0110,-0.0110]); viability_edge: perjudica (ΔIVC-R -0.0423, Δr -0.0110, CI r [-0.0110,-0.0110]); vram_favorable: neutral (ΔIVC-R -0.0019, Δr -0.0110, CI r [-0.0110,-0.0110]) |
| FAL_GUARD | perjudica | — | homogeneous_safe, heterogeneous_elevated, heterogeneous_warning, viability_edge | -0.0090 | homogeneous_safe: perjudica (ΔIVC-R -0.0059, Δr -0.0090, CI r [-0.0090,-0.0090]); heterogeneous_elevated: perjudica (ΔIVC-R -0.0446, Δr -0.0090, CI r [-0.0090,-0.0090]); heterogeneous_warning: perjudica (ΔIVC-R -0.0387, Δr -0.0090, CI r [-0.0090,-0.0090]); viability_edge: perjudica (ΔIVC-R -0.0316, Δr -0.0090, CI r [-0.0090,-0.0090]); vram_favorable: neutral (ΔIVC-R +0.0001, Δr -0.0090, CI r [-0.0090,-0.0090]) |
| IND | perjudica | — | homogeneous_safe, heterogeneous_elevated, heterogeneous_warning, viability_edge | -0.0080 | homogeneous_safe: perjudica (ΔIVC-R -0.0051, Δr -0.0080, CI r [-0.0080,-0.0080]); heterogeneous_elevated: perjudica (ΔIVC-R -0.0409, Δr -0.0080, CI r [-0.0080,-0.0080]); heterogeneous_warning: perjudica (ΔIVC-R -0.0346, Δr -0.0080, CI r [-0.0080,-0.0080]); viability_edge: perjudica (ΔIVC-R -0.0326, Δr -0.0080, CI r [-0.0080,-0.0080]); vram_favorable: neutral (ΔIVC-R -0.0026, Δr -0.0080, CI r [-0.0080,-0.0080]) |
| PLAN | perjudica | — | homogeneous_safe, heterogeneous_elevated, heterogeneous_warning, viability_edge | -0.0110 | homogeneous_safe: perjudica (ΔIVC-R -0.0068, Δr -0.0110, CI r [-0.0110,-0.0110]); heterogeneous_elevated: perjudica (ΔIVC-R -0.0453, Δr -0.0110, CI r [-0.0110,-0.0110]); heterogeneous_warning: perjudica (ΔIVC-R -0.0370, Δr -0.0110, CI r [-0.0110,-0.0110]); viability_edge: perjudica (ΔIVC-R -0.0260, Δr -0.0110, CI r [-0.0110,-0.0110]); vram_favorable: neutral (ΔIVC-R -0.0001, Δr -0.0110, CI r [-0.0110,-0.0110]) |
| OPT | neutral | — | heterogeneous_elevated, heterogeneous_warning, viability_edge | -0.0100 | homogeneous_safe: neutral (ΔIVC-R -0.0048, Δr -0.0100, CI r [-0.0100,-0.0100]); heterogeneous_elevated: perjudica (ΔIVC-R -0.0416, Δr -0.0100, CI r [-0.0100,-0.0100]); heterogeneous_warning: perjudica (ΔIVC-R -0.0336, Δr -0.0100, CI r [-0.0100,-0.0100]); viability_edge: perjudica (ΔIVC-R -0.0289, Δr -0.0100, CI r [-0.0100,-0.0100]); vram_favorable: neutral (ΔIVC-R -0.0005, Δr -0.0100, CI r [-0.0100,-0.0100]) |

## 4. Sinergia deliberativa (PLAN+OPT)

- **homogeneous_safe**: ΔIVC-R PLAN -0.0068, OPT -0.0048, combo -0.0051 ⇒ sinergia -0.0003 (redundante); Δr combo -0.0210.
- **heterogeneous_elevated**: ΔIVC-R PLAN -0.0453, OPT -0.0416, combo -0.0448 ⇒ sinergia -0.0032 (redundante); Δr combo -0.0210.
- **heterogeneous_warning**: ΔIVC-R PLAN -0.0370, OPT -0.0336, combo -0.0386 ⇒ sinergia -0.0049 (redundante); Δr combo -0.0210.
- **viability_edge**: ΔIVC-R PLAN -0.0260, OPT -0.0289, combo -0.0360 ⇒ sinergia -0.0099 (redundante); Δr combo -0.0210.
- **vram_favorable**: ΔIVC-R PLAN -0.0001, OPT -0.0005, combo -0.0013 ⇒ sinergia -0.0012 (redundante); Δr combo -0.0210.

## 5. Núcleo ABD/ANA/CAU/CTF/DED/PROB — contrafactual intra-episodio

El núcleo no es ablacionable (los floors de cierre lo protegen); su aporte se mide con el contrafactual intra-episodio `family_delta_ivc_r` sobre la celda `core_only`.

| Régimen | ABD | ANA | CAU | CTF | DED | PROB |
|---|---:|---:|---:|---:|---:|---:|
| homogeneous_safe | +0.01047 | +0.00525 | +0.00454 | +0.01022 | +0.01079 | +0.00798 |
| heterogeneous_elevated | +0.09222 | +0.04625 | +0.08052 | +0.09002 | +0.09502 | +0.08002 |
| heterogeneous_warning | +0.09091 | +0.04281 | +0.07302 | +0.08332 | +0.08795 | +0.07267 |
| viability_edge | +0.08073 | +0.03801 | +0.06248 | +0.07399 | +0.07810 | +0.06235 |
| vram_favorable | +0.00932 | +0.00467 | +0.00404 | +0.00910 | +0.00961 | +0.00711 |

## 6. Economía del razonamiento (recompensa canon r = ΔIoC* − λE·coste)

La recompensa semi-Markov internaliza el coste: Δr vs core es ganancia neta en la escala del canon. Ranking global (media de Δr sobre regímenes):

1. **HEUR** — Δr̄ -0.0060 (rol: neutral)
2. **IND** — Δr̄ -0.0080 (rol: perjudica)
3. **FAL_GUARD** — Δr̄ -0.0090 (rol: perjudica)
4. **OPT** — Δr̄ -0.0100 (rol: neutral)
5. **DIA_ADV** — Δr̄ -0.0110 (rol: neutral)
6. **PLAN** — Δr̄ -0.0110 (rol: perjudica)

## 7. Coherencia multi-contexto (Ω / IoC*)

- **HEUR**: ΔΩ̄ +0.0000, ΔIoC*̄ +0.0000 (no reduce obstrucción).
- **DIA_ADV**: ΔΩ̄ +0.0000, ΔIoC*̄ +0.0000 (no reduce obstrucción).
- **FAL_GUARD**: ΔΩ̄ +0.0000, ΔIoC*̄ +0.0000 (no reduce obstrucción).
- **IND**: ΔΩ̄ +0.0000, ΔIoC*̄ +0.0000 (no reduce obstrucción).
- **PLAN**: ΔΩ̄ +0.0000, ΔIoC*̄ +0.0000 (no reduce obstrucción).
- **OPT**: ΔΩ̄ +0.0000, ΔIoC*̄ +0.0000 (no reduce obstrucción).

## 8. Sensibilidad al presupuesto natural (6 pasos)

Bajo el presupuesto por defecto, la inserción legacy de overlays expulsa a DED (el validador Z3) y rompe el cierre — el artefacto que contaminó el dictamen de abril. Tasas de éxito por perfil (primario `steps10` vs `natural`):

| Régimen | Perfil | Éxito steps10 | Éxito natural | Cierre roto natural | r natural |
|---|---|---:|---:|---:|---:|
| homogeneous_safe | exploración total | 1.00 | 1.00 | 1.00 | -0.0983 |
| heterogeneous_elevated | núcleo + HEUR | 1.00 | 1.00 | 1.00 | -0.0983 |
| heterogeneous_elevated | exploración total | 1.00 | 1.00 | 1.00 | -0.0983 |
| heterogeneous_warning | núcleo + HEUR | 1.00 | 1.00 | 1.00 | -0.0983 |
| heterogeneous_warning | núcleo + PLAN | 1.00 | 1.00 | 1.00 | -0.0983 |
| heterogeneous_warning | núcleo + PLAN + OPT | 1.00 | 1.00 | 1.00 | -0.0983 |
| heterogeneous_warning | exploración total | 1.00 | 1.00 | 1.00 | -0.0983 |
| viability_edge | núcleo + DIA_ADV | 1.00 | 1.00 | 1.00 | -0.0983 |
| viability_edge | núcleo + FAL_GUARD | 1.00 | 1.00 | 1.00 | -0.0983 |
| viability_edge | núcleo + PLAN | 1.00 | 1.00 | 1.00 | -0.0983 |
| viability_edge | núcleo + OPT | 1.00 | 1.00 | 1.00 | -0.0983 |
| viability_edge | núcleo + PLAN + OPT | 1.00 | 1.00 | 1.00 | -0.0983 |
| viability_edge | exploración total | 1.00 | 1.00 | 1.00 | -0.0983 |
| vram_favorable | núcleo + HEUR | 1.00 | 1.00 | 1.00 | -0.0983 |
| vram_favorable | exploración total | 1.00 | 1.00 | 1.00 | -0.0983 |

## 9. Comparación con la campaña de abril 2026

- Abril (`adaptive_v2_intelligence_full_20260421_prompt1`): `no hay ganancia cognitiva suficiente` — medía solo HEUR/DIA_ADV/FAL_GUARD, sin IND/PLAN/OPT reales ni IoC*/Ω/recompensa.
- Dos artefactos corregidos desde abril: (a) el perfil de cierre `adaptive_min` no reconocía PLAN/OPT como opcionales legítimas (rechazo automático de toda secuencia deliberativa); (b) el protocolo primario de abril (presupuesto 6) expulsaba a DED al activarse cualquier overlay.
- Ahora: `no hay ganancia cognitiva suficiente en ningún tipo de razonamiento`.
- Clases por régimen en abril: homogeneous_safe: sin ganancia; heterogeneous_elevated: sin ganancia; heterogeneous_warning: sin ganancia; viability_edge: sin ganancia; vram_favorable: sin ganancia.

## 10. Guardas y riesgos residuales

- Ningún dictamen se sostiene sin `success_rate`, `closure_break_rate` y `backbone_floor_satisfied_rate` estables (columna Cierre de la matriz §2).
- La señal primaria sigue siendo bootstrap sobre episodios; la recompensa canon añade la dimensión de coste pero su Dₜ (disipación física) es 0 hasta R4.
- Los contrafactuales intra-episodio (`family_delta_ivc_r`) son proxies aditivos, no ablaciones reales del núcleo.
- PLAN/OPT operan sobre el modelo de efectos declarado de la firma causal; su valor depende de la fidelidad de esa firma al mundo.
- `full_family_exploration` no cabe ni en el tope duro de 10 pasos (6 núcleo + 6 opcionales): su celda mide al perfil bajo desbordamiento real de presupuesto, no el valor ideal de sus familias.
