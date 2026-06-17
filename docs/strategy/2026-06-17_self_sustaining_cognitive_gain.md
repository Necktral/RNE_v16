# Roadmap — Organismo autosustentable y próspero: cerrar el Bucle A

> **Estatus: estrategia NO-NORMATIVA.** No redefine el canon; se alinea con él y con su roadmap de
> ascenso (R1→R2). Documento de dirección, no contrato. Fecha: 2026-06-17.
> Meta del organismo: **ganancia cognitiva** sostenida. Primer movimiento elegido: **Bucle A**.
> Anclas canon: `canon/normative/CANON_RNFE_v3_2_rc1.md`,
> `docs/analysis/AUDITORIA_CANON_F21_F24_VS_CODIGO.md`, `docs/analysis/17_SYNTHESIS.md`.

## 1. Problema y meta

El organismo RNFE tiene los **órganos** de un sistema autosustentable, pero los **bucles metabólicos
están desconectados o en sombra**. Tiene cómo medir, cómo recordar, cómo evolucionar y cómo
seleccionar — pero esas piezas no se realimentan en un ciclo que haga que la **ganancia cognitiva**
crezca sola.

Diagnóstico (verificado en código):

| Órgano | Existe | Estado real |
|---|---|---|
| Autoevolución | ✅ | **viva pero superficial**: corre cada episodio ([scenario_runner.py:146,598](../../runtime/world/scenario_runner.py)), `RNFE_AUTOEVOLUTION=1`; pero solo mueve 2 perillas (memory_retrieval_limit, memory_filter_mode) con diagnóstico hardcodeado, sin aprender del linaje. |
| Reward → conducta | ✅ computa | **en sombra**: `RNFE_REWARD_GUIDED_SELECTION=0` ([reward_guided.py:64](../../runtime/reasoning/scheduler_meta/reward_guided.py#L64)), `RNFE_REASONING_ACTUATES=0` ([intervention_override.py:35](../../runtime/world/intervention_override.py#L35)), `λV` por defecto 0.0 ([reward.py:88](../../runtime/reasoning/scheduler_meta/reward.py#L88)). Se calcula y acumula, pero **no cambia lo que el organismo elige**. |
| Riesgo (CVaR/B_safe) | ✅ | **sombra**: `sie_rule` ACEPTAR/BUFFER/RECHAZAR existe ([risk_engine.py:194](../../runtime/certification/risk_engine.py#L194)) pero no gatea la aceptación de episodios. |
| Ecología / herencia | ✅ | real (fitness gateada por certificación) pero **no integrada al bucle de un organismo**: cada corrida arranca en frío. |
| Economía / metabolismo | ❌ | **falta**: mutar es gratis; el `RecoveryPlan` se genera pero no se ejecuta ([viability.py:42,165,203](../../runtime/organism/viability.py)); no hay reserva finita ni inanición. |

**La meta**, dicha con precisión: que la **ganancia cognitiva** (medida con las señales canon — IVC-R,
precisión de intervención, margen de viabilidad, efectividad; ver
[cognitive_gain_by_family_lib.py](../../scripts/cognitive_gain_by_family_lib.py)) **gobierne la
conducta, se acumule en el tiempo y la seleccione una presión real** — de modo que mejore sola, sin
hand-holding, sin romper el cierre ni la seguridad.

## 2. Tesis: la ganancia cognitiva es autosustentable solo si cierran 4 bucles

```
        (0) MEDIR BIEN ──► (A) IMPULSAR LA CONDUCTA ──► (B) COMPONER EN EL TIEMPO ──► (C) PRESIÓN DE SELECCIÓN
         medir por             el reward gobierna           la ganancia se acumula        un costo/viabilidad real
         efectividad (ν)       qué hace el organismo        entre corridas                hace "mejorar o morir"
            │                       │                            │                              │
            ✅ HECHO                ◀── primer movimiento        esbozado                       esbozado
                                    │
                          ┌─────────┴─────────┐
                          │  R1 — COLUMNA DE   │  CVaR/B_safe del shadow al enforcement:
                          │   SEGURIDAD        │  la llave que habilita encender A/B/C EN VIVO
                          └───────────────────┘
```

- **Bucle 0 — medir bien (✅ cerrado esta línea de trabajo).** El estudio de ceguera de recompensa y el
  experimento del funcional J probaron que el reward debe valorar la **efectividad / coherencia causal
  (ν = `cau.helps_goal`)**, no solo la continuidad, y que **descomponer IoC cura la ceguera**.
  Evidencia: `data/reports/reward_blindness/`, `data/reports/critical_functional/`, y la ADR de
  descomposición de IoC (PR #24, `docs/adr/ADR_REWARD_COHERENCE_BLINDNESS.md`, en rama
  `report/reward-blindness-1fb81b9e`).
- **Bucle A — impulsar la conducta.** Que el reward (ya curado) decida qué razonamiento se activa.
- **Bucle B — componer en el tiempo.** Que la ganancia persista y se acumule entre corridas.
- **Bucle C — presión de selección.** Que un costo de recursos / viabilidad real haga que la ganancia
  sea seleccionada, no opcional.
- **R1 — columna de seguridad.** El canon (S-I-E + viabilidad) exige el gate de riesgo *antes* de
  encender en vivo cualquier auto-modificación o selección autónoma.

**El hilo honesto:** *gain* autosustentable ⇔ medir(0) + conducta(A) + tiempo(B) + presión(C), gateado
por R1. Y una advertencia que **ya medimos**: encender los bucles *ingenuamente* (IoC colapsado +
λV crudo) **amplifica la ceguera** (umbral real λV≈20). Por eso A **debe** usar el reward descompuesto.

## 3. Bucle A en detalle — el primer movimiento

**Qué es:** que el reward efectividad-aware **gobierne la selección de familias de razonamiento**, en
lugar de que la conducta la fije un perfil declarativo estático.

### 3.1 Estado de sombra actual
- `RNFE_REWARD_GUIDED_SELECTION=0` → el `RewardGuidedOverlaySelector` observa y acumula Δr̄ por
  (familia, régimen) pero **no** sobreescribe el perfil fijo.
- `λV=0` → el término de efectividad no entra al reward; ν no registra.
- `RNFE_REASONING_ACTUATES=0` → el override guardado no actúa.

Resultado: el organismo *sabe* qué familias pagan, pero *actúa* como si no lo supiera.

### 3.2 Qué significa "impulsar la conducta" (reutilizando lo ya construido)
1. El selector gobierna las opcionales: `directives(run_id, regime)` decide on/off por Δr̄
   ([reward_guided.py](../../runtime/reasoning/scheduler_meta/reward_guided.py): directives/observe/merge_from).
2. El reward valora la efectividad: `λV>0` en `compute_episode_reward`
   ([reward.py:72,103](../../runtime/reasoning/scheduler_meta/reward.py)).
3. El override actúa con guard cuando hay conflicto causal
   ([intervention_override.py](../../runtime/world/intervention_override.py)).

No hay que inventar piezas: hay que **conectarlas y validarlas**.

### 3.3 El riesgo que YA medimos (y cómo A lo evita)
Encender `λV` sobre el escalar IoC colapsado exige **λV≈20** porque IoC anti-correlaciona con la acción
efectiva (continuidad domina, peso 0.45). Por eso A debe representar la viabilidad como el **canal
causal limpio** `cau.helps_goal` ∈ {0,1}
([core_inference.py:152](../../runtime/reasoning/families/core_inference.py#L152)), tal como predijo el
funcional J (recuperación a λ_ν≈1 en vez de ≈20). **A queda así enlazado a la descomposición de IoC**
(ADR) y al experimento J (`data/reports/critical_functional/`).

### 3.4 Métrica de éxito — FALSABLE
> ¿La selección guiada-por-reward **mejora** la ganancia cognitiva (Δ IVC-R y efectividad media, con CI
> entre-semillas) **vs el perfil fijo**, **sin** aumentar la tasa de ruptura de cierre?

- **Confirma A** si: Δ IVC-R > 0 con CI que excluye 0, efectividad media sube, cierre estable ≥ baseline.
- **Refuta A** si: no mejora la ganancia, o mejora a costa de romper cierre, o necesita λV≫O(1) (señal
  de que sigue colapsado). Se reporta sin spin, como el resto de la línea.

### 3.5 Sub-secuencia A1 → A2 → A3
- **A1 — validación en harness/sombra (sin runtime).** Conducir el selector real con el reward
  descompuesto y comparar ganancia cognitiva guiada vs perfil fijo, en los regímenes existentes.
  Cero riesgo; es el puente natural desde el funcional J. **Es el candidato a "siguiente estudio".**
- **A2 — activación gated en runtime.** Encender `RNFE_REWARD_GUIDED_SELECTION` / `λV` / override
  detrás de flags off-por-defecto, camino nominal byte-idéntico, con tests. **Requiere R1** (gate de
  seguridad) porque ya cambia conducta viva.
- **A3 — default.** Hacerlo el comportamiento por defecto. Lejano; gateado por evidencia acumulada de
  A2 en múltiples regímenes (estabilidad A7 del canon: reaparición cross-seed).

## 4. Bucles B y C (esbozados — los movimientos siguientes)

- **Bucle B — componer en el tiempo (prosperidad).** Hoy cada corrida/ecología arranca en frío. Para
  que la ganancia *acumule*: (i) **registro persistente de pericia** (régimen, familia)→Δr̄ que
  sobreviva entre corridas; (ii) **catálogo de morfismos** cacheado (hoy se recomputa cada vez); (iii)
  **transferencia de política cross-run** (merge_from ya existe a nivel ecología). Reutiliza
  [mfm_lite/](../../runtime/memory/mfm_lite/) (promoción micro→meso→macro),
  [lineage.py](../../runtime/organism/lineage.py) (herencia gateada),
  [ecology.py](../../runtime/organism/ecology.py). Resultado: una **curva de capacidad** que sube en el
  tiempo, no una sucesión de arranques en frío.
- **Bucle C — presión de selección (metabolismo).** Hoy mutar es gratis y el `RecoveryPlan` no se
  ejecuta. Para que la ganancia sea *seleccionada*: (i) **reserva de recursos finita** que el costo
  cognitivo agota y la recertificación repone; (ii) **enforcement de viabilidad** — ejecutar el
  `RecoveryPlan` de [viability.py:203](../../runtime/organism/viability.py#L203), no solo generarlo;
  (iii) **inanición** — un umbral de `recovery_debt` que fuerza modo de carga reducida. Resultado: el
  "mejorar o morir" que vuelve a la ganancia condición de supervivencia, no lujo.

Ambos dependen de **R1** para activarse en vivo con seguridad.

## 5. R1 — columna de seguridad (precondición de A2/B/C en vivo)

El canon exige que la auto-modificación/selección autónoma pase por un gate de riesgo. La pieza ya
existe en sombra: `sie_rule` (ACEPTAR / BUFFER / RECHAZAR) con CVaR y B_safe
([risk_engine.py:194](../../runtime/certification/risk_engine.py#L194)). R1 = **promover ese gate del
shadow al enforcement**: calibrarlo contra el histórico de certificados y enlazarlo a la aceptación de
episodios y a la auto-modificación. Sin R1, "autoevolutivo" = "auto-mutante sin guarda". R1 es la llave
de A2, B y C.

## 6. Roadmap secuenciado

| Fase | Qué hace | Desbloquea | Éxito (falsable) | Canon |
|---|---|---|---|---|
| **A1** | Estudio harness: selección guiada-por-reward (descompuesto) vs perfil fijo | Evidencia de que A mejora la ganancia | Δ IVC-R>0 (CI excluye 0), efectividad ↑, cierre ≥ baseline | R3 |
| **R1** | CVaR/B_safe del shadow al enforcement (S-I-E vivo) | Encender A2/B/C en vivo con seguridad | Gate calibrado; rechaza lo inseguro sin falsos positivos | R1 |
| **A2** | Activar reward-guided + λV + override gated en runtime | Conducta gobernada por ganancia, en vivo | Nominal byte-idéntico con flags off; con flags on, ganancia sube y cierre estable | R3 |
| **B** | Persistencia: registro de pericia + morfismos + transfer cross-run | Que la ganancia componga | Curva de capacidad creciente entre corridas | R2/R7 |
| **C** | Metabolismo: reserva finita + enforcement de viabilidad + inanición | Que la ganancia se seleccione | El organismo prioriza ganancia bajo escasez; sobrevive recertificando | R2 |

Orden recomendado: **A1 (medir) → R1 (asegurar) → A2 (encender) → B (componer) → C (presionar).**
Cada fase se valida antes de la siguiente; ninguna se enciende en vivo sin R1.

## 7. Riesgos y honestidad

- **Es estrategia, no promesa.** Cada bucle se valida con un estudio falsable antes de cablear runtime.
- **El modo de fallo medido es el ancla cautelar.** La ceguera de recompensa demuestra que encender
  bucles ingenuamente (coherencia colapsada) **degrada** el desempeño. A usa el reward descompuesto
  *precisamente* para no repetirlo.
- **Sin overclaim.** Esto no es "prosperidad" ni "AGI": es cerrar bucles de realimentación medibles
  para que una métrica concreta (ganancia cognitiva) deje de resetearse y empiece a componer bajo
  presión, con seguridad.
- **A2/B/C tocan conducta viva** ⇒ disciplina de sombra (flags off-por-defecto, nominal byte-idéntico,
  tests, R1 primero). Coherente con el resto del proyecto.

## 8. Decisiones abiertas (para después de revisar este roadmap)

1. **¿Ejecutar A1 como el siguiente estudio harness?** (selección guiada-por-reward descompuesto vs
   perfil fijo, midiendo ganancia cognitiva). Es el paso de menor riesgo y mayor información.
2. ¿Priorizar R1 en paralelo a A1 (para no bloquear A2 después)?
3. ¿Alcance de B en su primer corte (solo registro de pericia, o también catálogo de morfismos)?
4. ¿Modelo de C (reserva por episodio / por corrida / por generación)?

---
*Continúa la línea: Bucle 0 (medir bien) → este roadmap define el Bucle A como primer movimiento hacia
un organismo cuya ganancia cognitiva se gobierna, compone y se selecciona sola — con seguridad.*
