# ADR — Cierre del Bucle A (reward→conducta): activación gated y disciplina de sombra

- **Fecha:** 2026-07-03
- **Estado:** aceptado (A1 confirmado en harness; R1 cableado; A2 gated, OFF por defecto)
- **Contexto:** roadmap `docs/strategy/2026-06-17_self_sustaining_cognitive_gain.md`. El organismo
  ya *sabe* qué familias de razonamiento pagan (acumula Δr̄) pero *actúa* con el perfil fijo declarativo.
  Cerrar el Bucle A = que la recompensa gobierne la selección, sin romper el cierre ni la seguridad.

## Línea de evidencia (previa a esta ADR)

1. **reward_blindness** (modo de fallo, medido): el IoC colapsado anti-correlaciona con la acción
   efectiva ⇒ subir λV sobre efectividad exige λV≈20 (inflación 40×).
2. **critical_functional** (cura, harness): tratar ν=`cau.helps_goal` como criterio aditivo de
   primera clase recupera la familia efectiva con λ_ν≈1.0 (cura 20×), específico del canal de viabilidad.
3. **bucle_a_activation / A1** (este trabajo, harness): la selección guiada-por-recompensa con la
   recompensa descompuesta supera al perfil fijo en efectividad (Δ>0, CI excluye 0) a λ_ν=O(1), y
   compartir Δr̄ en vivo (ecología) multiplica por velocidad de convergencia.

## Decisión — tres flags, todos OFF por defecto (nominal byte-idéntico)

| Flag | Default | Efecto ON | PR |
|---|---|---|---|
| `RNFE_REWARD_LAMBDA_NU` | `0.0` | Término aditivo `λ_ν·ν` (ν=cau.helps_goal) en la recompensa | PR1 |
| `RNFE_REWARD_GUIDED_SELECTION` | `0` (previo) | El selector guiado gobierna las familias opcionales | previo |
| `RNFE_RISK_ENFORCEMENT` | `0` | El freno de riesgo de cola S-I-E (CVaR/prob) BLOQUEA la auto-modificación ρₜ | PR3 (R1) |
| `RNFE_REASONING_ACTUATES` | `0` (previo) | Actuación determinista por override certificado | previo |

Con TODOS los flags OFF, la conducta es **byte-idéntica** a antes (verificado: el escalar `reward`
no cambia con λ_ν=0; el gate ρₜ solo bloquea por violación hard/barrera, igual que antes). La
activación A2 = encender los flags, y está **condicionada** a:

- **Gate A1** (§3.4): Δ efectividad guiado−fijo > 0 con CI excluyendo 0; activación efectiva ↑;
  umbral λ_ν=O(1). — **confirmado en harness** (`data/reports/bucle_a_activation/REPORT.md`).
- **Gate R1**: el enforcement rechaza lo inseguro (CVaR>τ con evidencia) sin falsos positivos
  (no bloquea por historial insuficiente). — **cableado y testeado** (`tests/organism/test_autoevolution.py::TestR1RiskEnforcement`);
  umbrales calibrados 2026-06-10 (τ=0.10, prob=0.45; `scripts/calibrate_risk_cvar.py`).

## Rollback

Un solo switch por dimensión: poner el flag correspondiente a `0` restaura la conducta nominal.
No hay migración de datos ni estado persistente que revertir (los términos son aditivos y los gates
son evaluaciones puras).

## Fuera de alcance / trabajo futuro

- **A3** (hacer A por defecto): requiere evidencia cross-seed acumulada de A2 en varios regímenes (canon A7).
- Confirmación **in vivo** de la cura (A1 es harness): re-medir en runtime con un escenario de óptimo
  interior (`target_band`) real.
- Bucle B (persistencia/expertise/morfismos cross-run) y Bucle C (metabolismo/reserva finita).
- Term de disipación D_t (R4), hoy placeholder 0.0 en la recompensa.
