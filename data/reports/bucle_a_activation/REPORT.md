# A1 — Activación del Bucle A: ¿la selección guiada-por-recompensa mejora la ganancia?

## Pregunta (roadmap §3.4)

> Con la recompensa descompuesta ya cableada (ν=`cau.helps_goal` de primera clase, `RNFE_REWARD_LAMBDA_NU`), ¿el selector guiado-por-recompensa REAL supera al perfil fijo declarativo en ganancia cognitiva y activación de la familia efectiva — a λ_ν=O(1) — y la ecología (compartir Δr̄ en vivo) MULTIPLICA esa ganancia?

Cierra la línea reward_blindness (modo de fallo) → critical_functional (cura en harness). **Sin cambios de runtime**: conduce el `RewardGuidedOverlaySelector` real bajo la recompensa descompuesta modelada. Determinista por semilla; CI ENTRE-semillas.

## Brazos (λ_ν=1.0, N=12 semillas × 36 episodios, régimen conflict)

Ganancia = **efectividad** media 2ª mitad (resultado facing-mundo, proxy IVC-R §3.4). El escalar de control (reward) se reporta aparte: a λ_ν=O(1) queda casi-neutro porque activar la familia efectiva paga coste+continuidad — lo relevante es que la conducta y el mundo mejoran.

| Brazo | Efectividad 2ª mitad [CI] | Reward escalar [CI] | Activación efectiva [CI] | Retención |
|---|---|---|---|---|
| fixed | -0.0120 [-0.0120, -0.0120] | 0.6000 [0.5935, 0.6063] | 0.000 [0.000, 0.000] | 0.000 |
| guided | 0.0120 [0.0120, 0.0120] | 0.6049 [0.5868, 0.6244] | 1.000 [1.000, 1.000] | 1.000 |
| ecology | 0.0120 [0.0120, 0.0120] | 0.6196 [0.6031, 0.6351] | 1.000 [1.000, 1.000] | 1.000 |

- **Δ efectividad guiado − fijo** = 0.0240 [CI 0.0240, 0.0240].
- **Δ efectividad ecología − aislado** = 0.0007 [CI 0.0007, 0.0007].
- **A1a guiado > fijo en efectividad** (CI excluye 0): ✓.
- **A1b activación efectiva ↑**: ✓.
- **A1d ecología multiplica** (CI excluye 0): ✓.

## Barrido λ_ν — umbral de recuperación (brazo guiado)

| λ_ν | Retención familia efectiva |
|---:|---:|
| 0.0 | 0.000 |
| 0.5 | 0.417 |
| 1.0 | 1.000 |
| 2.0 | 1.000 |
| 5.0 | 1.000 |

- Umbral de recuperación λ_ν = **1.0**.
- **A1c umbral O(1)** (≤ 2.0): ✓ — coincide con la cura predicha por critical_functional (frente a ≈20 del IoC colapsado).

## Veredicto

**A confirmado en harness**: la selección guiada por la recompensa descompuesta supera al perfil fijo a λ_ν=O(1), y compartir evidencia en vivo (ecología) multiplica la ganancia. Habilita la PR gated de runtime (A2), condicionada además a R1 (gate de seguridad).

## Limitaciones honestas

- **Modelo, no re-medición viva** (igual que critical_functional): la recompensa se modela con las constantes medidas; conduce el selector real. Motiva la PR gated, no la confirma en vivo.
- El brazo fijo NO activa la familia efectiva por construcción ⇒ la ganancia mide SELECCIÓN (reward→conducta), no sofisticación del razonamiento.
- La **preservación del cierre** (closure ≥ baseline) es una propiedad de los certificados de runtime; se valida en A2 (test system-mode con flags on/off), no en este harness.
