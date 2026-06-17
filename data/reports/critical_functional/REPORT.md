# Funcional Crítico J(h|X): ¿la descomposición cura la ceguera de la recompensa?

## Afirmación

> Tratar la viabilidad/coherencia-causal (ν) como criterio aditivo de PRIMERA CLASE en un funcional multicriterio `J(h|X) = w1·κ + w2·σ + w3·ρ + w4·α + w5·ν − w6·u` recupera la familia efectiva con un peso pequeño λ_ν=O(1), eliminando la inflación ~40× del umbral que sufre el reward de coherencia colapsado; y, con un control `target_band`, activa la familia efectiva SOLO cuando hay brecha de viabilidad (especificidad limpia).

Continúa el estudio de ceguera de recompensa (modo de fallo MEDIDO). Aquí se prueba en el harness si la DESCOMPOSICIÓN propuesta por el ADR cura el fallo. **Sin cambios de runtime.**

## Hipótesis (pre-registradas)

- **G1 (cura / sin inflación):** umbral de recuperación bajo J ≈ O(1), MUY por debajo del umbral del IoC-colapsado (≈20). Dosis-respuesta de retención vs λ.
- **G2 (especificidad, el H2 con control limpio):** con `target_band`, J/ν activa la familia efectiva fuera de banda (con brecha) y NO en banda (sin brecha = óptimo interior).
- **G3 (necesidad del canal):** quitar ν reproduce la supresión; quitar σ (no-causal) no ⇒ el efecto es específico del canal de viabilidad (análogo estructural del control de ruido H3).
- **Falsación:** si J también exige λ≈20, o activa en banda, o cualquier canal extra recupera, la afirmación de descomposición es FALSA y se reporta así.

Constantes MEDIDAS (del estudio previo, no ajustadas a conveniencia): continuidad domina IoC con peso 0.45 (`ioc_proxy.py`); desviar baja IoC 0.24 (0.888→0.646); el margen de efectividad real es diminuto (~0.012); ν es el canal booleano `cau.helps_goal`∈{0,1} (`core_inference.py`).

## G1 — Cura: umbral de recuperación J vs IoC-colapsado (conflicto)

N=400 semillas × 30 episodios por celda. Mismo ruido por-episodio en ambos; difiere la SEÑAL (canal booleano limpio vs margen diminuto) y el PESO (continuidad 0.45 → 1/7).

| λ | Retención IoC-colapsado | Retención J (ν 1ª clase) |
|---:|---:|---:|
| 0.0 | 0.000 | 0.005 |
| 0.5 | 0.000 | 0.412 |
| 1.0 | 0.000 | 0.968 |
| 2.0 | 0.000 | 1.000 |
| 5.0 | 0.203 | 1.000 |
| 20.0 | 1.000 | 1.000 |
| 50.0 | 1.000 | 1.000 |

- Umbral de recuperación **colapsado** = 20.0; **J** = 1.0; **ratio** = 20.0×.
- **G1 J recupera con λ pequeño**: ✓.
- **G1 umbral colapsado ≫ J**: ✓.

## G2 — Especificidad: control `target_band` (el H2 que faltó)

J a λ_ν=1.0 sobre regiones con/ sin brecha. `band_in` = óptimo interior (mundo ya en banda ⇒ desviar no ayuda ⇒ ninguna acción tiene ventaja).

| Región | Activación familia efectiva |
|---|---:|
| conflict | 0.965 |
| band_out | 0.965 |
| band_in | 0.017 |

- **G2 activa donde hay brecha**: ✓.
- **G2 inactiva en banda (sin-brecha)**: ✓ — este es el control limpio que el térmico-minimize no ofrecía.

## G3 — Necesidad del canal: ablación ν vs σ (conflicto)

| Variante | Retención familia efectiva |
|---|---:|
| J_full | 1.000 |
| J_sin_nu | 0.018 |
| J_sin_sigma | 1.000 |

- **G3 ν necesario** (quitarlo suprime): ✓.
- **G3 σ no necesario** (quitarlo no suprime): ✓ ⇒ el efecto es específico del canal de viabilidad.

## Veredicto

Las hipótesis se **confirman** en el harness: la descomposición (ν de primera clase, continuidad degradada) cura la inflación del umbral, es específica del canal de viabilidad, y respeta la especificidad en el control sin-brecha.

## Limitaciones honestas

- **Es un modelo, no una re-medición viva.** El harness conduce el selector real con canales MODELADOS usando las constantes medidas en el estudio previo (continuidad 0.45, caída 0.24, margen ~0.012, ν booleano). La cura es una PREDICCIÓN del modelo que motiva la PR gated de runtime (descomponer IoC), no una medición de la arquitectura viva.
- La afirmación es sobre **especificación/forma de la evaluación** (re-ponderar + usar el canal causal limpio), no sobre sofisticación del razonamiento.
- `target_band` está modelado en el harness; un escenario de óptimo interior en runtime sigue siendo trabajo futuro para confirmarlo en vivo.
- Una sola familia efectiva y un conjunto de canales; generalización a tareas ricas pendiente.
