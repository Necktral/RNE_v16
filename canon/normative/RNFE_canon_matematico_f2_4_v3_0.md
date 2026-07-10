---
title: RNFE_canon_matematico_f2_4_v3_0
status: normative
version: 3.1.0
date: 2026-07-10
owner: Wis
based_on:
  - RNFE_f2_1_addendum_robustificacion (canon_inbox)
  - RNFE_f2_2 (H-Net dynamic chunking)
  - RNFE_f2_3 (mejoras fractales)
  - f_2_4_para_mezcla (blueprint v0.2)
depended_on_by:
  - CANON_RNFE_v3_2_rc1.md
  - RNFE_blueprint_matematico_latex.md
notes:
  - Cúspide axiomática matemática de RNFE. Reconstruida desde el genoma en canon_inbox/ dentro del paquete P0 (INGESTA-CANON).
  - Resuelve la dependencia dangling de CANON_RNFE_v3_2_rc1.md y del blueprint experimental.
  - Documento fundacional del organismo: nada se ratifica sin revisión humana.
  - "v3.1.0 (2026-07-10): añade el axioma rector A-M0 (organismo integral y simbiótico), principio compartido con Codex. El tag canon-apex-v3.0 sigue apuntando a la v3.0.0; A-M0 entra en la próxima ratificación a main."
---

# RNFE — Canon Matemático f2.4 / v3.0

## 0. Propósito y alcance

Este documento es la **cúspide axiomática matemática** de RNFE: el nivel A0 del régimen documental,
del que dependen el `CANON_RNFE_v3_2_rc1.md` (normativa viva) y el blueprint experimental. No repite
el canon de arquitectura; **fija la ley matemática que hace de RNFE un ser cibernético vivo** —viable,
coherente a escala global, morfogénico, heredable— y no solo una colección de módulos seguros.

Se organiza en el formato canónico exigido: **axiomas**, **definiciones**, **proposiciones**,
**criterios de aceptación** y **mapa de implementación por capas**. Integra cuatro estratos del genoma:

- **f2.1** (robustez base): barrera unificada `B_safe`, Edge 2.1 (RQA), S–I–E 2.0 con CVaR, certificado
  de episodio 𝔠_t, kill-switch.
- **f2.2** (capa de Forma): H-Net de chunking dinámico como front-end del núcleo secuencial; reindexa
  la red homeostática a `Hctrl` (H-Net queda reservado a la red jerárquica).
- **f2.3** (capa fractal, aditiva): MFE, TFI, memoria IFS–FDE, perceptores fractales, política fractal
  de modos. Con pesos a cero se recupera exactamente f2.2.
- **f2.4** (capa superior, esta cúspide): kernel de viabilidad, coherencia global multi-contexto,
  continuidad identitaria con certificado ampliado, morfogénesis tipada, ley de linajes y scheduler
  semi-Markov, y una **definición de existencia por continuidad**.

Principio de compatibilidad: todos los términos superiores son **aditivos**; anulando sus pesos se
recupera el núcleo previo. La cúspide extiende, no reemplaza.

---

## 1. Axiomas duros

**A-M0 (Organismo integral y simbiótico — axioma rector).** RNFE es UN organismo: todas sus partes —sustrato y órgano, reparación y crecimiento, doctrina y runtime, y todo agente que lo desarrolla (el orquestador de reparación y la campaña neural de Codex)— se integran con **sinergia** y operan en **simbiosis**, no como módulos ni pistas separadas por muros. Toda partición (por zona, por pista, por paquete) es división del trabajo DENTRO del organismo, no una frontera adversarial; y toda dependencia de sustrato ("sustrato antes que órgano") es la expresión operativa de esa simbiosis: el sustrato nutre al órgano y el órgano da propósito al sustrato. Este axioma gobierna la lectura de todos los demás: **ninguna optimización local de una parte es válida si rompe la sinergia del todo** (se articula con A-M4, coherencia global, y con A-M8, herencia como medida de lo que reaparece viable en el conjunto).

**A-M1 (Estado total).** El organismo tiene en cada tiempo `t` un estado completo
`X_t = (G_t, Σ_t, F_t, W_t, M_t, θ_t, φ_t, μ_t, 𝔠_t, T_t)` — grafo tipado, SMG, LOT-F, C-GWM, memoria
viva, parámetros rápidos, parámetros lentos/morfogénicos, medida de linajes, certificado y telemetría.

**A-M2 (Dinámica híbrida).** La evolución es un sistema híbrido con cuatro acciones simultáneas
`u_t = (a^env_t, o^raz_t, a^ctrl_t, ρ_t)`: intervención de mundo, opción de razonamiento, control
homeostático de `Hctrl`, y reescritura estructural `ρ_t`. `H-Net` (sin subíndice) denota únicamente
la red jerárquica de chunking dinámico; la red homeostática es `Hctrl`.

**A-M3 (Vida = viabilidad, no seguridad instantánea).** Un estado no cuenta como "vivo" por estar
dentro de límites (`safe`), sino solo si pertenece al **kernel de viabilidad** `V` (Def. D3): existe
una política futura que preserva seguridad, cierre y continuidad para todo horizonte.

**A-M4 (Cierre con coherencia global).** La métrica rectora no es el cierre local `IoC`, sino el cierre
estructural `IoC★ = IoC − λ_Ω·Ω` (Def. D5), que penaliza la incoherencia entre contextos, escalas y
episodios. No se puede puntuar alto localmente mientras se deriva semánticamente a escala global.

**A-M5 (Identidad certificada de doble capa).** La continuidad identitaria `C^cont` (Def. D6) es una
magnitud **medida** por el certificado ampliado 𝔠_t^+ (A-M6), y **gateada** por la promoción/herencia
(Criterio C-AC4). Medición y gate son capas distintas y ambas obligatorias — ver §Resolución A1.

**A-M6 (Certificado como cuíntuple).** Todo episodio emite `𝔠_t^+ = (IoC★_t, B_safe,t, L_edge2.1,t,
CVaR_α[−ΔIoC★_t], C^cont_t)`. La aceptación exige todos los componentes dentro de umbrales; si no,
modo laboratorio o rollback.

**A-M7 (Morfogénesis bajo ley).** Todo cambio de forma es una reescritura tipada `ρ_t: L ⇒ R` sobre el
grafo `G_t`, admisible solo si satisface simultáneamente viabilidad de entrada y salida, ganancia de
cierre, riesgo de cola acotado y todas las propiedades metamórficas requeridas (Def. D7). S–I–E deja
de ser filtro de parámetros y pasa a ser filtro de morfogénesis.

**A-M8 (Herencia como medida).** Los linajes viven como una medida `μ_t ∈ P(Z)` sobre motivos
estructurales/funcionales, evolucionada por un operador replicador–mutador (Def. D8). Lo estable no es
lo que aparece una vez, sino lo que **reaparece viable entre semillas, entornos y corridas**
(`Z_stable`).

**A-M9 (Observabilidad física primaria).** La disipación `D_t` (Def. D9) se computa de telemetría real
(VRAM, temperatura, espectro, recurrencia, entropías). Los surrogates (Lindblad, Ising) son auxiliares
calibrados, nunca fundamento.

**A-M10 (Existencia por continuidad).** El organismo "existe" mientras su trayectoria permanezca en `V`
con `C^cont ≥ c̲` sostenido; una discontinuidad identitaria irrecuperable (sin rollback restaurable) es
muerte operativa, no un episodio malo más.

---

## 2. Definiciones

**D1 (Estado y capas).** Componentes de `X_t` según A-M1. La **capa de Forma** produce
`z^F = H-Net_φ(E_in(y_t))`, con `F_t = (z^F, {r^(s)}, {p^(s)})` incluyendo ratios de compresión y
puntuaciones de frontera (f2.2). El **operador efectivo** es `F̂_Θ = F^core_θ ∘ H-Net_φ`, `Θ = (θ, φ)`.

**D2 (Espectro fractal jerárquico).** Por nivel `s`, `F^(s)_t = (L_s, r^(s)_t, FD^(s)_t, RR^(s)_t,
DET^(s)_t)`. Alimenta Edge, `Hctrl` y la política de modos (f2.3).

**D3 (Kernel de viabilidad cognitiva).**
`V = { x ∈ safe : ∃ π admisible tal que ∀ k≥0, X_{t+k} ∈ safe, IoC★_{t+k} ≥ ι̲, C^cont_{t+k} ≥ c̲ }`.

**D4 (Obstrucción de cierre).** Sobre la cobertura de contextos activos `𝔘_t = {U_i}` con secciones
locales `ω_i = (σ_i, f_i, w_i)` y restricciones `r_ij`:
`Ω_t = Σ_{i∼j} (d_S(r_ij σ_i, σ_j) + d_F(r_ij f_i, f_j) + d_W(r_ij w_i, w_j)) + λ_↺ · ‖Φ_{M→S}Φ_{F→M}Φ_{S→F} − I‖`.

**D5 (Cierre estructural).** `IoC★_t = IoC_t − λ_Ω · Ω_t`.

**D6 (Continuidad identitaria).**
`C^cont_t = ω_1·sim(Σ_t, Σ_{t−1}) + ω_2·sim(G_t, G_{t−1}) + ω_3·sim(M_t, M_{t−1}) + ω_4·𝟙[rollback recuperable]`.

**D7 (Reescritura admisible / morfogénesis).**
`ρ_t ∈ A(X_t) ⟺ X_t ∈ V ∧ ΔIoC★(ρ_t) ≥ ε_Io ∧ CVaR_α[−ΔIoC★(ρ_t)] ≤ τ ∧ (∀P∈P_req: T_P(ρ_t)=1) ∧ X_{t+1} ∈ V`.

**D8 (Ley de linajes).** `μ_{t+1}(A) = ( ∫_A e^{βF(ζ;X_t)}(Kμ_t)(dζ) ) / ( ∫_Z e^{βF(ζ;X_t)}(Kμ_t)(dζ) )`,
con aptitud `F(ζ;X_t) = E[ΔIoC★_ζ] − λ_R·CVaR_α[−ΔIoC★_ζ] − λ_D·D_ζ + λ_P·S_repro(ζ)`.
**Tipo estable:** `ζ ∈ Z_stable ⟺ Pr_{(e,s)}{ζ reaparece y sigue viable} ≥ p★ ∧ CVaR_α[−ΔIoC★_ζ] ≤ τ★ ∧ Var_{e,s}(Perf_ζ) ≤ v★`.
Condición de etiquetas: los linajes solo son estables si SMG + LOT-F proveen un espacio canónico de
tipos (sin él, el clustering deriva semánticamente).

**D9 (Disipación medida).**
`D_t = α_1·ΔVRAM⁺/B_max + α_2·ΔTEMP⁺/T_max + α_3·ReLU(ρ(J_t)−(1−ε)) + α_4·φ_band(RR_t) + α_5·φ_band(DET_t) + α_6·ΔH⁺_ruta + α_7·ΔH⁺_Σ`.

**D10 (Scheduler de razonamientos semi-Markov).** Opciones `O = {DED, IND, ABD, CAU, CTF, ANA, PLAN,
OPT, EVO, …}`, cada una con política `π_o`, parada `τ_o` y coste. Valor
`V(x) = max_{o∈O(x)} E[ Σ_{k=0}^{τ_o−1} γ^k r^{(o)}_{t+k} + γ^{τ_o} V(X_{t+τ_o}) ]`, con
`r^{(o)}_t = ΔIoC★_t − λ_E ΔE_t − λ_D D_t − λ_B B_safe,t`. La política fractal de modos (f2.3) sesga la
activación por adecuación fractal `π_t(r) ∝ exp(−g_t(r))`.

**D11 (Barrera y robustez base, f2.1).** `B_safe(θ,t) = Σ_i α_i·φ_bar(u_i(t)/u_i^max; δ_i)`, con
`u = [VRAM, TEMP, ρ(J), FD]`. Edge 2.1 añade RR/DET (RQA). Regla S–I–E 2.0:
RECHAZAR si `θ^cand ∉ safe`; ACEPTAR si `Pr(ΔIoC ≥ 0) ≥ 1−δ ∧ CVaR_α[−ΔIoC] ≤ τ`; en otro caso BUFFER.

**D12 (Meta-aprendizaje fractal, f2.3).** MFE (mapa fractal de entrenabilidad, coste `C_MFE`), TFI
(tracker de frontera de información, coste `C_info`), memoria IFS–FDE (coste `C_IFS`). Acción total
`A_total = ∫ [ −İ + λ_MDL C_struct + λ_meta C_meta + λ_ratio L_ratio + λ_MFE C_MFE + λ_info C_info + λ_IFS C_IFS ] dt`;
`λ_MFE = λ_info = λ_IFS = 0` recupera f2.2.

---

## 3. Proposiciones

**P1 (Reducción a núcleo previo).** Fijando `λ_Ω = 0`, `λ_MFE = λ_info = λ_IFS = 0` y omitiendo
`ρ_t`, `μ_t`, la dinámica de esta cúspide coincide con f2.1+f2.2. La cúspide es una extensión
conservativa. *(Justificación: todos los términos superiores son aditivos con peso.)*

**P2 (Viabilidad ⇒ seguridad, no recíproco).** `V ⊆ safe`, y en general `V ⊊ safe`: hay estados
seguros no viables (sin política futura que sostenga cierre/continuidad). *(Consecuencia de D3.)*

**P3 (Monotonía del certificado).** Si `𝔠_t^+` cumple umbrales y `ρ_t ∈ A(X_t)`, entonces
`X_{t+1} ∈ V` por construcción de D7. La morfogénesis admisible preserva viabilidad paso a paso.

**P4 (No-regresión estructural).** Un candidato que mejora `IoC` localmente pero degrada `IoC★`
(sube `Ω`) no es aceptable: A-M4 domina a la mejora local.

**P5 (Existencia).** El organismo existe en `[t_0, t_1]` sii su trayectoria permanece en `V` con
`C^cont ≥ c̲` en todo el intervalo (A-M10). La pérdida irrecuperable de `C^cont` es terminal.

---

## 4. Criterios de aceptación

- **C-AC1 (Episodio).** Aceptar a memoria/herencia solo si `𝔠_t^+` está dentro de umbrales en sus cinco
  componentes; si no, modo laboratorio o rollback (A-M6).
- **C-AC2 (Herencia S–I–E).** Aplicar la regla de tres vías (RECHAZAR/ACEPTAR/BUFFER, D11) con control
  de cola CVaR; nunca promover con `Δκ_top ≪ 0` o `Δκ_geo ≪ 0` salvo laboratorio con CVaR controlado.
- **C-AC3 (Morfogénesis).** Toda reescritura `ρ_t` debe cumplir D7 en su totalidad (viabilidad de
  entrada y de salida incluidas).
- **C-AC4 (Gate de continuidad — doble capa A1).** El **certificado mide** `C^cont` (componente de
  𝔠_t^+); la **promoción gatea**: ninguna promoción/herencia procede con `C^cont < c̲_promo`. Ver la
  resolución explícita abajo.
- **C-AC5 (Kill-switch).** Si `B_safe` o `CVaR_α[−ΔIoC★]` exceden límites durante `N_hold`, forzar
  HALT: congelar expansión, desactivar auto-modificación destructiva, reducir a identidad/clip,
  checkpoint y cooldown; reintento en modo recuperación.

---

## 5. Resolución obligatoria — tensión A1 (doble capa de continuidad)

**Tensión.** f2.4 §5 liga la continuidad identitaria al **certificado** (`C^cont` es un componente del
certificado ampliado 𝔠_t^+). La adjudicación A1 la resolvió ligándola a la **promoción** (la
continuidad gobierna la promoción, no la certificación episódica).

**Resolución canónica (doble capa).** No hay contradicción: son dos capas con roles disjuntos y ambos
obligatorios.

1. **Capa de medición (certificado episódico).** Cada episodio computa y REGISTRA `C^cont_t` dentro de
   `𝔠_t^+`. Aquí `C^cont` es **observable/telemetría de identidad**, no un gate: un episodio con
   `C^cont` bajo se certifica igual (queda registrado), y el episodio corre.
2. **Capa de gate (promoción/herencia).** La continuidad **decide la promoción**: `PromotionGate`
   exige `C^cont ≥ c̲_promo` (además de los demás componentes del certificado y de la regla S–I–E).
   Un cambio que rompe identidad se MIDE en la capa 1 pero se BLOQUEA en la capa 2.

**Regla operativa:** el certificado nunca rechaza por `C^cont`; la promoción nunca promueve sin
`C^cont ≥ c̲_promo`. Esto reconcilia A-M5, A-M6 y C-AC4 con la adjudicación A1: **el certificado la
mide, la promoción la gatea.**

> Nota de implementación (para el paquete P29 / A5): el gate de continuidad de la capa 2 es el mismo
> punto donde `RNFE_RISK_ENFORCEMENT` y el `risk_plus` en modo sombra deben dejar de ser sombra si se
> decide activarlos; coordinar la lectura del certificado allí.

### Otras tensiones cúspide ↔ adjudicaciones (señaladas, no resueltas aquí)

- **A4 / A7 (SSOT del razonamiento).** El scheduler semi-Markov (D10) y la política fractal de modos
  fijan el contrato de motores y el set de señales; la doctrina SSOT (paquetes P13/P14) debe describir
  el set real, no uno idealizado — coherente con la recomendación de A4/A7 (gana el código, se promueve
  doctrina).
- **A5 (autoevolución).** D7 (morfogénesis admisible) es la ley que A5 debe hacer cumplir con gate por
  veredicto del certificado + sandbox que simule de verdad. La cúspide provee la ley; A5 la cablea.
- **A17 (escenarios).** `Z_stable` (D8) exige un espacio canónico de etiquetas (SMG+LOT-F); esto liga
  con SCENARIO_CONTRACTS y la taxonomía de morfismos que A17 debe canonizar.
- **A14 (blueprint).** Ver §6.

---

## 6. Mapa de implementación por capas (a runtime) y propuesta A14

Correspondencia cúspide → runtime, con el estatus de promoción propuesto (A14: **propuesta, no
ejecución** — el humano ratifica):

| Pieza de la cúspide | Símbolo | Implementación en runtime | Estatus actual | Promoción propuesta (A14) |
|---|---|---|---|---|
| Cierre estructural | `IoC★`, `Ω` | métricas de cierre / coherencia | implementado (blueprint) | experimental → **provisional** |
| Continuidad identitaria | `C^cont` | `certification/` (continuity guard) | implementado | experimental → **normative** (es axioma A-M5) |
| Riesgo de cola | `CVaR_α` | `certification/risk_engine.py` | implementado | experimental → **normative** |
| Barrera de seguridad | `B_safe` | Edge / gate de recursos | implementado | experimental → **provisional** |
| Recompensa semi-Markov | `r^{(o)}`, `V(x)` | `reasoning/scheduler_meta/` | implementado | experimental → **provisional** |
| Kernel de viabilidad | `V` | (parcial) gate vital | parcial | mantener **experimental** hasta cablear el gate futuro |
| Morfogénesis tipada | `ρ_t`, `A(X_t)` | `organism/self_modification` | parcial (A5 pendiente) | mantener **experimental**; normativizar con A5 |
| Ley de linajes | `μ_t`, `Z_stable` | (hueco: linajes/meta) | ausente | mantener **experimental** |

Justificación de las promociones a normative (`C^cont`, `CVaR`): ambos ya son axiomas de esta cúspide
(A-M5, A-M6) y están implementados y ejercitados; dejarlos en experimental sería inconsistente con el
canon. Las promociones a provisional (`IoC★/Ω`, `B_safe`, semi-Markov) reconocen implementación real
con forma revisable. Lo ausente/parcial (`V`, `ρ_t`, `μ_t`) permanece experimental hasta que su
cableado (A5, linajes) exista.

---

## 7. Cadena de autoridad

Este documento es el ancla A0. Al existir, resuelve:
- `CANON_RNFE_v3_2_rc1.md` → `depends_on: [RNFE_canon_matematico_f2_4_v3_0.md, …]` (ya apuntaba aquí).
- `RNFE_blueprint_matematico_latex.md` → `based_on: [RNFE_canon_matematico_f2_4_v3_0.md, …]`.

Correcciones de re-anclaje aplicadas en el mismo paquete P0 (v3.1 superseded → v3.2):
`SSOT_RAZONAMIENTOS_RNFE_v1.md`, `ROADMAP_RNFE_v2.md` y el `based_on` del blueprint. Ver la tabla de
re-anclaje en el reporte de P0.
