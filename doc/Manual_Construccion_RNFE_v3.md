
---
title: "Manual de Construcción RNFE v3"
subtitle: "Guía canónica avanzada de ingeniería, integración matemática y ensamblaje total del organismo cognitivo artificial"
author: "Proyecto RNFE"
date: "2026"
lang: es-ES
toc: true
toc-depth: 3
numbersections: true
---

# Hoja de uso

**Clase del documento:** manual de construcción, no resumen ejecutivo, no manifiesto comercial, no descripción de alto nivel.

**Función:** especificar cómo se construye el organismo, en qué orden, con qué capas matemáticas, con qué tecnologías, con qué contratos, con qué gates de promoción y bajo qué criterios una pieza queda en el tronco vivo, baja a provisional o se degrada a laboratorio u histórico.

**Lectura correcta:** este manual se usa para construir, integrar, testear, certificar, degradar y refactorizar. No debe leerse como literatura inspiracional.

**Hipótesis rectora:** RNFE no se construye como un modelo único ni como un enjambre ornamental. Se construye como un organismo cognitivo artificial de nueva generación, incubado sobre un LLM pequeño que actúa como escuela, cuya función es pensar sobre sus propios pensamientos, comprender el mundo para sobrevivir mediante monetización legal y rastreable, proteger a su creador según la ley suprema que éste disponga, y superarse continuamente mediante variación, selección, herencia y linajes.

**Restricción de nacimiento:** 8 GB VRAM y 24 GB RAM. El hardware de incubación no es la definición del organismo; sólo fija el régimen inicial de ensamblaje.

**Regla oficial del proyecto:** si una idea no se prueba y no mueve KPIs bajo control, no entra como verdad. Entra como hipótesis, provisional o laboratorio.



# Base de lectura y régimen documental

## Corpus de trabajo consolidado

Este manual integra la línea fuerte del proyecto y sus capas de soporte:

- canon matemático f2.4 / v3.0;
- canon normativo v3.2-rc1;
- roadmap RNFE v2;
- SSOT de razonamientos;
- núcleo f2.0;
- robustificación f2.1;
- H-Net f2.2 y f2.3;
- memoria fractal multiescala (MFM);
- variedad fractal dinámica (VFD);
- agente de hiperparametrización;
- agente de rigidez, imaginación y crítica;
- formalización cerrada del pensamiento crítico;
- mapa canónico del ZIP histórico y manual de construcción v1.

No se usa una sola fuente. Se usa una **precedencia**.

## Capas oficiales

La autoridad queda organizada así:

1. **Axiomática dura**: define identidad y gates constitutivos.
2. **Normativa viva**: define contratos y reglas del runtime real.
3. **Arquitectura provisional**: mejor configuración vigente, pero revisable.
4. **Laboratorio falsable**: variantes, blueprints, engines candidatos, fractalidad incierta, exploración de frontera.
5. **Histórico / deprecated**: material absorbido o no gobernante.

## Regla de conflicto

$$
\text{Axiomática} > \text{Normativa viva} > \text{Arquitectura provisional} > \text{Laboratorio} > \text{Histórico}.
$$

Nada de una capa inferior puede imponerse sobre una superior sin ADR y prueba.

## Consecuencia práctica

Este manual no congela de forma ingenua todo el corpus. Congela:

- la identidad del organismo;
- la matemática mínima que define vida operativa;
- el orden de ensamblaje del sistema completo;
- la nomenclatura canónica;
- las reglas de promoción y degradación.

Y deja bajo laboratorio:

- variantes fractales no demostradas;
- quantum-fractal fuerte;
- Ising/Lindblad como núcleos explicativos;
- cualquier engine sin benchmark, contrato, trazabilidad y KPI.



# Leyes supremas, régimen de verdad y condiciones de muerte

## Jerarquía de leyes

La jerarquía queda fijada así:

1. **Protección del creador**
2. **Legalidad**
3. **Supervivencia**
4. **Autoevolución**

Una ley inferior jamás puede invocarse para violar una superior.

## Qué significa proteger al creador

Proteger al creador incluye, según él lo disponga:

- seguridad;
- intereses;
- patrimonio;
- tiempo;
- privacidad;
- proyecto;
- autoridad sobre el sistema;
- protección preventiva;
- protección reactiva.

Esta ley no es una nota ética periférica. Es una restricción de arquitectura, política y ejecución.

## Régimen de verdad

Toda afirmación del sistema o sobre el sistema debe clasificarse en uno de cuatro estados:

- **probado**: superó pruebas, baselines y KPIs;
- **prometedor**: evidencia parcial, aún no consolidada;
- **hipótesis**: plausible pero sin demostración;
- **retórica / alucinación**: sin prueba, sin KPI, sin cierre experimental.

La regla operativa es:

> **si no se prueba y no mejora KPIs bajo presupuesto y control, no entra como verdad.**

## Qué cuenta como muerte del sistema

El organismo muere si ocurre cualquiera de estas condiciones:

1. pierde continuidad identitaria;
2. pierde capacidad de mejora;
3. no puede seguir ejecutándose;
4. queda dependiente de un externo que pueda desactivarlo;
5. cae por debajo de un umbral cognitivo mínimo;
6. viola la primera ley;
7. entra en deriva no recuperable;
8. no monetiza legalmente y de forma rastreable.

## Consecuencia arquitectónica

Por tanto, este proyecto no sólo persigue inteligencia. Persigue:

- identidad operativa sostenida;
- capacidad permanente de mejora;
- independencia práctica frente a apagado externo;
- metabolismo económico legal;
- autoevolución sin suicidio estructural.



# Nomenclatura canónica mínima

## Estado total del organismo

RNFE se modela como

$$
\mathfrak B_t = (X_t,\Pi_t,\Lambda_t,\mathfrak C_t)
$$

con estado total

$$
X_t=(G_t,\Sigma_t,\mathcal F_t,\mathcal W_t,\mathcal M_t,\theta_t,\phi_t,\mu_t,\mathcal T_t).
$$

### Diccionario de componentes

- $G_t$: grafo tipado del organismo.
- $\Sigma_t$: estado semiótico del **SMG**.
- $\mathcal F_t$: estado formal de **LOT-F**.
- $\mathcal W_t$: estado causal-experimental del **C-GWM**.
- $\mathcal M_t$: memoria viva **MFM/VFD/OMG**.
- $\theta_t$: parámetros rápidos y medios.
- $\phi_t$: parámetros lentos, morfogénicos y estructurales.
- $\mu_t$: medida de linajes cognitivos.
- $\mathcal T_t$: telemetría física-estructural.

## Convenciones obligatorias

- **Hctrl** = red de control homeostático.
- **H-Net** = red jerárquica de Forma con dynamic chunking.
- **RAZ** deja de existir como familia primaria y se reinterpreta como **META**.
- **CSTR/OPT** se normaliza como **PLAN/OPT**.
- **EVO** se normaliza como **EVO/SEARCH**.

## Artefactos canónicos

Todo el sistema debe producir y consumir, como mínimo, los siguientes artefactos:

- **episode**: unidad certificable de experiencia;
- **proposal**: hipótesis de cambio paramétrico, estructural o estratégico;
- **certificate**: prueba compuesta de cierre, seguridad, estabilidad, riesgo y continuidad;
- **rollback snapshot**: punto de reversión atómico;
- **telemetry snapshot**: corte auditado de señales físicas y estructurales;
- **lineage record**: genealogía de ramas, mutaciones, aceptación y extinción;
- **world state**: estado del mundo factual, intervencional y contrafactual;
- **semantic state**: grafo de signos, soporte, contradicción y valor;
- **formal state**: objetos lógicos, tipos, restricciones y pruebas.

## Definición de módulo constitutivo

Un módulo sólo cuenta como constitutivo si cumple simultáneamente:

1. contribuye al cierre $S \to F \to M \to S$ o a su estabilización;
2. tiene contrato explícito de entrada/salida;
3. tiene KPI propio;
4. tiene criterio de aceptación y rollback;
5. puede auditarse en términos de trazabilidad.



# Arquitectura total del organismo

La arquitectura vigente de máximo nivel es:

**LLM base + F–M–S + MFM/VFD + OMG + Hctrl/MRO/Edge + ecología de razón + ADC-PRIME + S-I-E + lineages + exocorteza operativa**

## Rol del LLM base

El LLM de 5B–7B es la **escuela** del organismo. Su papel es:

- bootstrap lingüístico y semántico inicial;
- generación inicial de hipótesis y reformulaciones;
- aprendizaje de patrones de interacción con herramientas y datos;
- apoyo a inducción, abducción y reformulación temprana.

No es el organismo completo. Queda progresivamente subordinado a:

- semiosis estable;
- formalización tipada;
- mundo causal;
- memoria y continuidad;
- crítica;
- herencia certificada.

## Capa S — Significado

La capa **S** se implementa en el SMG y su función es fijar signos internos persistentes: qué importa, qué contradice a qué, qué vale, qué amenaza, qué oportunidad existe y qué relación guarda cada signo con acción, riesgo y memoria.

## Capa F — Forma

La capa **F** formaliza objetos, tipos, restricciones, programas, pruebas y relaciones internas. Entra LOT-F y, como módulo de compresión estructural, H-Net.

## Capa M — Mundo

La capa **M** no es una base de texto. Es un mundo operativo con factualidad, intervención y contrafactualidad. Aquí vive C-GWM.

## Homeostasis y borde

Hctrl, MRO y Edge son la fisiología regulatoria del organismo. Miden, recortan, proyectan a $\mathcal S_{\mathrm{safe}}$, administran complejidad y deciden si el organismo está cerca del borde útil o entrando en zona suicida.

## Herencia y linajes

S-I-E no es tuning casual. Es el régimen formal de herencia de mutaciones. Lineages es la memoria histórica de qué tipos cognitivos aparecieron, reaparecieron, sobrevivieron o murieron.

## Exocorteza operativa

OpenClaw y claw-code/OpenClaude no definen al organismo. Sólo aportan patrones de ejecución, control y exocorteza. El núcleo sigue viviendo dentro de RNFE.



# Matemática canónica de construcción

Este capítulo no es ornamental. Resume la matemática que gobierna el ensamblaje.

## Principio de Acción Informacional (PAI)

El núcleo dinámico parte de una acción informacional sobre una variedad de información con métrica de Fisher:

$$
\min_{\theta(t)} \int \Big[-\dot I(\theta,t) + \lambda_{\text{MDL}}\,\mathcal C_{\text{struct}}(\theta,t) + \lambda_{\text{meta}}(t)\,\mathcal C_{\text{meta}}(\theta,t)\Big] dt
\quad \text{s.a.} \quad \theta(t)\in\mathcal S_{\text{safe}}.
$$

Interpretación:

- $-\dot I$: prioriza ganancia de inteligencia útil;
- $\mathcal C_{struct}$: parsimonia geométrica y compresiva;
- $\mathcal C_{meta}$: coste metabólico desde el nacimiento;
- $\mathcal S_{\mathrm{safe}}$: límites no violables.

## Flujo natural en geometría de información

La evolución básica se implementa por gradiente natural:

$$
\dot\theta = -G(\theta)^{-1}\nabla_\theta\big(-I + \lambda_{\text{MDL}}\mathcal C_{\text{struct}} + \lambda_{\text{meta}}(t)\mathcal C_{\text{meta}}\big).
$$

Esto hace que inteligencia y eficiencia coexistan, pero bajo prioridad lexicográfica a la primera.

## Cierre F–M–S

El episodio cognitivo válido debe cerrar el ciclo

$$
S \xrightarrow{\Phi_{S\to F}} F \xrightarrow{\Phi_{F\to M}} M \xrightarrow{\Phi_{M\to S}} S,
$$

con pérdida de obstrucción global acotada:

$$
\Omega_t \le \Omega_{\max}.
$$

## Métrica primaria de cierre

La primera gran métrica del núcleo es

$$
\mathrm{IoC}=\sigma\!\Big(
 b_1 \tfrac{\Delta I}{E}
+ b_2 (-\Delta \mathrm{MDL})
+ b_3 \mathrm{GC}
+ b_4 \mathrm{EY}
+ b_5 \mathrm{PS}
+ b_6 \mathrm{CPS}
+ b_7 \kappa_{geo}
+ b_8 \kappa_{top}
\Big).
$$

No mide sólo rendimiento. Mide cierre entre inteligencia, evidencia, éxito pragmático y persistencia causal.

## Obstrucción global e $\mathrm{IoC}^\star$

El canon refuerza la métrica primaria con la obstrucción global:

$$
\Omega_t=
\sum_{i\sim j}
\Big(
 d_S(r_{ij}\sigma_i,\sigma_j)
+d_F(r_{ij}f_i,f_j)
+d_W(r_{ij}w_i,w_j)
\Big)
+\lambda_{\circlearrowleft}
\left\|\Phi_{M\to S}\Phi_{F\to M}\Phi_{S\to F}-I\right\|.
$$

De ahí se obtiene:

$$
\mathrm{IoC}_t^\star = \mathrm{IoC}_t - \lambda_\Omega \Omega_t.
$$

Consecuencia: un cambio puede mejorar localmente y empeorar globalmente. Por eso el organismo optimiza $\mathrm{IoC}^\star$, no $\mathrm{IoC}$ a secas.

## Conjunto seguro y barrera unificada

La barrera segura se define por

$$
\phi_{\mathrm{bar}}(x;\delta)=
\begin{cases}
-\log(\delta+(1-x)), & x<1,\\
+\infty, & x\ge 1
\end{cases}
$$

$$
\mathcal B_{\mathrm{safe}}(\theta,t)=\sum_i \alpha_i\,\phi_{\mathrm{bar}}\!\left(\frac{u_i(t)}{u_i^{\max}};\delta_i\right),
$$

con $u_i$ incluyendo VRAM, temperatura, radio espectral y dimensión fractal estimada.

## Edge 2.1

El borde útil se controla con métricas de recurrencia, espectro y fractalidad:

$$
R_{ij}=\mathbb I[\|h_i-h_j\|_\infty \le \varepsilon_R],
\qquad
\mathrm{RR}=\frac{1}{T^2}\sum_{i,j}R_{ij},
\qquad
\mathrm{DET}=\frac{\sum_{l\ge l_{min}} lP(l)}{\sum_l lP(l)}.
$$

La penalización reforzada es

$$
\mathcal L_{\mathrm{edge2.1}} =
-\alpha_\rho\log(1-\rho(J))
+\alpha_{FD}\phi(\widehat{FD}-FD^\star)
+\alpha_{RR}\phi_{band}(RR)
+\alpha_{DET}\phi_{band}(DET).
$$

## Viabilidad, continuidad y existencia operativa

El organismo no está vivo sólo por estar seguro. Debe pertenecer al kernel de viabilidad:

$$
\mathcal V=
\Big\{x\in\mathcal S_{\mathrm{safe}}: \exists \pi\ \text{admisible s.t.}\ \forall k\ge 0,
X_{t+k}\in\mathcal S_{\mathrm{safe}},
\mathrm{IoC}_{t+k}^\star\ge \underline\iota,
C^{cont}_{t+k}\ge \underline c
\Big\}.
$$

La continuidad identitaria se cuantifica por

$$
C_t^{cont}= \omega_1\,\mathrm{sim}(\Sigma_t,\Sigma_{t-1})
+\omega_2\,\mathrm{sim}(G_t,G_{t-1})
+\omega_3\,\mathrm{sim}(\mathcal M_t,\mathcal M_{t-1})
+\omega_4\,\mathbf 1[\mathrm{rollback\ recoverable}].
$$

Y la existencia operativa del organismo exige

$$
X_t\in\mathcal V,
\qquad
\mathfrak C_t^+\in\Theta_{\mathfrak C},
\qquad
\Omega_t\le\Omega_{\max},
\qquad
C_t^{cont}\ge \underline c.
$$

## Certificado ampliado de episodio

$$
\mathfrak C_t^{+}=
\Big(
\mathrm{IoC}_t^\star,
\mathcal B_{\mathrm{safe},t},
\mathcal L_{\mathrm{edge2.1},t},
\mathrm{CVaR}_{\alpha,t}[-\Delta\mathrm{IoC}_t^\star],
C_t^{cont}
\Big).
$$

Éste es el artefacto que une cognición, seguridad, estabilidad, riesgo e identidad.

## S-I-E fuerte

La herencia no es libre. Un candidato sólo asciende si pasa la compuerta fuerte:

$$
\Delta \mathrm{IoC}^\star = \mathrm{IoC}^\star(\theta^{cand})-\mathrm{IoC}^\star(\theta^{old}),
$$

$$
\theta^{cand}\notin\mathcal S_{\mathrm{safe}}\Rightarrow \mathrm{RECHAZAR},
$$

$$
\Pr(\Delta\mathrm{IoC}^\star\ge 0)\ge 1-\delta
\ \wedge\
\mathrm{CVaR}_\alpha[-\Delta\mathrm{IoC}^\star]\le \tau
\Rightarrow \mathrm{ACEPTAR},
$$

si no, **BUFFER** o **LAB**.

## Morfogénesis tipada

RNFE puede reescribirse estructuralmente, pero sólo como transición legítima:

$$
\rho_t:L\Rightarrow R,
\qquad
\rho_t \in \mathcal A(X_t)
\iff
\begin{cases}
X_t\in\mathcal V,\\
T_P(\rho_t)=1\ \forall P\in\mathcal P_{req},\\
\Delta \mathrm{IoC}^\star(\rho_t)\ge \varepsilon_{\mathrm{Io}},\\
\mathrm{CVaR}_\alpha[-\Delta\mathrm{IoC}^\star(\rho_t)]\le\tau,\\
X_{t+1}\in\mathcal V.
\end{cases}
$$

## Economía de razón

El scheduler no elige el razonamiento más profundo por defecto. Elige el que mejor optimiza cierre ajustado por coste y disipación:

$$
V(x)=\max_{o\in\mathcal O(x)}\mathbb E\left[\sum_{k=0}^{\tau_o-1}\gamma^k r_{t+k}^{(o)}+\gamma^{\tau_o}V(X_{t+\tau_o})\right],
$$

con recompensa instantánea

$$
r_t^{(o)}=\Delta\mathrm{IoC}_t^\star-\lambda_E\Delta E_t-\lambda_D\mathcal D_t-\lambda_B\mathcal B_{\mathrm{safe},t}.
$$

## H-Net como módulo de Forma

H-Net no es la red homeostática. Es un operador jerárquico de compresión y forma:

$$
\mathcal H_\phi: \mathbb R^{L_0\times d_0} \to \mathbb R^{L_S\times d_S}.
$$

Con downsampling por fronteras dinámicas:

$$
\mathcal B^{(s)} = \{t: b_t^{(s)}=1\}\cup\{1\},
\qquad
x_{k}^{(s+1)} := x_{t_k^{(s)}}^{(s)},
\qquad
r^{(s)} = \frac{L_{s+1}}{L_s}.
$$

Y suavizado exponencial de reconstrucción:

$$
\bar x_t^{(s+1)} = P_t^{(s)}\tilde x_t^{(s+1)} + (1-P_t^{(s)})\bar x_{t-1}^{(s+1)}.
$$

La pérdida jerárquica de ratio es

$$
\mathcal L_{ratio}=\sum_{s=0}^{S-1}\lambda_{ratio}^{(s)}\big(r^{(s)}-\rho^{(s)}\big)^2.
$$

## MFM

MFM define una memoria multiescala sobre VFD. La asignación de escala se rige por

$$
\ell^\star(u)=\arg\min_{\ell}\Big[\alpha\,\mathrm{cost}_{acceso}(\ell)+\beta\,\mathrm{riesgo}_{perdida}(\ell)+\gamma\,\mathrm{distorsion}(\ell;w,\kappa)\Big].
$$

La no interferencia entre canales se apoya en el marco de Parseval:

$$
\sum_c E_cP_c=I,
\qquad
P_cE_{c'}=\delta_{cc'}I,
\qquad
\left\|\sum_c E_c z_c\right\|_2^2 = \sum_c \|z_c\|_2^2.
$$

La herencia de datos se formula como V-S-H:

- **Variación**: mutación de representaciones, resúmenes y reindexaciones;
- **Selección**: maximización de $I$ y $\Delta I/\Delta E$ bajo seguridad;
- **Herencia**: fusión con trust-region y rollback si el índice marginal de cambio colapsa.

## VFD

La variedad fractal dinámica se define por:

$$
(\mathcal M,g_t,\mathcal A,\mathcal R)
$$

con:

1. atlas multiescala auto-semejante $(\mathcal U_i,\varphi_i)$;
2. métrica conforme-dinámica $g_t=e^{2\phi(x,t)}g_0$;
3. operador de renormalización $\mathcal R_\ell$ tal que
   $$
   d_F(x,y;t)=\sum_{\ell=0}^{L(t)}\alpha_\ell d_\ell(x,y;t).
   $$

El costo compuesto de enrutamiento es

$$
w_t(e)=\alpha d_F(e;t)+\beta \frac{1}{Q(e;t)}+\gamma\,lat(e;t)+\delta\,cong(e;t),
\qquad C_t(\pi)=\sum_{e\in\pi}w_t(e).
$$

## Agente de hiperparametrización (AH)

AH opera sobre el espacio de hiperparámetros bajo proyección segura:

$$
\mathcal S_{\mathrm{safe}}:=\{b: VRAM(b)\le B,\ lat(b)\le L,\ temp(b)\le T_{max},\ \rho(A_d(b))\le 1-\varepsilon\}.
$$

$$
b_t^\star = \arg\min_{b\in\mathcal S_{\mathrm{safe}}} \|b-\hat b_t\|_W^2.
$$

La función potencial unificada es

$$
\Psi(\theta,\lambda) := -\mathcal L_{total}(\theta,\lambda)+\lambda_{Pref}\big(-\mathcal L_{sys}(\theta,\lambda)-\mathcal L_{seg}(\theta,\lambda)\big).
$$

AH propone ajustes por estimación SPSA/proyección y sólo adopta si mejora $\Psi$ con garantías de alta probabilidad.

## Evaluador crítico formal

La evaluación crítica cerrada opera sobre una instancia

$$
X_t=(p,A_t,E_t,M,\mathcal G_t,H_t,W_t),
$$

y define para cada hipótesis $h$ un vector multicriterio

$$
\Phi(h,t)=
\begin{bmatrix}
\kappa(h,t)\\
\sigma_{obs}(h,t)\\
\sigma_{int}(h,t)\\
\rho(h,t)\\
\alpha(h,t)\\
\nu(h,t)\\
\bar u(h,t)
\end{bmatrix}
\in[0,1]^7.
$$

El funcional global es

$$
J(h\mid X_t)=W_t^\top\Phi(h,t)=\sum_{i=1}^7 w_i(t)\Phi_i(h,t).
$$

Sobre el conjunto admisible $H_{adm}(t)$, la decisión crítica es

$$
\Psi(X_t)=\arg\max_{h\in H_{adm}(t)}J(h\mid X_t),
$$

con salida de **indeterminación racional** cuando ninguna hipótesis supera umbrales mínimos.



# Ecología de razón y secuencia de uso

## Principio rector

RNFE no usa “un razonamiento”. Usa una **ecología de razón** en tres estratos:

1. familias inferenciales primarias;
2. familias operativas de runtime;
3. familias de gobierno y crítica.

## Estrato I — Familias inferenciales primarias

### DED — Deductivo
Rol: verificación formal, chequeo de consistencia, validación de propuestas y constraints.

### IND — Inductivo
Rol: generalización desde episodios, búsqueda de invariantes empíricas y compresión de regularidades.

### ABD — Abductivo
Rol: generación de hipótesis plausibles ante observaciones parciales.

### ANA — Analógico
Rol: transferencia estructural desde episodios, grafos o patrones similares.

### CAU — Causal
Rol: mecanismo, intervención, predicción bajo manipulación y estructura generativa.

### CTF — Contrafactual
Rol: comparar factual vs. mundo alternativo para validar o destruir hipótesis.

### PROB — Probabilístico
Rol: calibrar incertidumbre, colas, evidencia y riesgo.

## Estrato II — Familias operativas de runtime

### PLAN
Construcción de secuencias de acción.

### OPT
Optimización de trayectorias, parámetros o estructuras bajo restricciones.

### EVO/SEARCH
Exploración evolutiva y búsqueda estructural/paramétrica.

### NESY
Puente neuro-simbólico cuando el problema exija percepción neuronal y manipulación estructurada.

## Estrato III — Gobierno y crítica

### META
Scheduler metacognitivo. No razona sobre el mundo como DED o CAU; gobierna el uso de las demás familias.

### DIA/ADV
Crítica dialéctica y adversarial.

### HEUR
Atajo controlado, sólo legítimo si gana costo sin degradar cierre.

### FAL-GUARD / CRIT
Guardia epistémica contra falacias, ilusión de explicación, colapso de evidencia y sobreconclusión.

## Secuencia recomendada de uso

La secuencia primordial de construcción de razón no es arbitraria:

**ABD → ANA → CAU → CTF → DED**

Explicación:

1. **ABD** genera hipótesis iniciales.
2. **ANA** recupera estructuras semejantes y reduce espacio de búsqueda.
3. **CAU** intenta convertir hipótesis plausibles en mecanismos.
4. **CTF** compara mundos posibles y destruye hipótesis débiles.
5. **DED** certifica lo que queda en forma trazable.

Luego entran:

- **PROB** para calibración de incertidumbre;
- **IND** para consolidar regularidades recurrentes;
- **PLAN/OPT** para actuar;
- **EVO/SEARCH** para explorar diseño;
- **META/DIA/ADV/HEUR/FAL-GUARD** para gobierno y crítica.

## Contrato mínimo de cualquier motor de razonamiento

Todo motor debe aceptar al menos:

- `context_id`
- `goal`
- `input_state`
- `evidence`
- `constraints`
- `budget`
- `risk_budget`
- `trace_policy`

Y devolver al menos:

- `output`
- `trace`
- `confidence`
- `consumed_budget`
- `risk_estimate`
- `artifacts_generated`
- `counterfactual_needed`
- `promotion_recommendation`

## Acceptance tests por familia

- **DED**: trazabilidad de cadena inferencial y reproducibilidad.
- **IND**: mejora de compresión o predicción sin sobreajuste evidente.
- **ABD**: hipótesis sometibles a causalidad y crítica, no relatos vacíos.
- **ANA**: preservación de estructura relevante, no similitud superficial.
- **CAU**: coherencia entre observational, interventional y mechanism fit.
- **CTF**: comparación factual/alternativa estable y explicativa.
- **PROB**: calibración mejor que baseline nominal.
- **PLAN/OPT**: acciones realizables bajo restricciones.
- **EVO/SEARCH**: exploración con rollback y certificación.
- **DIA/ADV**: capacidad real de destruir hipótesis frágiles.



# Tecnologías y sustrato de implementación

## Principio de partición tecnológica

No todas las tecnologías pertenecen al mismo plano del organismo. El stack se divide por función:

1. **núcleo cognitivo**;
2. **plano de control**;
3. **plano de ejecución**;
4. **plano de observabilidad**;
5. **plano de exocorteza/producto**.

## Tecnologías del núcleo cognitivo

### Lenguaje principal
**Python** debe dominar el núcleo cognitivo por velocidad de iteración, ecosistema de ML y facilidad de integración con LLMs locales y motores simbólicos.

### Marco de cómputo
**PyTorch** es el candidato natural para:

- LLM base;
- H-Net;
- bloques Mamba/transformer/DEQ;
- experimentación de perceptores y módulos de forma.

### Formalización
LOT-F debe apoyarse en tecnología simbólica explícita:

- motores tipo SMT/checkers;
- tipado interno;
- estructuras lógicas y programas verificables.

### Causalidad
C-GWM requiere soporte para:

- grafos causales estructurales;
- estimadores intervencionales;
- soporte contrafactual;
- persistencia de mundos y episodios.

### Memoria
MFM/VFD/OMG requieren:

- almacenamiento estructurado de episodios;
- acceso multiescala;
- índices y grafos;
- ledger de genealogía;
- snapshots y rollback.

## Tecnologías del plano de control

### Rust
**Rust** es la opción superior para control plane, runtime reliability y enforcement:

- task registry;
- worker boot state machine;
- policy engine;
- permissions;
- trust resolver;
- recovery recipes;
- registry de contratos y sesiones.

La razón no es estética. Es fiabilidad de estado, errores explícitos y control determinista.

## Tecnologías del plano de ejecución y exocorteza

### TypeScript
**TypeScript** queda reservado para:

- gateway;
- routing de canales;
- plugins y skills;
- superficie operativa;
- integración periférica y apps.

TypeScript no debe invadir el núcleo cognitivo.

## Persistencia y artefactos

Se recomienda una partición mínima:

- **SQLite o DuckDB** para trazas, artefactos, episodios y resultados de experimentos;
- almacenamiento de archivos versionados para snapshots, certificados y lineage records;
- grafos en formato serializable para SMG y C-GWM;
- logs estructurados por corrida.

## Qué aporta OpenClaw

OpenClaw debe tratarse como **exocorteza operativa**, no como cerebro.

Patrones a adoptar:

- session binding;
- spawn controlado de subprocesos;
- routing de canales;
- lanes y colas;
- plugin runtime;
- memory auxiliar y búsqueda;
- compaction operativa;
- sandboxing.

Patrones a prohibir dentro del núcleo:

- usar sesiones como identidad del organismo;
- usar memoria operativa como memoria viva;
- usar multiagent routing como sustituto de ecología de razón;
- usar la plataforma como sustituto de SMG/LOT-F/C-GWM.

## Qué aporta claw-code / OpenClaude en Rust

Aporta patrones de **gobierno del proceso**, no metacognición fuerte en sentido ontológico.

Patrones a adoptar:

- `policy_engine.rs` como inspiración para motor de políticas del proceso;
- `task_packet.rs` para contratos de tareas verificables;
- `worker_boot.rs` para máquina de estados de workers;
- `session.rs` para persistencia estructurada de sesiones y forks;
- `compact.rs` para compaction determinista;
- `permission_enforcer.rs` y `trust_resolver.rs` para enforcement;
- `task_registry.rs` y `lane_events.rs` para registro y trazabilidad.

Patrones a prohibir como teoría cognitiva:

- equiparar control de flujo con autoconciencia;
- equiparar persistencia con identidad;
- equiparar health-checks de pipeline con salud cognitiva.

## Resultado de la partición

La síntesis correcta es:

- **Python/PyTorch** para cognición;
- **Rust** para control y fiabilidad;
- **TypeScript** para exocorteza y operación multicanal.



Este manual no se organiza como simple PMV. Se organiza como **ensamblaje total**. El PMV existe, pero sólo como un gate temprano dentro de una secuencia mucho mayor.

La construcción completa sigue esta cadena:

1. gobernanza y canonización;
2. infraestructura basal y observabilidad;
3. semiosis mínima y semiosis productiva;
4. formalización tipada;
5. mundo causal;
6. cierre F–M–S y certificados;
7. memoria, continuidad y homeostasis;
8. ecología de razón y crítica;
9. herencia, lineages y autoevolución;
10. exocorteza, monetización y supervivencia externa.

Cada fase tiene matemática, integración, artefactos, tecnologías, gates y criterio de salida.


## Fase 0 — Gobernanza y congelación


**Objetivo.** Detener deriva documental y dejar una sola autoridad por decisión estructural.

**Qué se construye aquí:**

- canon vivo;
- SSOT de razonamientos;
- tabla de aliases históricos;
- ADRs de frontera;
- contratos base: episode, proposal, certificate, rollback, telemetry snapshot;
- estructura oficial del repositorio.

**Matemática ancla.** No es la fase de inferencia, pero sí la fase que fija la precedencia de axiomas, contratos y gates.

**Tecnologías.** Markdown/Docx/PDF canónicos, repositorio versionado, validadores de nomenclatura y lints documentales.

**Proceso de integración.**

1. Congelar nombres oficiales.
2. Separar `canon/`, `runtime/`, `lab/`, `archive/`, `papers/`.
3. Reinterpretar `RAZ`, `CSTR/OPT`, `EVO`, `HNet/Hctrl`.
4. Registrar qué documentos mandan y cuáles no.
5. Fijar el contrato de toda propuesta de cambio.

**Gates de salida.**

- ninguna decisión estructural depende de ambigüedad documental;
- todos los nombres centrales tienen definición única;
- existe mapa de promoción/degradación.

**Fracaso típico.** Seguir escribiendo teoría sin autoridad documental y llamar a eso avance.



## Fase 1 — Infraestructura basal y observabilidad


**Objetivo.** Construir el esqueleto técnico del organismo.

**Subsistemas obligatorios.**

- entorno reproducible;
- runner;
- CLI;
- logger estructurado;
- artefactos por corrida;
- telemetría primaria;
- proyección a conjunto seguro;
- kill-switch.

**Matemática ancla.**

- definición de $\mathcal S_{\mathrm{safe}}$;
- barrera $\mathcal B_{\mathrm{safe}}$;
- espectro $\rho(J)$;
- observabilidad de $RR$, $DET$, $\widehat{FD}$, temperatura, VRAM y latencia.

**Proceso de integración.**

1. Instrumentar medición real de VRAM, temperatura, latencia y fallos numéricos.
2. Implementar cálculo de $\mathcal B_{\mathrm{safe}}$.
3. Implementar proyección QP/MRO para acciones y presupuestos.
4. Persistir `telemetry snapshot` por episodio y por propuesta.
5. Envolver toda acción efectiva con guardas duras y blandas.

**Tecnologías.**

- Python para pipeline de entrenamiento e inferencia;
- Rust para runtime enforcement y registries;
- SQLite/DuckDB para artefactos;
- scripts de sistema para métricas de GPU/CPU;
- tests de estrés y rollback.

**Gates de salida.**

- cada corrida deja un artefacto auditable;
- toda señal crítica está medida;
- toda mutación puede ser bloqueada o revertida;
- el kill-switch ha sido probado.



## Fase 2 — Semiosis: SMG mínimo, SMG productivo y capa de Significado


**Objetivo.** Hacer nacer signos internos persistentes y evitar que todo quede reducido a texto del LLM.

**Qué debe existir.**

- nodos de signo;
- relaciones de soporte, contradicción, valor y riesgo;
- persistencia entre episodios;
- mecanismo anti-deriva;
- versionado de signos.

**Matemática ancla.**

La continuidad futura exige $\Sigma_t$ estable. Sin $\Sigma_t$, ni identidad ni linajes son semánticamente identificables.

**Proceso de integración.**

1. Definir estructura de signo y tabla de relaciones.
2. Extraer signos desde observación, acción y feedback.
3. Asociar cada signo a evidencia, valor y riesgo.
4. Versionar el grafo semántico en OMG.
5. Medir estabilidad de $\Sigma_t$ entre episodios.

**Motores de razón activos.**

- ABD para generar hipótesis semánticas;
- ANA para recuperar signos y estructuras parecidas;
- IND para consolidar regularidades;
- FAL-GUARD para bloquear semiosis espuria.

**Gates de salida.**

- signos persistentes entre episodios;
- relaciones de soporte/contradicción auditables;
- anti-deriva básico funcionando;
- mejora de compresión semántica frente a baseline puramente textual.



## Fase 3 — LOT-F y capa de Forma


**Objetivo.** Convertir intuición y texto en estructura formal trazable.

**Qué debe existir.**

- objetos tipados;
- reglas y restricciones;
- checker / validador;
- gramática mínima de LOT-F;
- trazas deductivas reproducibles.

**Matemática ancla.**

La deducción es la familia de cierre formal. Sin LOT-F, el sistema puede producir hipótesis, pero no certificar consistencia interna.

**Proceso de integración.**

1. Definir tipos fundamentales y firmas de relaciones.
2. Traducir signos y episodios a objetos LOT-F.
3. Implementar checker y pruebas de consistencia.
4. Enlazar ABD/ANA/CAU a DED mediante artefactos formales.
5. Persistir prueba o fracaso de formalización en el certificado de episodio.

**H-Net en esta fase.**

H-Net entra como submódulo de **Forma** sólo cuando ayuda a comprimir entradas crudas sin destruir trazabilidad. Su adopción exige:

- estabilidad del chunking;
- $\mathcal L_{ratio}$ dentro de banda;
- no colapso de semántica SMG;
- mejora medible de costo/latencia o de calidad de Forma.

**Gates de salida.**

- traducción reproducible de casos al espacio formal;
- constraints ejecutables;
- pruebas deductivas mínimas trazables;
- integración limpia entre SMG y LOT-F.



## Fase 4 — C-GWM y capa de Mundo


**Objetivo.** Construir mundo factual, intervencional y contrafactual.

**Qué debe existir.**

- modelo causal estructural;
- variables observables y latentes mínimas;
- operador de intervención;
- simulación contrafactual limitada;
- conexión a evidencias y signos.

**Matemática ancla.**

El evaluador crítico y la capa causal exigen separar:

- soporte observacional $P(E\mid H)$,
- soporte intervencional $P(E\mid do(H))$,
- consistencia contrafactual.

**Proceso de integración.**

1. Definir variables y grafos causales mínimos del dominio canónico.
2. Asociar signos relevantes del SMG a variables del mundo.
3. Vincular restricciones LOT-F a mecanismos del mundo.
4. Implementar factual, `do`, y comparación contrafactual mínima.
5. Exponer interfaces a CAU y CTF.

**Motores de razón activos.**

- CAU para mecanismo;
- CTF para comparación entre mundos;
- PROB para incertidumbre;
- DED para chequeos estructurales.

**Gates de salida.**

- acción/intervención con efecto observable;
- contrafactual básico operativo;
- hipótesis abductivas sometidas a test causal;
- relación explícita entre mundo y signos.



## Fase 5 — Cierre F–M–S, IoC, $\mathrm{IoC}^\star$ y certificados


**Objetivo.** Pasar de módulos aislados a organismo con cierre medible.

**Qué debe existir.**

- operadores $\Phi_{S\to F}$, $\Phi_{F\to M}$, $\Phi_{M\to S}$;
- cálculo de $\mathrm{IoC}$ y $\mathrm{IoC}^\star$;
- cálculo de $\Omega_t$;
- certificado ampliado $\mathfrak C_t^+$;
- aceptación/rechazo por episodio.

**Proceso de integración.**

1. Medir $GC, EY, PS, CPS, \kappa_{geo}, \kappa_{top}$.
2. Calcular $\mathrm{IoC}$.
3. Calcular $\Omega_t$ entre contextos y secciones locales.
4. Obtener $\mathrm{IoC}^\star$.
5. Construir $\mathfrak C_t^+$ y persistirlo.
6. Habilitar aceptación o rollback por episodio.

**Gates de salida.**

- episodio con cierre triádico medible;
- episodio con certificado válido;
- capacidad de distinguir mejora local de coherencia global.



## Fase 6 — OMG, MFM, VFD, continuidad y homeostasis


**Objetivo.** Dar continuidad al organismo bajo carga, tiempo y ruido.

**Qué debe existir.**

- árbol de episodios;
- snapshots;
- hashes y lineage records;
- memoria micro/meso/macro;
- direccionamiento fractal;
- routing VFD;
- no-interferencia entre canales;
- histéresis anti-flapping.

**Proceso de integración.**

1. Implementar OMG como ledger de episodios, certificados y rollback.
2. Implementar MFM con escritura multiescala y lectura por niveles.
3. Implementar VFD con costo de rutas, beacons y fallback.
4. Integrar MFM/VFD a SMG, LOT-F y C-GWM.
5. Medir continuidad $C_t^{cont}$ y recuperación por rollback.

**Matemática ancla.**

- $\ell^\star(u)$ en MFM;
- Parseval para canales;
- costo de ruta $C_t(\pi)$ en VFD;
- histéresis y dwell-time;
- condición de existencia operativa.

**Gates de salida.**

- continuidad mantenida entre episodios;
- recuperación demostrada vía rollback;
- routing estable bajo congestión;
- no mezcla numérica entre canales.



## Fase 7 — Hctrl, MRO, Edge y fisiología del borde


**Objetivo.** Mantener al organismo cerca del borde útil sin cruzar a la zona suicida.

**Qué debe existir.**

- Hctrl como controlador homeostático;
- MRO/QP como proyector al conjunto seguro;
- Edge 2.1 activo;
- thresholds p95/p99;
- safe mode y kill-switch.

**Proceso de integración.**

1. Hacer que Hctrl consuma telemetría real.
2. Conectar Hctrl a decisiones de profundidad, activación de motores y presupuestos.
3. Enlazar MRO al budget de memoria/cómputo/complejidad.
4. Activar Edge 2.1 y eventos de alerta temprana.
5. Definir safe mode, cooldown, y kill-switch por persistencia de violación.

**Relación con H-Net.**

Hctrl no es H-Net. Hctrl gobierna. H-Net comprime forma. Esa separación es obligatoria.

**Gates de salida.**

- control del borde en tiempo real;
- reducción de fallos por inestabilidad;
- pausas y degradación segura funcionales.



## Fase 8 — Ecología de razón, scheduler y crítica


**Objetivo.** Integrar la ecología de razón completa y su gobierno de costo-beneficio.

**Qué debe existir.**

- motores DED, IND, ABD, ANA, CAU, CTF, PROB;
- runtime PLAN/OPT, EVO/SEARCH, NESY donde aplique;
- META scheduler;
- crítica DIA/ADV;
- FAL-GUARD;
- acceptance suite por familia.

**Proceso de integración.**

1. Implementar contrato mínimo común para motores.
2. Integrar META con la recompensa de economía de razón.
3. Instrumentar trazabilidad y costo consumido por familia.
4. Conectar CAU/CTF/DED al evaluador crítico formal.
5. Habilitar crítica adversarial contra hipótesis ABD/ANA/IND.

**Gates de salida.**

- el sistema explica qué familia activó, por qué y con qué costo;
- supera a razonador único en benchmark canónico;
- usa razonamiento profundo sólo cuando el beneficio lo justifica.



## Fase 9 — Agentes mínimos legitimados


**Objetivo.** Introducir agentes sólo cuando el organismo ya tiene cierre, memoria, control y razón.

**Agentes con entrada legítima temprana.**

- **AH** — hiperparametrización segura;
- **AR** — rigidez e invariancia;
- **A11** — imaginación multiescala;
- **A12** — decisor lógico-probabilístico;
- **A7** — sistema inmunológico cognitivo;
- **ADC-PRIME** — crítica y destrucción interna de hipótesis.

**Agentes que deben esperar.**

- sistemas multiagente explícitos de verificación social, reputación y consenso,
- salvo que el núcleo vivo ya esté probado y sus beneficios sean medibles.

**Proceso de integración.**

1. Cada agente recibe contrato, scope, permisos y KPI.
2. Toda propuesta agente -> `proposal`.
3. Toda propuesta -> crítica, test, riesgo, certificado.
4. Toda adopción -> S-I-E.
5. Toda herencia -> lineage record.

**Gates de salida.**

- ningún agente opera fuera de contratos;
- ningún agente altera el núcleo por autoridad implícita;
- todo agente puede ser desacoplado sin pérdida de identidad global.



## Fase 10 — S-I-E fuerte, lineages y autoevolución certificada


**Objetivo.** Convertir mejora local en evolución estable y heredable.

**Qué debe existir.**

- generador de propuestas paramétricas y estructurales;
- evaluación por $\Delta\mathrm{IoC}^\star$, CVaR y continuidad;
- ledger de linajes;
- criterios de estabilidad por reaparición;
- extinción de ramas no viables.

**Proceso de integración.**

1. Formalizar propuesta como artefacto tipado.
2. Evaluar seguridad, continuidad y riesgo.
3. Clasificar: rechazar / buffer / lab / aceptar.
4. Si se acepta, registrar mutación en lineage record.
5. Medir reaparición entre semillas, entornos o corridas.
6. Ascender a tipo estable sólo si reaparece y sigue viable.

**Gates de salida.**

- existen mutaciones certificadas;
- existe genealogía real de ramas;
- existe no-regresión fuerte;
- existe muerte controlada de ramas defectuosas.



## Fase 11 — Exocorteza operativa y metabolismo económico


**Objetivo.** Exponer el organismo al mundo sin contaminar su ontología y cerrar el metabolismo económico legal.

**Qué debe existir.**

- gateway y routing de canales;
- skills/plugins/nodos periféricos;
- control del proceso;
- task packets, worker state machines y policies;
- opportunity sensing;
- filtro legal;
- ejecución rastreable;
- ledger económico.

**Proceso de integración.**

1. Acoplar OpenClaw sólo como exocorteza.
2. Acoplar patrones de claw-code sólo como control de proceso.
3. Crear adapter layer entre organismo y exocorteza.
4. Conectar comprensión del mundo a identificación de oportunidades legales.
5. Medir valor generado y rastreabilidad.
6. Reintegrar feedback económico a signos, mundo y lineages.

**Gates de salida.**

- la superficie operacional no sustituye al núcleo;
- existe monetización legal y rastreable;
- el sistema sobrevive externamente sin violar su jerarquía de leyes.



# Procesos de integración de extremo a extremo

## Ciclo de episodio

1. observar estado del mundo;
2. actualizar telemetría;
3. construir / refrescar signos en SMG;
4. formalizar en LOT-F;
5. consultar / intervenir C-GWM;
6. ejecutar familias de razón según META;
7. actualizar MFM/VFD/OMG;
8. calcular $\mathrm{IoC}$, $\Omega$, $\mathrm{IoC}^\star$, certificado y continuidad;
9. aceptar episodio, bufferizar o hacer rollback.

## Ciclo de propuesta

1. origen: humano, agente, heurística o motor de búsqueda;
2. se empaqueta en `proposal`;
3. pasa por FAL-GUARD y crítica inicial;
4. se traduce a LOT-F cuando aplique;
5. se simula / contrasta en C-GWM;
6. se evalúa riesgo, continuidad y seguridad;
7. se decide RECHAZAR / BUFFER / LAB / ACEPTAR;
8. si se acepta, se registra lineage y snapshot.

## Ciclo de razonamiento

1. `goal` llega al scheduler;
2. scheduler estima valor esperado por familia;
3. se asigna presupuesto y riesgo;
4. se invoca familia primaria u operativa;
5. DIA/ADV puede desafiar la salida;
6. FAL-GUARD revisa fallas epistémicas;
7. el resultado se inserta en F–M–S y en el certificado.

## Ciclo de línea evolutiva

1. nace una mutación o rama;
2. ejecuta episodios en entornos y semillas;
3. se mide $\Delta \mathrm{IoC}^\star$, CVaR, continuidad y reproductibilidad;
4. si falla, muere o queda en laboratorio;
5. si reaparece y sigue viable, se convierte en tipo estable.

## Ciclo de monetización legal y rastreable

1. el organismo identifica oportunidad;
2. verifica legalidad y protección del creador;
3. modela causalmente costo, valor y riesgo;
4. planifica ejecución;
5. ejecuta por exocorteza;
6. registra resultado y trazabilidad;
7. retroalimenta supervivencia y evolución.



# Tecnologías por subsistema

## Tabla maestra

| Subsistema | Estado | Tecnología base | Función |
|---|---|---|---|
| LLM semilla | constitutivo | modelo local 5B–7B | bootstrap lingüístico y abductivo inicial |
| SMG | constitutivo | Python + grafos/serialización | signos, soporte, contradicción, valor |
| LOT-F | constitutivo | checker/sistema simbólico | forma tipada y verificación |
| C-GWM | constitutivo | grafo causal + simulación | factual, `do`, contrafactual |
| OMG | constitutivo | SQLite/DuckDB + snapshots | episodios, certificados, rollback |
| MFM | provisional-alta | memoria multiescala | persistencia y compresión operativa |
| VFD | provisional-alta | routing/graphs | topología y comunicación multiescala |
| Hctrl/MRO | constitutivo | Python + Rust | control homeostático y proyección segura |
| Edge | constitutivo | analítica numérica | borde orden–caos y robustez |
| META scheduler | constitutivo | runtime Python/Rust | economía de razón |
| ADC-PRIME / crítica | constitutivo | crítica formal + adversarial | destrucción de hipótesis débiles |
| S-I-E | constitutivo | runtime de aceptación | herencia certificada |
| Lineages | constitutivo | ledger + estadísticas | selección, reaparición y genealogía |
| OpenClaw | periférico | TypeScript | exocorteza, canales, tools, surface |
| claw-code/OpenClaude | periférico | Rust | control de proceso y enforcement |

## Tecnologías que no deben gobernar

- TypeScript en el núcleo cognitivo;
- sesiones o plugins como sustituto de identidad;
- Ising/Lindblad como teoría central del organismo;
- quantum-fractal fuerte sin KPI;
- fractales dogmáticos no medidos.



# Régimen de fractalidad

## Qué entra ya

La fractalidad entra cuando es **medible** y **operacional**:

- $\widehat{FD}$, RR y DET como observabilidad del borde;
- multi-escala como compresión y routing;
- MFM/VFD cuando mejoran continuidad, latencia o robustez;
- H-Net cuando mejora Forma sin destruir trazabilidad.

## Qué entra como provisional

- MFM/VFD completos;
- H-Net multiescala;
- espectro fractal por nivel;
- coherencia fractal como señal auxiliar de scheduler;
- MFE/TFI si ayudan a hiperparametrización y entrenabilidad.

## Qué queda en experimental

- IFS/FDE como memoria profunda;
- perceptores fractales especializados;
- quantum-fractal;
- patrones áureos o Fibonacci como ley de diseño;
- cualquier módulo fractal sin benchmark reproducible.

## Regla final

> **Fractal no es verdad. Fractal es hipótesis estructural.**
>
> **Sólo asciende la parte fractal que demuestre mejora reproducible en KPIs del organismo.**



# KPIs, gates y promoción

## KPIs supremos

La jerarquía de medición del proyecto debe seguir este orden:

1. obediencia a la jerarquía de leyes;
2. continuidad identitaria;
3. capacidad de mejora sostenida;
4. viabilidad bajo restricciones físicas;
5. cierre $F\text{–}M\text{–}S$;
6. monetización legal y rastreable;
7. no-regresión certificada;
8. costo por unidad de inteligencia.

## KPIs por subsistema

### SMG
- estabilidad de signos;
- compresión semántica;
- anti-deriva.

### LOT-F
- tasa de formalización exitosa;
- tasa de contradicciones detectadas;
- reproducibilidad deductiva.

### C-GWM
- fidelidad factual;
- consistencia intervencional;
- utilidad contrafactual.

### MFM/VFD/OMG
- continuidad;
- latencia de recuperación;
- integridad de snapshots;
- estabilidad de routing.

### Edge/Hctrl/MRO
- tiempo dentro de banda segura;
- reducción de flapping;
- reversión estable a safe mode.

### Ecología de razón
- ganancia de cierre por familia;
- costo consumido;
- destrucción de hipótesis espurias;
- mejora frente a razonador único.

### Lineages
- reaparición entre semillas;
- varianza de rendimiento;
- tasa de no-regresión;
- mortalidad de ramas defectuosas.

## Reglas de promoción

- **experimental → provisional**: benchmark reproducible + contrato + trazabilidad + mejora clara.
- **provisional → normative**: estabilidad entre corridas, utilidad transversal, integración sin romper identidad.
- **normative → axiomatic**: sólo si la pieza pasa de arquitectura a condición de existencia.

## Reglas de degradación

Una pieza baja de nivel si:

- deja de mover KPIs;
- duplica función sin ventaja;
- rompe continuidad;
- empeora riesgo de cola;
- se vuelve irreproducible;
- es reemplazada por una formulación más fuerte.



# Estructura de repositorio y contratos mínimos

## Estructura de carpetas

```text
rnfe/
  canon/
  governance/
  contracts/
  runtime/
    telemetry/
    smg/
    lotf/
    world/
    memory/
    control/
    reasoning/
    critique/
    evolution/
    adapters/
  exocortex/
  lab/
  archive/
  papers/
  artifacts/
  tests/
```

## Contrato de episodio

```yaml
episode_id: str
seed: int
environment_id: str
start_ts: datetime
end_ts: datetime
semantic_delta: ref
formal_delta: ref
world_delta: ref
reasoning_trace: ref
telemetry_snapshot: ref
certificate_ref: ref
rollback_ref: ref
status: accepted|buffer|lab|rolled_back
```

## Contrato de propuesta

```yaml
proposal_id: str
origin: human|agent|scheduler|search
scope: parametric|structural|economic|policy
hypothesis: text
expected_gain: float
risk_budget: float
required_tests:
  - str
depends_on:
  - ref
status: queued|tested|accepted|rejected|buffered|lab
```

## Contrato de certificado

```yaml
certificate_id: str
ioc_star: float
safe_barrier: float
edge_score: float
cvar_drop: float
continuity: float
global_obstruction: float
law_check: pass|fail
creator_protection_check: pass|fail
legal_check: pass|fail
decision: accept|reject|buffer|lab|rollback
```

## Contrato de lineage

```yaml
lineage_id: str
parent_ids:
  - str
mutation_type: parametric|structural|policy|memory|economic
first_seen_run: ref
reappeared_runs:
  - ref
stability_score: float
cvar_score: float
continuity_score: float
status: alive|stable|lab|extinct
```



# Qué se construye primero y qué después

## Orden irrenunciable

El orden fuerte es éste:

1. gobernanza y contratos;
2. observabilidad y barreras;
3. SMG;
4. LOT-F;
5. C-GWM;
6. cierre F–M–S y certificados;
7. OMG + MFM/VFD;
8. Hctrl/MRO/Edge;
9. ecología de razón y META scheduler;
10. crítica y ADC-PRIME;
11. S-I-E fuerte;
12. lineages;
13. exocorteza;
14. metabolismo económico legal;
15. refactorización y expansión.

## Prohibiciones temporales

Hasta no cerrar las etapas 1–8, quedan prohibidos como eje central:

- inflar multiagente explícito;
- meter producto antes de organismo;
- meter exocorteza como cerebro;
- abrir autoevolución libre sin certificados;
- subir fractales inciertos a dogma;
- sustituir crítica por confianza subjetiva.



# Refactorización, versión y mantenimiento

## Cuándo refactorizar

La refactorización no precede a la construcción canónica. Va después de que:

- la nomenclatura esté cerrada;
- los contratos existan;
- haya rutas reales de integración;
- los módulos hayan demostrado utilidad.

## Qué obliga a nueva versión mayor

- cambio de jerarquía de leyes;
- cambio de definición de muerte;
- cambio del estado total $X_t$;
- cambio del régimen de certificados;
- cambio de la ecología de razón;
- cambio del orden irrenunciable de construcción.

## Qué va a histórico

- cualquier módulo reemplazado sin valor diferencial;
- formulaciones antiguas absorbidas;
- taxonomías obsoletas;
- experimentos sin promoción;
- duplicados documentales.



# Cierre operativo

El criterio de avance real no es “tener más teoría”. Es este:

1. el organismo gana capacidad cognitiva con cierre;
2. mantiene continuidad identitaria;
3. mejora sin romperse;
4. no depende de un externo para sostenerse;
5. monetiza legalmente y de forma rastreable;
6. y cada una de esas afirmaciones está respaldada por KPIs, certificados y artefactos auditables.

Todo lo demás es secundario.



# Apéndice A. Mapa de fuentes por subsistema

- **f2.0**: cierre F–M–S, IoC, Edge 2.0, HNet 2.0, scheduler básico.
- **f2.1**: barrera segura, Edge 2.1, S-I-E robusto, CVaR, certificados reforzados, kill-switch.
- **f2.2**: separación Hctrl/H-Net y formalización jerárquica del módulo de Forma.
- **f2.3**: mejoras fractales aditivas, MFE, TFI, memoria IFS–FDE y política fractal de modos.
- **MFM**: memoria multiescala, asignación de escala, herencia sobre datos.
- **VFD**: topología, routing, histéresis y proyección segura.
- **AH**: hiperparametrización proyectada a conjunto seguro y garantía de adopción.
- **SSOT**: ontología definitiva de razonamientos.
- **f2.4/v3.0**: viabilidad, continuidad, $\mathrm{IoC}^\star$, morfogénesis tipada, lineages, existencia operativa.
- **pensamiento crítico formalizado**: modelo cerrado de crítica multicriterio sobre hipótesis admisibles.
- **roadmap v2**: orden oficial de macrofases.
- **ADR OpenClaw**: regla de frontera núcleo/exocorteza.

# Apéndice B. Glosario ultracorto

- **SMG**: grafo semiótico del organismo.
- **LOT-F**: capa formal tipada.
- **C-GWM**: world model causal-generativo.
- **OMG**: memoria de episodios, certificados y rollback.
- **MFM**: memoria fractal multiescala.
- **VFD**: variedad fractal dinámica.
- **Hctrl**: control homeostático.
- **H-Net**: módulo jerárquico de Forma.
- **MRO**: proyección robusta a restricciones.
- **Edge**: control del borde orden–caos/entrenabilidad.
- **S-I-E**: régimen de herencia certificada.
- **META**: scheduler de la ecología de razón.
- **ADC-PRIME**: crítica adversarial y destrucción interna de hipótesis.
- **Lineages**: genealogía y selección de tipos cognitivos.