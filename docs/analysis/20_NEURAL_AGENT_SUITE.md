# Auditoría de realidad — capa de agentes neurales v1

Fecha: 2026-07-15 · Base: `integration/neural-tech-connectomic-v1@f88a314`

## Arquitectura observada

El camino vivo es `ScenarioEpisodeRunner → SymbioticNeuralCoordinator → N0 →
adaptadores N1–N6 → ConsumerReceipt → conectoma/certificación/vida`. Los modelos
entrenados siguen en laboratorio o `SHADOW`; la autoridad permanece en scheduler,
familias canónicas, certificación, MSRC y almacenamiento.

La capa nueva no crea un segundo runtime. Consume trazas y recibos ya existentes y
produce un bloque aditivo de cinco reportes:

Todos los reportes declaran `experimental=true`. La misión que los gobierna es
construir un ser cibernético con comprensión y mejora cognitiva combinando familias
de razonamiento, entrenamiento neuronal y guía de un maestro local 7B.

La lectura cruzada del canon, la carta de crecimiento, los planos de integración y
el código vivo obliga a no colapsar esos tres pilares:

- las familias forman una ecología de razón con motores primarios, operativos y
  críticos gobernados por META;
- el entrenamiento neural permanece offline/lab hasta probar generalización,
  calibración y beneficio cognitivo retenido;
- el 7B local tiene dos contratos diferentes: razonador caro `tier_3_external` y
  maestro post-herida en `runtime/organism/teacher.py`. Producir una lección no prueba
  que la lección cambió conducta ni redujo daño.

| Agente | Entrada | Salida | Autoridad |
|---|---|---|---|
| Orquestación | cuatro reportes especializados | estado y hash del ciclo | none |
| Conectómica | topología, actividad, trazas, recibos | aislamiento y plasticidad observada | none |
| Comunicación latente | confianza, incertidumbre, veredictos informativos | ganancia propuesta `[0.75,1.25]` | none |
| Adversarial | identidad, hashes, recibos, autoridad | hallazgos y cuarentena | none |
| Simbiosis/sinergia | consumo real y reportes previos | cobertura e integración | evidence_only |

### Extensión 1 — metacognición epistémica

`MetacognitiveEpistemicAgent` ya está conectado en
`SymbioticNeuralCoordinator.prepare_certification`. Consume la secuencia y validación
de META, `prob_point`/`prob_lcb`, el acuerdo CAU↔CTF, órganos y recibos. Emite un
reporte especializado separado del ciclo base para no romper el contrato de cinco
agentes. Si falta PROB se abstiene; si CAU y CTF discrepan propone crítica o consulta;
si hay medición consistente difiere todo juicio de ganancia hasta observar outcome.
Nunca autoriza al maestro ni modifica la selección.

### Extensión 2 — memoria y consolidación

`MemoryConsolidationAgent` ya consume la recuperación real del episodio y enlaza sus
referencias con N3/N5. Detecta memoria sin procedencia, duplicados y cruces de
escenario no atestados. Produce una propuesta para el gate MFM existente, pero no
escribe, consolida ni promociona por sí mismo. La consolidación final sigue diferida
hasta observar certificado y outcome.

### Extensión 3 — pedagogía del maestro 7B

`PedagogicalTeacherAgent` corre después del outcome y compara la severidad actual con
`from_severity` de la lección aplicada. Distingue maestro inactivo, lección no
aplicable, no aplicada, aplicada con mejora singular, aplicada sin mejora y outcome
no medido. La preferencia generada por el 7B dejó de persistirse como episodio
certificado de severidad cero; ahora es propuesta no probada. Una observación positiva
no prueba causalidad ni autoriza currículo.

### Extensiones 4–11 — integración del organismo

- `model_data_immune` audita pares manifest/artefacto, recibos y durabilidad; sólo
  propone cuarentena.
- `curriculum_learning` compara ensayos pareados `no_teacher`, `local_7b` y
  `codex_frontier` por situación/semilla. Mide reducción de daño y beneficio por
  segundo cuando hay latencia, pero no clasifica eficiencia sin controles.
- `sensorimotor_world_model` exige observación, atestación causal y N4 ligado a la
  acción comprometida; el error de predicción queda pendiente del outcome.
- `interoceptive_homeostatic` une viabilidad, distancia al borde, rollback, MSRC,
  presiones físicas y durabilidad. Distingue valores medidos de defaults y conserva
  autoridad en MSRC y el kernel de viabilidad.
- `metabolic_budget` observa presión física y MSRC; N0 conserva el presupuesto.
- `development_lineage` exige linaje y token de rollback para propuestas N6.
- `horizontal_creativity` mide amplitud entre familias y propone alternativas
  shadow; no requiere Mamba2 ni confunde novedad con calidad.
- `social_exocortex` admite evidencia externa sólo con `source_id` y
  `content_hash`; la fuente externa nunca obtiene autoridad por identidad.

## Cambios de integridad

- N4 emite recibo sólo después de enlazarse a la intervención comprometida.
- La plasticidad deja de contar clases neutrales como positivas.
- Cada ciclo exige cinco roles únicos, identidad común y hashes canónicos.
- La ausencia de medición provoca abstención, no ganancia neutral fabricada.

## Riesgos y límites

- La modulación latente es una política de referencia, no una representación
  aprendida ni un cambio de pesos vivo. No usa Mamba2. Registra artefactos separados
  de medición, clasificación, análisis y deliberación; el entrenamiento es una etapa
  posterior. `[0.75,1.25]` es una envolvente de seguridad, no un setpoint aprendido.
- El reporte adversarial complementa, pero no sustituye, certificación y sandbox.
- El bloque de agentes adjunto a certificación se calcula antes del certificado y
  reward finales. Por eso v1 mide integridad/conectividad del ciclo, no ganancia
  cognitiva longitudinal. El agente pedagógico/de aprendizaje futuro debe consumir
  la traza finalizada y comparar episodios o vidas.
- N4 aún no descubre topología causal y conserva una ejecución preliminar shadow
  para diagnóstico; sólo la ejecución final queda en la traza soberana.
- La plasticidad sigue siendo propuesta no aplicable; no existe rollback de una
  mutación porque no existe mutación.
- No se afirma ganancia cognitiva hasta ejecutar campañas held-out multisemilla.

## Agentes de integración — estado v1

1. **Agente metacognitivo/epistémico acoplado a META — IMPLEMENTADO v1**:
   cobertura, contradicción y certeza comprometida pre-outcome. Falta la extensión
   longitudinal que calcule ganancia y calibración contra resultados.
2. **Agente pedagógico del maestro 7B — IMPLEMENTADO v1**: cierra
   herida→lección→sesgo→resultado y evita certificar preferencias no probadas. Falta
   estimación multiepisodio de reincidencia y efecto causal.
3. **Agente de aprendizaje y currículo neural — IMPLEMENTADO v1**: protocolo
   comparativo 7B/Codex/control, procedencia y eficiencia; falta ejecutar la campaña
   multisemilla y habilitar un dataset sólo después del gate.
4. **Agente de memoria y consolidación — IMPLEMENTADO v1**: enlaza N3/N5 con la
   recuperación MFM, audita duplicación/procedencia/escenario y difiere promoción al
   gate existente. Faltan olvido y consolidación post-certificado.
5. **Agente sensoriomotor/world-model — IMPLEMENTADO v1**: cierra
   observación→predicción→acción→outcome y mide fidelidad contrafactual antes de
   adelantar autoridad a modelos shadow.
6. **Agente inmunológico de datos/modelos — IMPLEMENTADO v1**: integridad de
   artefactos/manifests, receipts y persistencia. Deriva estadística queda pendiente.
7. **Agente interoceptivo/homeostático — IMPLEMENTADO v1**: integra viabilidad y
   señales físicas con estados explícitos estable, parcial, estrés y emergencia;
   nunca convierte defaults en medición ni autoriza actuación o rollback.
8. **Agente metabólico/presupuesto — IMPLEMENTADO v1**: traduce MSRC a propuestas sin
   convertir energía en objetivo cognitivo.
9. **Agente de desarrollo/linaje — IMPLEMENTADO v1**: audita herencia, rollback y
   compatibilidad entre versiones del organismo.
10. **Agente de creatividad horizontal — IMPLEMENTADO v1**: mide rutas laterales entre familias,
   busca analogías y rutas no dominantes, y devuelve propuestas contrastables. Un
   backend Mamba2 podría evaluarse en `SHADOW`, pero es una alternativa experimental,
   no la comunicación rápida canónica ni una fuente de autoridad.
11. **Agente social/exocórtex — IMPLEMENTADO v1**: evidencia externa bajo identidad,
    recibos, sandbox y política de confianza explícita.

El orden ejecutado priorizó contratos y procedencia antes de cualquier entrenamiento:
epistemología → memoria → pedagogía → inmunidad → currículo → cierre sensoriomotor →
interocepción → presupuesto → linaje → creatividad → exocórtex.
