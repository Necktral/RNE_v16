---
date: 2026-07-15
description: "Operación neural nocturna sin operador, con recovery y autoridad SHADOW"
tags:
  - rnfe
  - neural
  - operations
  - active
---

# RNFE v16 — Supervisor nocturno desatendido

El humano autorizó que las campañas neurales nocturnas se preparen y gestionen sin
supervisión humana activa. La implementación vive en
`scripts/supervise_integral_neural_campaign.py` y se inicia mediante Windows Task
Scheduler hacia el filesystem ext4 de `Ubuntu-24.04`.

## Decisión

- La autorización permanente cubre diagnóstico nocturno y staging SHADOW sujeto a
  gates deterministas.
- No cubre entrenamiento, promoción operativa, activación ni mutaciones soberanas.
- PostgreSQL sigue siendo evidencia oficial; SQLite permanece como contingencia
  separada.
- Todo fallo de gates produce cuarentena; todo bloque interrumpido se repite desde
  cero.

Ver [[North Star]] y la documentación viva
`docs/analysis/25_UNATTENDED_NEURAL_SUPERVISOR.md`.
