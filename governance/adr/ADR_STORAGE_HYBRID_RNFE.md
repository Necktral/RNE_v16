---
title: ADR_STORAGE_HYBRID_RNFE
status: normative
version: 1.0.0
date: 2026-04-16
owner: Codex
type: architecture-decision-record
subject: Persistencia híbrida RNFE (PostgreSQL principal + SQLite fallback + artifact plane filesystem)
---

# ADR — Persistencia Híbrida RNFE

## 1. Contexto

RNFE requiere trazabilidad continua, rollback auditable y operación degradada sin dependencia total de red/servicios externos.
El runtime ya dispone de un ledger SQLite compatible legacy y una capa `runtime/storage` con backend PostgreSQL.

## 2. Decisión

Se adopta arquitectura híbrida de persistencia:

1. `PostgreSQL` como store transaccional principal.
2. `SQLite` como fallback/local journal y modo degradado.
3. Artifact plane en filesystem con indexación en DB.

## 3. Reglas normativas

1. No se almacenan blobs pesados en PostgreSQL ni SQLite.
2. Todo artefacto persistido se representa por metadatos mínimos:
   - `run_id`, `kind`, `path`, `sha256`, `size_bytes`, `mime_type`.
3. `SQLite` mantiene compatibilidad legacy del ledger (`eve_type -> event_type`).
4. El runtime debe operar en modo local sin PostgreSQL cuando `RNFE_STORAGE_MODE=sqlite`.
5. En modo `hybrid`, el write-path es dual-write con lectura preferente desde PostgreSQL.

## 4. Configuración operativa

- `RNFE_STORAGE_MODE=sqlite|postgres|hybrid`
- `RNFE_POSTGRES_DSN=postgresql://...`
- `AEON_EVENT_DB=<ruta sqlite>`
- `RNFE_ARTIFACT_ROOT=/mnt/d/rnfe_artifacts` (default laboratorio WSL2)

## 5. Consecuencias

### Ventajas

- Continuidad operativa ante caída de PostgreSQL.
- Trazabilidad consistente para episodios, telemetry y traces de razonamiento.
- Escalabilidad de metadatos sin degradar DB con blobs.

### Costos

- Complejidad adicional por reconciliación DB↔filesystem.
- Necesidad de disciplina explícita de backup y validación.

## 6. Criterios de verificación

1. `pytest` basal pasa sin dependencias optativas.
2. Integración PostgreSQL pasa en modo opt-in.
3. Migración SQLite→PostgreSQL es idempotente por `(legacy_db_path, legacy_event_id)`.
4. Artifact plane preserva integridad por hash y tamaño.
