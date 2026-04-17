# POLICY — Backup y Reconciliación DB ↔ Artifact Store

## Objetivo

Garantizar continuidad, trazabilidad y recuperabilidad entre metadatos persistidos en DB y archivos físicos del artifact plane.

## Alcance

- `runtime/storage` en modos `sqlite`, `postgres` y `hybrid`.
- Índices de `artifacts`, `ledger_events`, `telemetry_snapshots`, `reasoning_traces`, `sessions`.

## Política de backup

1. SQLite:
   - backup diario del archivo DB con retención configurable.
2. PostgreSQL:
   - `pg_dump` lógico programado con retención mínima de 7 snapshots.
3. Artifact plane:
   - snapshot incremental diario por `run_id`.
4. Todas las copias deben incluir fecha/hora UTC y checksum de archivo.

## Política de reconciliación

1. Reconciliación periódica (`al menos diaria`) entre tabla `artifacts` y filesystem.
2. Casos de fallo:
   - Registro en DB sin archivo físico.
   - Archivo físico sin registro en DB.
   - Hash o tamaño divergente.
3. Toda divergencia se registra como evento con severidad operacional.

## Estrategia de recuperación

1. Prioridad 1: restaurar metadatos DB.
2. Prioridad 2: restaurar artifacts por `run_id`.
3. Si artifact no es recuperable:
   - marcar estado como `missing_artifact`,
   - bloquear promoción/certificado para ese episodio.

## Gates de aceptación

1. Existe procedimiento reproducible de backup y restore.
2. Existe verificación automática de reconciliación en entorno de pruebas.
3. No se promueve estado sin integridad DB↔filesystem validada.
