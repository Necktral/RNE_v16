# POLICY — Artifact Plane en Disco D (WSL2)

## Objetivo

Definir política física para almacenamiento de artefactos del runtime RNFE bajo Windows 11 + WSL2.

## Default normativo

- `RNFE_ARTIFACT_ROOT=/mnt/d/rnfe_artifacts`

## Reglas de operación

1. El runtime persiste únicamente metadatos de artefactos en DB.
2. Todo artefacto materializado debe incluir hash SHA-256 y tamaño en bytes.
3. La ruta física debe organizarse por `run_id/kind/hash-prefix`.
4. El artifact plane no debe vivir dentro de `archive/`.

## WSL2 y rendimiento

1. Evitar hot-path de escritura intensiva directamente sobre NTFS montado en `/mnt/d` para loops críticos.
2. Para cargas de alto throughput usar volumen Linux (ext4/VHDX) y replicar a `D` por lotes.
3. Mantener PostgreSQL data dir en volumen Linux del contenedor; no mapear data principal directamente a NTFS.

## Seguridad operativa

1. No persistir secretos dentro de metadatos de artifacts.
2. Limitar permisos de escritura al usuario de runtime.
3. Validar hash al leer artefactos críticos antes de promoción/certificación.

## Gate de aceptación

1. Artifact index en DB sin blobs.
2. Integridad verificada (`sha256`, `size_bytes`) en tests de regresión.
3. Política documentada y aplicada en despliegues locales.
