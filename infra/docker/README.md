# Infra Docker de Persistencia RNFE

Este compose solo levanta infraestructura de persistencia.

## Servicios

- `postgres` (principal)
- `adminer` (opcional, perfil `admin`)

## Uso rapido

1. Crear `.env` desde el ejemplo:

```bash
cp infra/docker/.env.example infra/docker/.env
```

2. Levantar PostgreSQL:

```bash
docker compose -f infra/docker/docker-compose.yml up -d postgres
```

3. Levantar Adminer (opcional):

```bash
docker compose -f infra/docker/docker-compose.yml --profile admin up -d
```

4. DSN sugerido para runtime:

```text
postgresql://rnfe:rnfe_local_dev_only@localhost:5432/rnfe
```

## WSL2 + disco D (operación recomendada)

1. Mantener el data dir de PostgreSQL dentro del volumen Docker (`rnfe_pgdata`) para evitar degradación en I/O.
2. Usar `/mnt/d` para artifact plane y exportes, no como hot path del data directory de PostgreSQL.
3. Para cargas altas en artefactos, preferir escritura inicial en ruta Linux (ext4) y sincronización por lote a `D`.

## Comandos de verificación rápida

```bash
docker compose -f infra/docker/docker-compose.yml ps
docker compose -f infra/docker/docker-compose.yml logs -f postgres
```

## Integración con tests opt-in de PostgreSQL

```bash
export RNFE_RUN_PG_TESTS=1
export RNFE_POSTGRES_DSN=postgresql://rnfe:rnfe_local_dev_only@localhost:5432/rnfe
pytest -q -m requires_postgres
```
