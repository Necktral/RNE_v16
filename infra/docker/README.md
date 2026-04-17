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
