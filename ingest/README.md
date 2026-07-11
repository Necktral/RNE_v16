# ingest/

Zona de **staging/cuarentena para ingestión externa**, requerida por CANON §13
(`canon/normative/CANON_RNFE_v3_2_rc1.md`, contratos mínimos del repositorio).

Rol: todo material externo (corpus, documentos, datos de terceros) que aspire a
entrar al organismo debe aterrizar primero acá, y solo se promociona al tronco
vivo mediante el proceso de gobernanza (contratos + certificación). CANON
(errores estratégicos prohibidos, v3.1 §9.5) prohíbe explícitamente "conectar
ingestión externa directa al tronco vivo": este directorio es la frontera que
materializa esa prohibición.

Estado actual: **vacío por diseño** — hoy no existe pipeline de ingestión
externa en el runtime (verificado 2026-07-10, B33: los únicos usos de "ingest"
en código son métodos internos como `ConstitutionalCourtRuntime.ingest_episode`,
que no son ingestión externa). El directorio existe para cumplir el contrato
estructural del CANON y fijar el punto de entrada cuando ese pipeline exista.

Nota de gobernanza: la equivalencia del otro directorio faltante de CANON §13
(`roadmap/`) quedó registrada en `governance/aliases.yaml` (→ `docs/strategy` +
`governance/backlog`).
