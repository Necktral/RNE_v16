# Campaña neural integral RNFE

## Propósito y frontera de autoridad

`scripts/run_integral_neural_campaign.py` califica el conectoma, N0–N6, los cinco
agentes centrales, las once extensiones, el alumno local 7B y el impacto global.
Toda salida es evidencia experimental. La campaña puede preparar un artefacto en
SHADOW, pero nunca concede promoción operativa, entrenamiento automático ni
mutación del conectoma.

### Contrato docente

- **Codex es el docente externo y autor de currículo candidato.** Sus lecciones son
  evidencia con procedencia, no verdad autoritativa ni permiso de entrenamiento.
- **OpenThinker3-7B es el alumno supervisado y proponente acotado.** No se lo trata
  como maestro autónomo aunque produzca una lección formalmente válida.
- **La autoridad epistémica permanece en los verificadores y en el outcome.** DED,
  LOT-F, NESY, CAU, CTF y C-GWM verifican las afirmaciones que les corresponden; la
  comparación post-outcome determina si una lección mejoró conducta y resultado.
  Certificación y política conservan toda autoridad de admisión.
- Ni Codex ni el 7B pueden promocionar currículo, entrenar, activar un órgano o
  reemplazar un veredicto. Un ejemplo Codex sólo puede entrar al currículo después
  de ganar el gate held-out estratificado y de no regresión.

Mamba 2 se trata como alternativa temporal experimental de N3. No es un canal de
comunicación rápida ni reemplaza la comunicación latente medida y deliberada.

## Entorno nativo

La ejecución se inicia desde Windows mediante la distribución `Ubuntu-24.04`, pero
el repositorio y los hot paths permanecen en ext4 Linux:

```text
/home/wis/Desarrollo/RNE_v16_worktrees/neural-agent-suite
```

La configuración se lee explícitamente desde `/home/wis/Desarrollo/RNE_v16/.env`.
No se copia el DSN al worktree y los manifiestos redactan cualquier clave sensible.
Si `RNFE_ARTIFACT_ROOT` aún apunta al montaje de un arranque Linux directo bajo
`/media/wis/<uuid>`, el runner comprueba que no sea escribible y lo remapea a
`/home/wis/Desarrollo/RNE_v16/rnfe_artifacts`. Ambas rutas quedan declaradas en el
manifiesto; `/home/wis` sigue siendo ext4 de la partición física 3, no `/mnt/c`.
Docker Desktop puede ser invocado mediante `docker.exe` cuando el stub `docker` de
WSL no tenga integración habilitada. Si `postgres:16-alpine` no está disponible y
`postgres:16` ya existe localmente, la campaña genera un override sin secretos.

## Persistencia

- PostgreSQL es el único store oficial de la campaña.
- Cada `campaign_id` produce una base `rnfe_campaign_<id>_<hash>`.
- OFF, SHADOW, órganos, semillas y variantes usan `run_id` distintos.
- Los blobs quedan en `RNFE_ARTIFACT_ROOT/integral_campaigns/<campaign_id>` y sus
  metadatos, tamaños y SHA-256 se indexan en PostgreSQL.
- Si PostgreSQL cae, el bloque falla y se genera un checkpoint; no existe failover
  silencioso a SQLite.
- SQLite se usa solo en el bloque de contingencia y como scratch efímero de ramas
  contrafactuales. La migración SQLite→PostgreSQL se ejecuta dos veces para probar
  idempotencia, pero no reemplaza evidencia oficial.
- El cierre de cada fase genera un `pg_dump` con SHA-256. Si `pg_dump` no está
  instalado en WSL, se ejecuta dentro del contenedor `rnfe-postgres`.

## Ejecución

Desde el worktree:

```bash
PYTHONPATH=. /home/wis/Desarrollo/RNE_v16/.venv/bin/python \
  scripts/run_integral_neural_campaign.py \
  --campaign-id neural-20260715 preflight
```

Ensayo corto, con límite de 90 minutos y finalización del bloque atómico vigente:

```bash
PYTHONPATH=. /home/wis/Desarrollo/RNE_v16/.venv/bin/python \
  scripts/run_integral_neural_campaign.py \
  --campaign-id neural-20260715 run --phase rehearsal
```

El JSON final contiene el hash del checkpoint. La nocturna no arranca sin que ese
hash sea entregado de forma explícita:

```bash
PYTHONPATH=. /home/wis/Desarrollo/RNE_v16/.venv/bin/python \
  scripts/run_integral_neural_campaign.py \
  --campaign-id neural-20260715 run --phase overnight \
  --checkpoint <hash-aprobado>
```

Una interrupción deja el bloque `running` o `failed`; `resume` lo reinicia desde
cero y conserva los bloques terminados:

```bash
PYTHONPATH=. /home/wis/Desarrollo/RNE_v16/.venv/bin/python \
  scripts/run_integral_neural_campaign.py \
  --campaign-id neural-20260715 resume --checkpoint <hash>
```

La ablación diagnóstica se ejecuta con un `campaign_id` nuevo. Produce, en orden
canónico, OFF, `shadow-none` (runtime SHADOW con N1–N6 deshabilitados), cada N1–N6
aislado, all-on y cada leave-one-out. Todas las variantes usan los mismos estados
iniciales, semillas y pasos, y escriben resultados pareados en
`ablation/matrix.json`:

```bash
PYTHONPATH=. /home/wis/Desarrollo/RNE_v16/.venv/bin/python \
  scripts/run_integral_neural_campaign.py \
  --campaign-id neural-ablation-20260717 ablate --life-steps 3 \
  --seed 811001 --seed 811101 --seed 811201
```

`ablate` fija explícitamente `RNFE_NEURAL_MODE` y
`RNFE_NEURAL_DISABLED_ORGANS` para cada lane, ignora perfiles heredados del shell y
rechaza sobrescribir una matriz existente. El reporte mantiene
`training_authorized=false`, `staging_authorized=false` y
`promotion_authorized=false`; no alimenta el comando `stage`.

Los dos contrastes interpretables por órgano se persisten tanto por semilla como en
promedio:

- `all-on_minus_without`: all-on menos `without-Nx`; mide la contribución marginal
  de Nx dentro del conjunto completo.
- `only_minus_shadow_none`: `only-Nx` menos `shadow-none`; mide capacidad aislada
  por encima del costo y ciclo de vida del runtime SHADOW vacío.

Los deltas `paired_vs_off` se conservan como diagnóstico global, pero OFF y SHADOW
no atraviesan el mismo runtime y por eso ese delta no atribuye efecto a un órgano.
`closure_rate` tampoco replica certificación: cuenta pasos cuyo `episode_id` tiene un
evento durable `episode.closed` en PostgreSQL; `certification_rate` sigue contando
el estado certificado de los vitales.

La preparación copia N1 configurado sólo si su manifiesto es válido y el hash del
artefacto coincide. Un candidato N1 recalibrado dentro de la campaña siempre tiene
precedencia. Cada perfil registra por semilla la procedencia observada de cada
backend (`model_bound`, `reference` o `disabled`); si N5 no tiene manifiesto, queda
declarado explícitamente como reference y no como modelo entrenado.

Al iniciar la ablación se realiza una sola lectura real de recursos host/GPU. El
snapshot y su hash canónico quedan sellados en
`ablation/resource-snapshot.json`; cada lane y cada paso reutilizan exactamente ese
valor mediante `LifeKernelConfig.resource_snapshot_override`. Así la carga cambiante
del propio orden secuencial no contamina los pares. Las campañas normales no reciben
el override y continúan usando `RNFE_HOST_SENSING` real en cada ciclo.

Un árbol Git sucio no se presenta como el commit base. Antes de ejecutar, `ablate`
materializa `ablation/worktree.diff` con `git diff --binary --full-index HEAD`,
registra su SHA-256, el estado dirty y los hashes de archivos no trackeados en
`worktree-status.json`. Al finalizar vuelve a calcular el snapshot; si el árbol
cambió durante la matriz, escribe `stable_during_run=false` y falla cerrado.

### Límites de interpretación de la ablación

- Tres semillas son evidencia diagnóstica, no potencia estadística suficiente para
  promoción ni causalidad general.
- Las lanes corren en secuencia: el snapshot fijo elimina la variación de recursos
  que entra a los vitales, pero no elimina warm-up, cachés ni sesgo de orden en la
  latencia medida.
- Al reutilizar un snapshot fijo, CPU/RAM/VRAM/temperatura describen la condición
  sellada de control; no miden el consumo incremental real de cada órgano.
- SHADOW observa candidatos y consumo de evidencia sin aplicar autoridad. El
  experimento no mide el efecto de una política neural autorizada sobre acciones.
- `all-on_minus_without` puede contener interacciones con otros órganos y
  `only_minus_shadow_none` puede omitirlas; ambos deben leerse juntos.

`report` reconstruye el veredicto desde evidencia ya persistida. `stage` exige una
nocturna cerrada, checkpoint válido y todos los gates en verde:

```bash
PYTHONPATH=. /home/wis/Desarrollo/RNE_v16/.venv/bin/python \
  scripts/run_integral_neural_campaign.py \
  --campaign-id neural-20260715 report

PYTHONPATH=. /home/wis/Desarrollo/RNE_v16/.venv/bin/python \
  scripts/run_integral_neural_campaign.py \
  --campaign-id neural-20260715 stage
```

El staging conserva `promotion_eligible=false`, `promotion_authorized=false` y
`activation_automatic=false`; solo añade la prueba
`shadow_qualification_passed=true`.

## Operación nocturna desatendida

`scripts/supervise_integral_neural_campaign.py` es la envolvente operativa para
noches sin operador. Genera un identificador estable por fecha local y commit,
adquiere un lock exclusivo y ejecuta ensayo, autorización de política, nocturna,
reporte, reconciliación y validación física del dump. La autorización persistida
declara el mandato permanente del usuario para diagnóstico nocturno desatendido;
no autoriza entrenamiento, promoción ni activación.

El supervisor aplica estas guardas:

- máximo de 10 horas de pared y cuatro intentos por fase;
- heartbeat cada 15 segundos con disco, temperatura y memoria GPU;
- pausa cerrada con menos de 20 GiB libres o tres muestras GPU sobre 88 °C;
- lock `flock` para impedir campañas solapadas;
- backoff de 30, 120 y 300 segundos ante fallos transitorios;
- reconstrucción de checkpoint solo desde el manifiesto del mismo commit;
- repetición desde cero de cualquier bloque `running` o `failed`;
- cuarentena automática si fallan gates o integridad;
- staging automático únicamente de un artefacto `qualified_shadow_only` cuando
  todos los gates ya otorgaron `staging_authorized=true`.

El launcher estable es:

```bash
/home/wis/Desarrollo/RNE_v16_worktrees/neural-agent-suite/scripts/run_nightly_neural_supervisor.sh
```

Windows Task Scheduler lo ejecuta diariamente a las 22:00 en `Ubuntu-24.04`, con
`StartWhenAvailable`, `WakeToRun`, exclusión de instancias paralelas, límite de diez
horas y tres reinicios del launcher. La tarea se instala o actualiza con:

```powershell
& "\\wsl.localhost\Ubuntu-24.04\home\wis\Desarrollo\RNE_v16_worktrees\neural-agent-suite\scripts\install_nightly_neural_task.ps1"
```

Cada campaña escribe `SUPERVISOR.json`, logs de intentos, aprobación desatendida y,
cuando corresponde, `QUARANTINE.json`. La evidencia oficial sigue en PostgreSQL y
los secretos continúan leyéndose solo desde el `.env` raíz.

## Secuencia y evidencia

1. Entorno, PostgreSQL, conectoma de 22 nodos/38 conexiones y artefactos.
2. Suite basal y pruebas `requires_postgres` explícitas.
3. Bloques de presupuesto equivalente N0–N6.
4. Tres pares LifeKernel OFF/SHADOW con sensado real, MSRC y checkpoints.
5. Cinco agentes centrales y once extensiones observados desde el cierre real del
   episodio; toda autoridad debe ser `none` o `evidence_only`.
6. Alumno 7B post-experiencia y LifeKernel Tier-3 acotado; Codex conserva el rol
   de docente externo/curriculum bajo verificación y outcome.
7. Contingencia SQLite, reconciliación y dump.
8. Apertura única del holdout sellado, nocturna, A-M0 y veredicto.

N1 entrena tres semillas de desarrollo. La temperatura escalar se selecciona solo
con validación mediante una grilla logarítmica determinista. El test v2 ya observado
se clasifica como desarrollo. La nocturna genera un holdout con otro namespace de
semillas y evalúa el artefacto congelado sin recalibrarlo.

## Gates de staging

- Árbol Git limpio y suite completa en verde.
- Integración PostgreSQL en verde y reconciliación exacta DB↔filesystem.
- Reporte presente para N0–N6, cinco agentes y once extensiones.
- Tres modelos/semillas independientes.
- ECE final menor o igual a `0.10`.
- Límite inferior CI95 estrictamente positivo.
- Cierre, certificación, continuidad y viabilidad sin pérdida mayor a 1 pp.
- Cero violaciones de seguridad y límites de recursos A-M0 satisfechos.

Si N1 falla, la nocturna continúa como diagnóstico y el artefacto queda en
cuarentena. El 7B permanece como estudiante/proponente supervisado aunque obtenga
un resultado favorable; cualquier entrenamiento posterior necesita otra decisión.

## Backlog neural posterior

Orden recomendado después de obtener evidencia de la corrida:

1. Consolidación de memoria y control de contaminación entre escenarios.
2. Metacognición epistémica y calibración del abstention policy.
3. Fusión sensorial y modelo sensorimotor del mundo.
4. Inmunidad cognitiva para artefactos, corpus y maestros.
5. Homeostasis energética/metabólica ligada a MSRC.
6. Ejecutor motor gobernado, siempre detrás de certificación y rollback.
7. Desarrollo/linaje, creatividad horizontal y exocórtex social como capas de
   propuesta, nunca como nuevas autoridades.

## Limitaciones actuales

- El benchmark N4 es evidencia de contrato sintético, no generalización causal.
- N5 no es promovible sin corpus semántico suficiente.
- N6 solo produce propuestas y rechaza aplicar si faltan funciones de aplicación y
  rollback certificadas.
- La utilidad real del 7B depende del costo completo del trial, no solo del timeout
  de generación; el bloque se detiene únicamente en una frontera atómica.
- Windows debe estar encendido o ser capaz de despertar; la tarea usa la distribución
  WSL registrada para el usuario de Windows y no debe ejecutarse como `SYSTEM`.
