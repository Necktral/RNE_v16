# Campaña neural integral RNFE

## Propósito y frontera de autoridad

`scripts/run_integral_neural_campaign.py` califica el conectoma, N0–N6, los cinco
agentes centrales, las once extensiones, el maestro local 7B y el impacto global.
Toda salida es evidencia experimental. La campaña puede preparar un artefacto en
SHADOW, pero nunca concede promoción operativa, entrenamiento automático ni
mutación del conectoma.

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
6. Maestro 7B post-experiencia y LifeKernel Tier-3 acotado.
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
