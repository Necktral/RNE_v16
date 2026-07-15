# Supervisor nocturno neural desatendido

Fecha: 2026-07-15
Estado: implementado, SHADOW y fail-closed

## Problema observado

La campaña integral tenía reanudación lógica, pero dependía de un operador para
entregar el checkpoint de ensayo. Además, una interrupción dura podía ocurrir entre
la transición `running` del manifiesto y la escritura del checkpoint, y `resume`
rechazaba checkpoints cuya fase ya era `overnight`.

## Solución viva

`scripts/supervise_integral_neural_campaign.py` aporta una máquina operativa externa
al organismo:

1. identidad determinista `neural-nightly-<fecha>-<commit>`;
2. exclusión con `flock` y árbol Git limpio;
3. ejecución/reanudación de ensayo y nocturna;
4. autorización auditable derivada del mandato permanente del usuario;
5. monitor de pared, disco y GPU;
6. reintentos con backoff y checkpoint reconstruible desde el manifiesto;
7. reporte, reconciliación y verificación SHA-256/tamaño del `pg_dump`;
8. cuarentena al fallar cualquier gate;
9. staging opcional, únicamente SHADOW calificado y sin activación automática.

El runner ahora escribe un recovery checkpoint al entrar y salir de cada bloque.
`resume` distingue un inicio nocturno —checkpoint de ensayo cerrado— de una
reanudación nocturna explícita. En ambos casos el bloque incompleto se reinicia desde
cero; no se reutilizan transacciones parciales ni artefactos temporales.

## Frontera de autoridad

El supervisor decide continuidad operacional, no verdad cognitiva. No puede:

- autorizar entrenamiento del 7B o N1;
- promover un modelo a autoridad operativa;
- activar automáticamente un perfil;
- aplicar mutaciones N6;
- degradar PostgreSQL a SQLite como evidencia oficial.

Un staging exitoso conserva `promotion_eligible=false`,
`promotion_authorized=false` y `activation_automatic=false`.

## Riesgos residuales

- Si Windows está apagado y no puede despertar, la tarea arranca al próximo momento
  disponible, pero no puede crear energía ni conectividad.
- WSL pertenece al usuario de Windows; ejecutar la tarea como `SYSTEM` no garantiza
  acceso a la distribución registrada.
- No se elimina evidencia antigua automáticamente. El guard de 20 GiB pausa antes
  de comprometer integridad; la retención requiere una política separada y explícita.
- Una caída física durante la sustitución atómica de un archivo puede requerir
  reconstruir el checkpoint desde el manifiesto actual; nunca desde evidencia del
  bloque parcial.
