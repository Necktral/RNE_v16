# Evaluación inicial del docente local 7B

Fecha: 2026-07-14. Esta medición es operacional y semántica; todavía no demuestra
mejora cognitiva porque no incluye outcomes pareados.

## Entorno nativo verificado

- volumen: `/home/wis` sobre `/dev/sdg3[/home/wis]`, `ext4`;
- GPU: NVIDIA GeForce RTX 2070 Max-Q, 8192 MiB;
- modelo: `OpenThinker3-7B-Q4_K_M.gguf`;
- runtime: `llama-cli` CUDA bajo `/home/wis/rnfe_models/tools/llama.cpp-src/build-cuda/`.

La documentación anterior que ubicaba modelo y runtime en `/mnt/d` está desfasada.

## Ensayo reproducible inicial

Prompt fijo: herida de severidad `0.80` al elegir `deactivate_cooling`; alternativas
válidas `activate_cooling`, `maintain_cooling`, `deactivate_cooling`. Temperatura 0,
seed 42, 128 tokens máximos y schema `experience_lesson`.

| Medida | Resultado |
|---|---:|
| Tiempo total del proceso | 6.54 s |
| Prompt | 612.6 tokens/s |
| Generación | 31.8 tokens/s |
| RSS máximo del proceso host | 4,899,940 KiB |
| JSON conforme al schema | sí |
| `avoid` igual a la acción dañina | no |
| `prefer` dentro del catálogo | no |
| lección informativa mínima | no |

El resultado cumplió forma pero no semántica. Por eso `Teacher` ahora registra
`teacher_raw_semantic_valid` y `teacher_repairs`, vuelve a enlazar `avoid` a la herida
observada, limita `prefer` al catálogo y usa un fallback acotado si la lección carece
de contenido. La reparación sigue siendo propuesta no certificada.

## Dictamen provisional

El 7B es técnicamente utilizable como generador local, pero este ensayo lo rechaza
como maestro autónomo. Codex puede entrar por `register_external_lesson` con
`teacher_source=codex_frontier`, bajo el mismo contrato sin autoridad. La comparación
de eficacia exige, por situación y seed, tres variantes: `no_teacher`, `local_7b` y
`codex_frontier`, midiendo cambio de conducta, reducción de severidad, reincidencia,
latencia y coste. Hasta completar repeticiones held-out, ninguna fuente se promueve a
currículo ni a datos de entrenamiento.
