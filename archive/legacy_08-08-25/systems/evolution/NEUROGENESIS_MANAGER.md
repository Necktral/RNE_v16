# NeurogenesisManager: Documentación

## Descripción General
`NeurogenesisManager` es una clase diseñada para gestionar la expansión dinámica y consolidación de capas neuronales en modelos de redes neuronales profundas, siguiendo principios inspirados en neurociencia y aprendizaje auto-adaptativo. Permite el crecimiento controlado de unidades neuronales, la consolidación de memoria asociativa y la gestión de plasticidad en arquitecturas modulares.

---

## Uso Principal
- **Expansión de capas**: Añade nuevas unidades a capas seleccionadas cuando se detecta baja eficiencia, baja densidad de Fisher, baja variación epistémica y suficiente información mutua.
- **Plasticidad controlada**: Ajusta los nuevos parámetros con un factor de plasticidad durante varios pasos de entrenamiento.
- **Consolidación**: Fija los parámetros de las nuevas unidades cuando la memoria asociativa alcanza estabilidad.

---

## Métodos Clave

### `should_grow_layer(name, ctx)`
Determina si una capa debe expandirse según métricas de eficiencia, densidad de Fisher, variación epistémica y foco consciente. Implementa un sistema de cooldown para evitar crecimiento excesivo.

### `compute_diversity(acts)`
Calcula la diversidad entre activaciones de una capa usando la correlación inversa. Favorece la selección de unidades menos redundantes.

### `grow(context)`
Marca en el contexto que se ha solicitado crecimiento. (Implementación dummy, puede ser extendida para lógica personalizada.)

### `step(context)`
Marca en el contexto que se ha realizado un paso de neurogénesis. (Implementación dummy.)

### `update_dependent_layer(dep_name, old_size, new_size)`
Actualiza capas dependientes para que sean compatibles con el nuevo tamaño de la capa expandida, preservando pesos y memoria asociativa.

### `apply_soft_grow()`
Ajusta gradualmente los nuevos parámetros de las unidades añadidas usando los gradientes acumulados y el factor de plasticidad. El proceso se repite durante varios pasos definidos por `soft_grow_steps`.

### `consolidate(stability_threshold)`
Fija los parámetros de las capas expandidas cuando la desviación estándar de la memoria asociativa cae por debajo de un umbral, consolidando el aprendizaje.

---

## Parámetros Importantes
- `growth_unit`: Número de unidades a añadir por expansión.
- `plasticity_factor`: Factor de ajuste para los nuevos parámetros.
- `eff_threshold`, `fisher_threshold`, `delta_epist_min`, `mutual_info_min`: Umbrales para decidir el crecimiento.
- `soft_grow_steps`: Pasos de ajuste suave tras el crecimiento.
- `cooldown_steps`: Pasos de espera antes de permitir un nuevo crecimiento en la misma capa.

---

## Ejemplo de Uso
```python
manager = NeurogenesisManager(model, ["layer1"], {}, growth_unit=8)
context = {"delta_epist": -0.02, "mutual_info": 0.1, "efficiency": 0.1, "fisher_density": 0.1, "conscious_focus": 1.0}
if manager.should_grow_layer("layer1", context):
    manager.grow(context)
    manager.apply_soft_grow()
    manager.consolidate()
```

---

## Notas
- El diseño permite extender la lógica de crecimiento y consolidación para arquitecturas personalizadas.
- Se recomienda monitorear el uso de memoria y la estabilidad del entrenamiento al usar neurogénesis dinámica.
