# Checklist de Reconstrucción para `ModuleOrchestrator`

**Objetivo:** Transformar el orquestador de un "actor" monolítico a un "director" desacoplado que coordina módulos especializados.

---

### Fase 1: Descomposición y Limpieza (Principio de Responsabilidad Única)

- [ ] **Mover Modelos Probabilísticos:** Extraer las clases `GenerativeModel` y `ApproximatePosterior` a un nuevo archivo dedicado: `src/core/probabilistic_models.py`.
- [ ] **Mover Infraestructura Asíncrona:** Extraer las clases `WorkerPool` y `EventBus` a un nuevo archivo de infraestructura: `src/core/infrastructure.py`.
- [ ] **Mover Métricas:** Extraer la clase `SelfAwarenessMetrics` a `src/homeostasis/metrics.py` para que viva junto a los otros componentes de monitoreo.
- [ ] **Centralizar Constantes:** Mover las constantes físicas (`MAX_VRAM_GB`, `THERMAL_THRESHOLD`, etc.) a `config/config.yaml` o a un archivo `src/constants.py` para que sean accesibles globalmente.

---

### Fase 2: Integración de Módulos (Delegación de Tareas)

- [ ] **Integrar Monitoreo Físico:**
    - [ ] Eliminar el uso directo de `pynvml` y `psutil` del orquestador.
    - [ ] Instanciar el `PhysicsMonitor` (de `src/homeostasis/energy_sensors.py` o un módulo similar) en el `__init__` del orquestador.
    - [ ] Reemplazar el contenido de `_update_metrics` con una única llamada al método del monitor (ej. `self.physics_monitor.get_system_metrics()`).

- [ ] **Integrar Control Homeostático:**
    - [ ] Instanciar el `HomeostasisController` en el `__init__` del orquestador.
    - [ ] Reemplazar las acciones homeostáticas simuladas (`_thermal_veto`, `_inject_noise`, etc.) con llamadas a los métodos correspondientes del `HomeostasisController`. El orquestador solo envía la señal; el controlador ejecuta la acción.

- [ ] **Integrar Módulos de Evolución:**
    - [ ] Instanciar `AutoMutator`, `KatanaPruner` y `NeurogenesisManager` en el `__init__` del orquestador.
    - [ ] Reemplazar `_trigger_pruning` con una llamada real a `self.katana_pruner.prune(...)`.
    - [ ] Reemplazar `_reduce_complexity` con una llamada a `self.auto_mutator.trigger_adaptation(...)`, que a su vez decidirá si podar o generar nuevas neuronas.

---

### Fase 3: Refinamiento del Rol del Orquestador

- [ ] **Recibir Modelo Externo:** Modificar el `__init__` para que el orquestador reciba una instancia del modelo principal (`AeonModel`) en lugar de crear sus propios modelos. Todas las operaciones se realizarán sobre esta instancia externa.
- [ ] **Confirmar Flujo de Control:** Verificar que el flujo final sea:
    1.  `_monitor_vitals` detecta una anomalía en las métricas.
    2.  Publica un `Event` en el `EventBus` (ej. `VRAMUsageHigh`).
    3.  El `handler` correspondiente en el orquestador (`handle_vram_crisis`) se activa.
    4.  El `handler` **delega la tarea** al `WorkerPool`, llamando al método del **módulo especializado** apropiado (ej. `self.homeostasis_controller.execute_pruning()`).
- [ ] **Limpiar Código Muerto:** Eliminar todas las clases y métodos que se volvieron redundantes después de la refactorización.
