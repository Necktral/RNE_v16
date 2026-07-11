"""Loop principal de entrenamiento/orquestación desacoplado."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import torch

from runtime.core.events import Event
from runtime.core.metrics import CRITICAL_TEMP, MAX_VRAM_GB


class TrainingLoop:
    def __init__(
        self,
        *,
        orchestrator: Any,
        telemetry_collector,
        snapshot_service,
        adaptation_controller,
        lifecycle,
    ):
        self.orchestrator = orchestrator
        self.telemetry_collector = telemetry_collector
        self.snapshot_service = snapshot_service
        self.adaptation_controller = adaptation_controller
        self.lifecycle = lifecycle

    async def _get_next_observation(self):
        try:
            return next(self.orchestrator.train_iter)[0]
        except StopIteration:
            self.orchestrator.train_iter = iter(self.orchestrator.train_loader)
            return next(self.orchestrator.train_iter)[0]

    async def run(self) -> None:
        orch = self.orchestrator
        orch.logger.info("[DEBUG] training_loop.run iniciado")
        challenger = None
        if hasattr(orch, "modules") and orch.modules:
            # B9: el nombre real de la clase es CognitiveSelfChallengeAGI y vive en
            # families/dia_adv (el shim legacy fue podado en B13). Requiere monitor y
            # estados cuánticos: se cablean desde el meta_optimizer, tras guard.
            from runtime.reasoning.families.dia_adv.cognitive_self_challenge import (
                CognitiveSelfChallengeAGI,
            )

            meta_optimizer = getattr(orch, "meta_optimizer", None)
            monitor = getattr(meta_optimizer, "physics_monitor", None)
            quantum_state = None
            if meta_optimizer is not None:
                quantum_state = getattr(meta_optimizer, "state", {}).get("quantum_state")
            if monitor is not None and quantum_state is not None:
                challenger = CognitiveSelfChallengeAGI(
                    orch.modules,
                    monitor,
                    {uid: quantum_state for uid in orch.modules},
                )
            else:
                orch.logger.warning(
                    "CognitiveSelfChallengeAGI no cableado: meta_optimizer sin "
                    "physics_monitor/quantum_state disponibles."
                )

        cycle = 0
        if orch.tensorboard_writer is not None:
            hparams = {
                "latent_dim": getattr(orch, "latent_dim", 32),
                "optimizer": type(orch.optimizer).__name__,
                "lr": orch.optimizer.param_groups[0]["lr"]
                if hasattr(orch.optimizer, "param_groups")
                else None,
            }
            orch.tensorboard_writer.add_text("hparams", str(hparams), 0)

        while not orch._shutdown.is_set():
            try:
                cycle += 1
                orch.logger.info(f"Ciclo {cycle} iniciado.")
                z_prev = orch.z.to(orch.device)
                a_prev = orch.a.to(orch.device)
                o_t = (await self._get_next_observation()).to(orch.device)

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    orch.executor,
                    lambda: orch.trainer._distributed_train_step(
                        z_prev, a_prev, o_t.view(1, -1)
                    ),
                )
                if result is None:
                    orch.logger.warning("Paso de entrenamiento devolvió None.")
                    await asyncio.sleep(0.1)
                    continue
                loss, new_z = result
                if loss is None or new_z is None:
                    orch.logger.warning("Paso de entrenamiento devolvió loss/new_z nulo.")
                    await asyncio.sleep(0.1)
                    continue

                orch.z = new_z
                orch.history.append(loss)
                self.telemetry_collector.update_metrics(
                    metrics=orch.metrics,
                    history=list(orch.history),
                    pynvml_module=orch.pynvml,
                    gpu_handle=getattr(orch, "gpu_handle", None),
                )
                metrics_dict = orch.metrics.as_dict()

                self.snapshot_service.persist_snapshot(
                    cycle=cycle,
                    metrics=metrics_dict,
                    run_id=getattr(orch, "current_run_id", None),
                )
                orch.event_bus.emit(
                    "orchestrator_metrics", {"cycle": cycle, "metrics": metrics_dict}
                )

                vfe = getattr(orch.metrics, "vfe", None)
                eta = getattr(orch.metrics, "eta_bayes", None)
                orch.drift_predictor.update(eta, vfe)
                alerta, razon = orch.drift_predictor.check_drift(cycle)
                if alerta:
                    orch.logger.warning(f"[DERIVA EPISTÉMICA] {razon}. Intervención.")
                    orch.drift_predictor.force_mutation(razon)

                if (
                    orch.metrics.vram_usage_gb < 0.95 * MAX_VRAM_GB
                    and orch.metrics.temperature < 0.95 * CRITICAL_TEMP
                ):
                    orch.meta_optimizer.step(
                        {
                            "vram": orch.metrics.vram_usage_gb / MAX_VRAM_GB,
                            "thermal": orch.metrics.temperature / CRITICAL_TEMP,
                            "entropy": orch.metrics.entropy,
                            "cognitive_load": orch.metrics.stability,
                        },
                        lambda uid: eta if eta is not None else 1.0,
                    )
                else:
                    orch.logger.warning(
                        "Mutación/NAS bloqueada por límites físicos: "
                        f"vram={orch.metrics.vram_usage_gb:.2f}, temp={orch.metrics.temperature:.2f}"
                    )

                ctx = self.adaptation_controller.build_context(
                    metrics=orch.metrics,
                    history=list(orch.history),
                    model=orch.combined_model,
                    optimizer=orch.optimizer,
                    device=orch.device,
                )
                orch.optimizer = await self.adaptation_controller.apply_adaptations(
                    loop=loop,
                    executor=orch.executor,
                    auto_mutator=orch.auto_mutator,
                    trainer=orch.trainer,
                    optimizer=orch.optimizer,
                    context=ctx,
                    logger=orch.logger,
                )

                if challenger is not None:
                    resultado = challenger.generate_challenge(cycle)
                    if resultado is not None:
                        orch.event_bus.emit(
                            "cognitive_challenge", {"cycle": cycle, "result": resultado}
                        )

                await orch.bus.publish(
                    Event(topic=orch.HEARTBEAT_TOPIC, payload=metrics_dict, severity="INFO")
                )
                orch.event_bus.emit("heartbeat", {"cycle": cycle, "metrics": metrics_dict})

                if cycle % 1000 == 0 and getattr(orch, "scheduler", None) is not None:
                    try:
                        val_loss = await loop.run_in_executor(
                            orch.executor,
                            lambda: orch.eval_loop(orch.combined_model, orch.val_loader),
                        )
                        if not isinstance(val_loss, (float, int)) or not np.isfinite(val_loss):
                            val_loss = float("nan")
                        if (
                            orch.scheduler is not None
                            and val_loss is not None
                            and np.isfinite(val_loss)
                        ):
                            orch.scheduler.step(val_loss)
                        if hasattr(torch, "save") and hasattr(orch.combined_model, "state_dict"):
                            torch.save(orch.combined_model.state_dict(), f"checkpoints/aeon_{cycle}.pt")
                    except Exception as exc:
                        orch.logger.exception(f"Error en validación/checkpoint: {exc}")

                await asyncio.sleep(0.1)
                if orch.max_cycles and cycle >= orch.max_cycles:
                    orch.logger.info(f"✓ Alcanzados {cycle} ciclos — apagando Orchestrator.")
                    orch._shutdown.set()
                    break
            except Exception as exc:
                self.lifecycle.mark_degraded()
                orch.logger.exception(f"Excepción inesperada en ciclo {cycle}: {exc}")
                orch._shutdown.set()
                break

        if orch.tensorboard_writer is not None:
            orch.tensorboard_writer.flush()
            orch.tensorboard_writer.close()
        orch.logger.info("[DEBUG] training_loop.run finalizado")
