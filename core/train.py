import torch
from typing import Dict, Any
from types import SimpleNamespace

class QuantumDistributedTrainer:
    def __init__(self, model, train_loader, config, tensorboard_writer=None):
        self.model = model
        self.train_loader = train_loader
        self.config = config if config is not None else SimpleNamespace()
        # El optimizador se asignará desde el orquestador
        self.optimizer = None
        # Los hooks de poda se registrarán aquí.
        self.pruning_hooks: Dict[str, torch.Tensor] = {}
        self.tensorboard_writer = tensorboard_writer

    def _distributed_train_step(self, z_prev, a_prev, o_t):
        """
        Realiza un único paso de entrenamiento con los datos de la secuencia.
        Devuelve (loss, new_z) para que el orquestador pueda continuar el ciclo.
        """
        if self.optimizer is None:
            raise ValueError("Optimizer not set.")
        self.model.train()
        self.optimizer.zero_grad()
        try:
            # Diagnóstico: verifica finitud de entradas
            for name, p in self.model.named_parameters():
                if not torch.isfinite(p).all():
                    print(f"🔴 Peso corrupto: {name}")
            assert torch.isfinite(z_prev).all() and torch.isfinite(a_prev).all() and torch.isfinite(o_t).all(), "Entradas corruptas"
            output = self.model.generative_model(z_prev, a_prev)
            target = o_t[:, :output.shape[1]]
            print(f"output.shape: {output.shape}, target.shape: {target.shape}")
            # Imprime pred y target cada 20 pasos
            if hasattr(self, 'step_count'):
                self.step_count += 1
            else:
                self.step_count = 1
            if self.step_count % 20 == 0:
                print(f"pred[0][:5] = {output[0][:5].detach().cpu().numpy()}")
                print(f"target[0][:5] = {target[0][:5].detach().cpu().numpy()}")
            assert output.shape == target.shape, f"{output.shape} vs {target.shape}"
            loss = torch.nn.functional.mse_loss(output, target)
            if not torch.isfinite(loss):
                print(f"🔴 Loss corrupto en iter: {loss}")
                return None
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            print(f"GradNorm @step{self.step_count}: {grad_norm:.2e}")
            self.optimizer.step()
            # Log a TensorBoard si está disponible
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar("train/loss", loss.item(), self.step_count)
                self.tensorboard_writer.add_scalar("train/grad_norm", grad_norm, self.step_count)
                # Log de histogramas de pesos
                for name, param in self.model.named_parameters():
                    self.tensorboard_writer.add_histogram(f"weights/{name}", param, self.step_count)
                # Log de histogramas de gradientes
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.tensorboard_writer.add_histogram(f"grads/{name}", param.grad, self.step_count)
            new_z = output.detach()
            return loss.item(), new_z
        except Exception as e:
            print(f"[AEON][Trainer] Error en _distributed_train_step: {e}")
            return None

    def apply_pruning_payload(self, payload: Dict[str, torch.Tensor]):
        """
        Aplica un payload de poda al modelo.
        En un caso real, esto registraría hooks de forward.
        """
        for layer_path, mask in payload.items():
            # En un sistema real, aquí se registraría un hook
            # para aplicar la máscara durante el forward pass.
            # Para el test, simplemente poblamos el diccionario de hooks.
            self.pruning_hooks[layer_path] = mask
            print(f"INFO: Pruning hook 'registered' for {layer_path}")

    def apply_adaptation(self, adaptation_payload: Dict[str, Any]):
        """
        Aplica un payload de adaptación genérico desde el AutoMutator.
        Discrimina entre poda y neurogénesis.
        """
        payload_type = adaptation_payload.get("type")
        payload = adaptation_payload.get("payload")

        if payload_type == "pruning":
            if isinstance(payload, dict):
                self.apply_pruning_payload(payload)
            else:
                raise ValueError(f"Invalid payload for pruning: {type(payload)}")
        elif payload_type == "neurogenesis":
            # La neurogénesis implica la creación de nuevas neuronas, por lo que
            # el optimizador debe ser reiniciado para incluir los nuevos parámetros.
            self.reset_optimizer()
            print("INFO: Neurogenesis adaptation applied (optimizer reset).")
        else:
            print(f"WARNING: Unknown adaptation payload type received: {payload_type}")

    def reset_optimizer(self):
        """
        Crea una nueva instancia del optimizador.
        """
        if self.config and hasattr(self.config, 'lr'):
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
            print("INFO: Trainer optimizer has been reset.")
        else:
            raise ValueError("Trainer configuration is missing or does not have a learning rate.")

