import os
import pandas as pd

import torch
from src.aeon_fenix.models.fenix_agent import FenixAgent
from src.aeon_fenix.core.loss_elite import loss_elite_v1_2
from src.aeon_fenix.envs.env_dummy import DummyEnv

def run_fenix_training(epochs=3, batch_size=16, seq_len=10, input_dim=32):
    print("🔥 Entrenamiento real Fenix + Elite v1.2")

    log_path = "logs/fenix_train_log.csv"
    os.makedirs("logs", exist_ok=True)
    log_data = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = FenixAgent(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)

    env = DummyEnv(seq_len=seq_len, input_dim=input_dim, batch_size=batch_size)
    obs = env.reset().to(device)

    e_prev, e_sum = 0.0, 0.0

    for epoch in range(epochs):
        print(f"\n📦 Epoch {epoch+1}/{epochs}")

        # === Forward: modelo fenix ===
        x_pred, post, prior, cov = agent(obs)

        # === Configuración experimental ===
        efe = torch.tensor(1.5).to(device)
        grad_dummy = torch.tensor(1.0).to(device)
        w_norm = sum(p.norm(2) for p in agent.parameters())

        # === Loss elite ===
        loss, details, e_prev, e_sum = loss_elite_v1_2(
            obs, x_pred, post, prior, efe, cov,
            grad_norm=grad_dummy, weight_norm=w_norm,
            flops=1.2e12, act_mem_gb=torch.tensor(6.5),
            e_prev=e_prev, e_sum=e_sum,
            d_target=1.0
        )
        print(loss)  # debe ser tipo: tensor(0.7205, grad_fn=<AddBackward0>)
        # === Backprop ===
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # === Interacción con el entorno ===
        z = post.loc  # latente mu
        next_obs, reward, done = env.step(z.detach().cpu())
        obs = next_obs.to(device)
        epoch_reward = reward.mean().item()
        print(f"  reward        : {epoch_reward:.5f}")
        log_data.append({
            "epoch": epoch + 1,
            "loss": loss.item(),
            "recon": details["recon"],
            "kl": details["kl"],
            "beta": details["beta"],
            "efe": details["efe"],
            "spec_reg": details["spec_reg"],
            "grad_penalty": details["grad_penalty"],
            "hw_cost": details["hw_cost"],
            "reward": epoch_reward
        })
        # === Reporte ===
        print(f"🔧 Loss total: {loss.item():.5f}")
        for k, v in details.items():
            print(f"  {k:<14}: {v:.5f}")

    # Guardar métricas en archivo CSV
    df = pd.DataFrame(log_data)
    df.to_csv(log_path, index=False)
    print(f"\n📁 Log guardado en {log_path}")
