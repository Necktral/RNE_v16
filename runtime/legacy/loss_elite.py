import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal

def beta_pid_controller(kl_val, d_target, e_prev, e_sum, k_p=0.1, k_i=0.01, k_d=0.01):
    """Controlador PID adaptativo para escalar KL"""
    e_t = kl_val - d_target
    beta = k_p * e_t + k_i * (e_sum + e_t) + k_d * (e_t - e_prev)
    return beta, e_t, e_sum

def loss_elite_v1_2(x, x_pred, z_post, z_prior, efe_1T, cov_matrix,
                    grad_norm, weight_norm, flops=1e12, act_mem_gb=8,
                    e_prev=0.0, e_sum=0.0, d_target=1.0, params=None):
    """
    x         : Ground truth
    x_pred    : Reconstrucción generada
    z_post    : Posterior Normal(loc, scale)
    z_prior   : Prior Normal(loc, scale)
    efe_1T    : Expected Free Energy acumulado
    cov_matrix: Covarianza de z*
    grad_norm : ||∇L||
    weight_norm: ||θ||
    """
    if params is None:
        params = dict(
            lambda_adv = 0.05,
            alpha_spec = 0.01,
            gamma = 0.1,
            xi = 0.25,
            zeta_hw = 0.01
        )

    # Reconstrucción
    recon_loss = F.mse_loss(x_pred, x)

    # KL adaptativo con PID
    kl = kl_divergence(z_post, z_prior).sum(dim=1).mean()
    beta, e_t, e_sum = beta_pid_controller(kl.item(), d_target, e_prev, e_sum)

    # Adversarial: se puede simular con ruido
    adv_loss = torch.rand_like(recon_loss) * 0.01

    # log|Σ| ~ logdet regularization (HK spectral term)
    spec_reg = torch.logdet(cov_matrix + 1e-5 * torch.eye(cov_matrix.size(-1)).to(cov_matrix.device))

    # Gradient penalty
    grad_penalty = grad_norm / (weight_norm + 1e-6)

    # Costo computacional normalizado
    if isinstance(act_mem_gb, torch.Tensor):
        act_mem_gb = act_mem_gb.item()
    hw_cost = (act_mem_gb / 8.0) + 0.1 * (flops / 1e12)

    total = (recon_loss +
             beta * kl +
             params["lambda_adv"] * adv_loss -
             params["alpha_spec"] * spec_reg +
             params["gamma"] * grad_penalty -
             params["xi"] * efe_1T +
             params["zeta_hw"] * hw_cost)

    return total.mean(), {
        "recon": recon_loss.mean().item(),
        "kl": kl.item(),
        "beta": beta,
        "efe": efe_1T.item(),
        "spec_reg": spec_reg.mean().item(),
        "grad_penalty": grad_penalty.mean().item(),
        "hw_cost": hw_cost
    }, e_t, e_sum

__all__ = ["loss_elite_v1_2"]
