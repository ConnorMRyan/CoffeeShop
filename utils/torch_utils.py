import torch

def select_action(ac, obs_tensor: torch.Tensor, deterministic: bool = True) -> int:
    """Extract a discrete action from the ActorCriticNet."""
    if obs_tensor.dim() == 1:
        obs_tensor = obs_tensor.unsqueeze(0)

    with torch.no_grad():
        dist, _ = ac(obs_tensor)  # ActorCritic returns (Categorical, value)

    if deterministic:
        return int(dist.probs.argmax(dim=-1).item())
    return int(dist.sample().item())
