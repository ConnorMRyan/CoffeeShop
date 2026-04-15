import torch
from torch.distributions import Categorical

def select_action(ac, obs_tensor: torch.Tensor, deterministic: bool = True) -> int:
    """Extract a discrete action from the ActorCriticNet."""
    if obs_tensor.dim() == 1:
        obs_tensor = obs_tensor.unsqueeze(0)

    with torch.no_grad():
        logits, _ = ac(obs_tensor)   # ActorCriticNet always returns (logits, value)

    if deterministic:
        return int(torch.argmax(logits, dim=-1).item())
    return int(Categorical(logits=logits).sample().item())
