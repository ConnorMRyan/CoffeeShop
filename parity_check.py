import torch
import numpy as np
from einops import rearrange, reduce

def parity_check():
    print("Running Parity Check for einops refactors...")

    # 1. models/common.py: Reshape
    # Original: x.view(-1, C, H, W)
    C, H, W = 3, 64, 64
    batch = 8
    x = torch.randn(batch, C * H * W)
    
    old_op_1 = x.view(-1, C, H, W)
    new_op_1 = rearrange(x, 'b (c h w) -> b c h w', c=C, h=H, w=W)
    
    assert torch.allclose(old_op_1, new_op_1), "Reshape parity failed"
    print("✅ models/common.py Reshape: PASSED")

    # 2. coffeeshop/train.py: td_errors aggregation
    # Original: td_errors.view(num_timesteps, num_actors).max(dim=1)
    num_timesteps = 10
    num_actors = 4
    td_errors = torch.randn(num_timesteps * num_actors)
    
    old_res_2, old_idx_2 = td_errors.view(num_timesteps, num_actors).max(dim=1)
    new_res_2 = reduce(td_errors, '(t a) -> t', 'max', a=num_actors)
    new_idx_2 = rearrange(td_errors, '(t a) -> t a', a=num_actors).argmax(dim=1)
    
    assert torch.allclose(old_res_2, new_res_2), "coffeeshop/train.py reduce parity failed"
    assert torch.all(old_idx_2 == new_idx_2), "coffeeshop/train.py argmax parity failed"
    print("✅ coffeeshop/train.py aggregation: PASSED")

    # 3. scripts/train.py: per_step_td
    # Original: td_errors.view(-1, num_agents).max(dim=1).values
    num_timesteps_3 = 12
    num_agents = 2
    td_errors_3 = torch.randn(num_timesteps_3 * num_agents)
    
    old_res_3 = td_errors_3.view(-1, num_agents).max(dim=1).values
    new_res_3 = reduce(td_errors_3, '(t a) -> t', 'max', a=num_agents)
    
    assert torch.allclose(old_res_3, new_res_3), "scripts/train.py reduce parity failed"
    print("✅ scripts/train.py aggregation: PASSED")

    print("\nAll einops parity checks PASSED! 🚀")

if __name__ == "__main__":
    parity_check()
