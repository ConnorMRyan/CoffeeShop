import pytest
import torch
from core_marl.mediator import CoffeeShopMediator
from core_marl.memory import ScoredMemory, Transition

def test_mediator_trust_bounds():
    mediator = CoffeeShopMediator(global_obs_dim=8)
    
    # Trust should decay with high loss, approach 1.0 with low loss
    trust_high = mediator.get_verifiable_trust(td_error=0.001)
    trust_mid = mediator.get_verifiable_trust(td_error=1.0)
    trust_low = mediator.get_verifiable_trust(td_error=10.0)
    
    assert 0.0 <= trust_high <= 1.0
    assert 0.0 <= trust_mid <= 1.0
    assert 0.0 <= trust_low <= 1.0
    
    assert trust_high > trust_mid > trust_low
