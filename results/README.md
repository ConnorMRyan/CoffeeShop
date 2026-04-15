# CoffeeShop — Experiment Results

All runs use the configs in `configs/experiments/`. Metrics are episode reward
(mean over last 100 episodes) unless otherwise noted.

## Crafter

| Run | Config | Seed | Steps | Mean Reward | Notes |
|-----|--------|------|-------|-------------|-------|
| VanillaCrafter | `baseline_vanilla_ppo_crafter.yaml` | 0 | 1M | TODO | Baseline |
| CoffeeShop_Crafter | `coffeeshop_crafter_comparison.yaml` | 0 | 1M | TODO | Social enabled |

## Overcooked — cramped_room

| Run | Config | Seed | Steps | Mean Reward | Notes |
|-----|--------|------|-------|-------------|-------|
| CoffeeShop_Overcooked_Cramped | `coffeeshop_overcooked_cramped.yaml` | 42 | 2M | TODO | Social enabled |

## Notes

- All runs on Connor-PC (local GPU)
- TensorBoard logs in `checkpoints/{run_name}/tb/`
- To view: `tensorboard --logdir checkpoints/`
