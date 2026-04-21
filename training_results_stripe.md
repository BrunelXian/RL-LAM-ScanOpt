# Training Results

## Model Performance

| Planner | Coverage | Thermal Mean | Thermal Peak | Thermal Variance | Steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| raster | 1.000 | 0.006 | 0.555 | 0.001 | 890 |
| random | 1.000 | 0.006 | 0.292 | 0.000 | 890 |
| greedy_cool_first | 1.000 | 0.006 | 0.326 | 0.000 | 890 |
| rl_maskable_ppo | 1.000 | 0.006 | 0.512 | 0.001 | 890 |

## Graphical Comparison

### Scan Paths
![Scan Order Comparison](assets/figures/order_map_comparison_grid_stripe.png)

### Thermal Maps
![Thermal Map Comparison](assets/figures/thermal_map_comparison_grid_stripe.png)

### RL Scan Animation
![RL Scan Path GIF](assets/figures/scan_path_rl_maskable_ppo_stripe.gif)

## Training Curves

![Training Curves](assets/figures/maskable_ppo_twi_stripe_training_curves.png)

## Discussion

The current Maskable PPO model reaches full coverage on the `TWI` letter mask and is compared against raster, random, and greedy cool-first baselines using the same lightweight thermal proxy. The updated stripe-based environment chooses the next legal vertical stripe inside the letter mask, then applies sequential thermal updates over the cells in that stripe. The reward mixes coverage, thermal variance penalties, hotspot penalties, temperature-difference bonuses, and invalid-action penalties.

Current limitations:
- This is still a proxy thermal environment rather than a calibrated physical model.
- Training uses one fixed geometry and one environment, so generalisation is limited.
- Even with longer training, RL may still trail the strongest handcrafted baselines on this simplified task.

### Training Snapshot

- Episodes recorded: 1792
- Final recorded coverage: 1.000
- Final recorded thermal variance: 0.001
- Final recorded thermal peak: 0.517
