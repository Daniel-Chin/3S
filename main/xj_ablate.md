|   | xj | dan |
|---|----|-----|
| rnn_width | 256 | 16 |
| vae_channels | [64, 128, 256] | [16, 32, 64] |
| relu_leak | f | t |
| lossWeight['self_recon'] | 655360 | 1 |
| lossWeight['kld'] | 0.32 | 1e-5 |
| lossWeight['predict']['z'] | 3840 | 0 |
| lossWeight['predict']['image'] | 1310720 | 1 |
| K | 2 | 1 |
| symm | T \| R | T $\circ$ R |
| lr_sched | 100% -> 35% over 4687 epochs | off |
| n_epochs | 4687 | 40000 |
| grad_clip | off | 1 |
| rnn_min_context | 5 | 4 |
| sched_sampling | sigmoid($\alpha=2200, \beta=8000$) | linear(1 -> 0 over 40000) |
| residual | off | on |
| image_loss | bce | mse |
| deep_spread | on | off |
| dataset | xj | dan |
