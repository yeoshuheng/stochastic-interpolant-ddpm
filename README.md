### Notes

Stochastic interpolants allow for non-gaussian sampling by acting as a bridge between the source and target distributions.

Interpolants can be done in a deterministic form using the Transport ODE or in a stochastic form via Fokker-Planck SDEs.

In the deterministic form, we only need to learn the drift of the model, which captures the velocity and score of the model.

In the stochastic form, while drift captures velocity, we need another model to capture score as both drift and score are components for our forward and backward Fokker-Planck drifts.

### Papers
A.S. Albergo, N. M. Boffi, E. Vanden-Eijnden, [Stochastic Interpolants: A Unifying Framework for Flows and Diffusions](https://arxiv.org/abs/2303.08797)

