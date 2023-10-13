# AbacusLagrangianBias
codes used in my two papers analyzing the AbacusSummit sims:

paper I: https://arxiv.org/abs/2109.13948 non-parametric Lagrangian bias model based on minimizing $\chi^2$-like statistics and solving matrix equations.
See `AbacusSummit_base_c000_ph006/usage.txt` and `AbacusSummit_base_c000_ph006/matrixA/usage_continued.txt` for how to run things.

paper II: https://arxiv.org/abs/2212.08095 an extension of paper I using neural nets to predict Lagrangian biasing given particle properties.
`run_NN_NNB.py` is the most important one.  Need to use the particle features calculated in paper I though, so run the things listed in `AbacusSummit_base_c000_ph006/usage.txt`.
