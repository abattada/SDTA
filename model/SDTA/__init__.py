"""SDTA: Shifted-Window Diffusion with Temporal-Distance Attention.

The proposed method of the paper. Self-supervised pretraining (paper: Method)
encodes one clean anchor window once and denoises W diffusion-corrupted future
windows, each shifted a distance s in patch units (drawn independently at
random from [1, s_max]; the first window is pinned to s=1), through a
causal-masked cross-attention decoder; temporal-distance attention injects a
continuous distance code on the attention key side only. Downstream forecasting
fine-tunes only the patch embedding + encoder with a linear flatten head.

Stages: python -m model.SDTA.pretrain / .train / .test
"""
