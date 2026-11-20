"""Mesure de latence du forward du modèle seul (warm-up + synchronisation GPU)."""
import time

import torch


def mesurer_latence(model, hw, n_warmup=20, n_iter=100):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.model.to(dev).eval()
    x = torch.randn(1, 3, *hw, device=dev)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    with torch.inference_mode():
        for _ in range(n_warmup):
            net(x)
        sync()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            net(x)
        sync()
    return 1000 * (time.perf_counter() - t0) / n_iter
