import numpy as np

# Fix de compatibilidad para pickles antiguos que requieren numpy._core
if not hasattr(np, "_core"):
    np._core = np.core