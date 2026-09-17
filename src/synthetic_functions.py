"""Branin and Park fidelity pairs."""
import numpy as np

def branin_hf(x: np.ndarray) -> np.ndarray:
    x1 = x[:, 0] * 15 - 5
    x2 = x[:, 1] * 15
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    result = (x2 - b * x1 ** 2 + c * x1 - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return result.reshape(-1, 1)

def branin_lf(x: np.ndarray, alpha: float=0.6) -> np.ndarray:
    x1 = x[:, 0] * 15 - 5
    x2 = x[:, 1] * 15
    b_lf = 5.1 / (4 * np.pi ** 2) - 0.1 * (1 - alpha)
    c = 5 / np.pi
    result = (x2 - b_lf * x1 ** 2 + c * x1 - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return result.reshape(-1, 1)

def park_hf(x: np.ndarray) -> np.ndarray:
    eps = 1e-08
    x1 = np.maximum(x[:, 0], eps)
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = np.maximum(x[:, 3], eps)
    inner = (x2 + x3 ** 2) * x4 / (x1 ** 2 + eps)
    term1 = x1 / 2 * np.sqrt(np.maximum(inner, 0))
    term2 = (x1 + 3 * x4) * np.exp(1 + np.sin(x3))
    term1 = np.nan_to_num(term1, nan=0.0, posinf=10.0, neginf=-10.0)
    term2 = np.nan_to_num(term2, nan=0.0, posinf=100.0, neginf=-100.0)
    result = term1 + term2
    return result.reshape(-1, 1)

def park_lf(x: np.ndarray, alpha: float=0.6) -> np.ndarray:
    eps = 1e-08
    x1 = np.maximum(x[:, 0], eps)
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = np.maximum(x[:, 3], eps)
    x4_coef = 3 - 1.5 * (1 - alpha)
    inner = (x2 + x3 ** 2) * x4 / (x1 ** 2 + eps)
    term1 = x1 / 2 * np.sqrt(np.maximum(inner, 0))
    term2 = (x1 + x4_coef * x4) * np.exp(1 + alpha * np.sin(x3))
    term1 = np.nan_to_num(term1, nan=0.0, posinf=10.0, neginf=-10.0)
    term2 = np.nan_to_num(term2, nan=0.0, posinf=100.0, neginf=-100.0)
    result = term1 + term2
    return result.reshape(-1, 1)
