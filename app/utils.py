#app/utils.py
import math

def sanitize_number(x, max_value=1000.0):
    """Limpa floats inválidos (NaN, inf) para garantir JSON válido."""
    if x is None:
        return None

    if isinstance(x, (int, float)):
        if math.isnan(x) or math.isinf(x):
            if x > 0:
                return max_value
            else:
                return 0.0
        return float(x)

    return x


def sanitize_series(series, max_value=1000.0):
    """Limpa uma lista de floats."""
    if series is None:
        return None

    cleaned = []
    for v in series:
        cleaned.append(sanitize_number(v, max_value))
    return cleaned


def sanitize_dict(d, max_value=1000.0):
    """Sanitiza recursivamente qualquer dicionário retornado pela pipeline."""
    if d is None:
        return None

    out = {}

    for key, value in d.items():
        if isinstance(value, dict):
            out[key] = sanitize_dict(value, max_value)
        elif isinstance(value, list):
            out[key] = sanitize_series(value, max_value)
        else:
            out[key] = sanitize_number(value, max_value)

    return out
