TABPFN_MAX_SAMPLES = int(1e4)
TABPFN_MAX_FEATURES = 500
TABPFN_MAX_CLASSES = 10


def get_dynamic_inference_limit(nof_samples: int, nof_features: int, nof_classes: int) -> int:
    limit = TABPFN_MAX_SAMPLES
    if nof_samples > 0.8 * TABPFN_MAX_SAMPLES:
        limit = int(limit * 0.8)

    if nof_features > 0.8 * TABPFN_MAX_FEATURES:
        limit = int(limit * 0.8)

    if nof_classes > 0.8 * TABPFN_MAX_CLASSES:
        limit = int(limit * 0.8)
    return limit
