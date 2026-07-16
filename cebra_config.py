"""Shared helpers for constructing CEBRA models from a base config plus overrides."""


def merge_cebra_params(defaults, overrides=None):
    """Return CEBRA constructor kwargs from `defaults` with any `overrides` applied.

    Only keys whose value in `overrides` is not None replace the matching default,
    so a caller can tune a subset of hyperparameters (e.g. from the CLI) while every
    unset parameter keeps the decoder's tuned default.
    """
    applied = {key: value for key, value in (overrides or {}).items() if value is not None}
    return {**defaults, **applied}
