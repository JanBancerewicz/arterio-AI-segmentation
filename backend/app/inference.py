"""Exceptions safe to translate into public processing-error codes."""


class ModelLoadError(RuntimeError):
    code = "MODEL_LOAD_FAILED"


class OutOfMemoryError(RuntimeError):
    code = "OUT_OF_MEMORY"
