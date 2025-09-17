import contextvars

# Holds the current neurosymbolic engine instance for the active context (Task/thread)
CURRENT_ENGINE_VAR = contextvars.ContextVar("symai_current_engine", default=None)
