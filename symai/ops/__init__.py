from . import primitives as _primitives

__all__ = getattr(_primitives, "__all__", None) # noqa
if __all__ is None:
    __all__ = [name for name in dir(_primitives) if not name.startswith("_")]

for _name in __all__:
    globals()[_name] = getattr(_primitives, _name)

SYMBOL_PRIMITIVES = [
    _primitives.OperatorPrimitives,
    _primitives.IterationPrimitives,
    _primitives.ValueHandlingPrimitives,
    _primitives.StringHelperPrimitives,
    _primitives.CastingPrimitives,
    _primitives.ComparisonPrimitives,
    _primitives.ExpressionHandlingPrimitives,
    _primitives.DataHandlingPrimitives,
    _primitives.UniquenessPrimitives,
    _primitives.PatternMatchingPrimitives,
    _primitives.DictHandlingPrimitives,
    _primitives.QueryHandlingPrimitives,
    _primitives.ExecutionControlPrimitives,
    _primitives.TemplateStylingPrimitives,
    _primitives.DataClusteringPrimitives,
    _primitives.EmbeddingPrimitives,
    _primitives.IndexingPrimitives,
    _primitives.IOHandlingPrimitives,
    _primitives.PersistencePrimitives,
    _primitives.OutputHandlingPrimitives,
    _primitives.FineTuningPrimitives,
]

del _name
del _primitives
