from symai.ops import primitives as _primitives

__all__ = getattr(_primitives, "__all__", None)  # noqa
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
    _primitives.PatternMatchingPrimitives,
    _primitives.QueryHandlingPrimitives,
    _primitives.TemplateStylingPrimitives,
    _primitives.EmbeddingPrimitives,
    _primitives.PersistencePrimitives,
]

del _primitives
