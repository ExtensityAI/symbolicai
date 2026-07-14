from symai.ops import primitives

SYMBOL_PRIMITIVES = [
    primitives.OperatorPrimitives,
    primitives.IterationPrimitives,
    primitives.ValueHandlingPrimitives,
    primitives.StringHelperPrimitives,
    primitives.CastingPrimitives,
    primitives.ComparisonPrimitives,
    primitives.ExpressionHandlingPrimitives,
    primitives.DataHandlingPrimitives,
    primitives.PatternMatchingPrimitives,
    primitives.QueryHandlingPrimitives,
    primitives.TemplateStylingPrimitives,
    primitives.EmbeddingPrimitives,
    primitives.PersistencePrimitives,
]

__all__ = ["SYMBOL_PRIMITIVES", "primitives"]
