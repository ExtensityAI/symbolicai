from typing import Any, Dict, List, Type, Union
from fastapi import APIRouter, Body
from pydantic import BaseModel, create_model
from components import Symbol, Expression

router = APIRouter()

class BaseSymbolModel(BaseModel):
    value: Any
    static_context: str

class BaseExpressionModel(BaseModel):
    value: Any

def create_symbol_model(symbol_cls: Type[Symbol]) -> Type[BaseModel]:
    fields = {
        "value": (Any, ...),
        "static_context": (str, None)
    }
    symbol_model = create_model(f'{symbol_cls.__name__}Model', **fields, __base__=BaseSymbolModel)
    return symbol_model

def create_expression_model(expression_cls: Type[Expression]) -> Type[BaseModel]:
    fields = {
        "value": (Any, ...)
    }
    expression_model = create_model(f'{expression_cls.__name__}Model', **fields, __base__=BaseExpressionModel)
    return expression_model

def create_class_endpoint(cls: Union[Type[Symbol], Type[Expression]], cls_model: BaseModel):
    @router.post(f"/{cls.__name__.lower()}/", response_model=Dict[str, Any])
    def create_instance(data: cls_model = Body(...)):
        instance = cls(**data.dict())
        return instance.__dict__

    methods = [func for func in dir(cls) if callable(getattr(cls, func)) and not func.startswith("_")]

    for method_name in methods:
        @router.post(f"/{cls.__name__.lower()}/{method_name}/", response_model=Dict[str, Any])
        def method(data: Dict[str, Any] = Body(...), name: str = method_name):
            instance = cls()  # To replace with actual instance retrieval
            method = getattr(instance, name, None)
            if method:
                result = method(**data)
                return {"result": result.__dict__ if isinstance(result, (Symbol, Expression)) else result}
            return {"error": f"Method {name} not found or is not callable."}

# Automatically generate models and endpoints for each Symbol and Expression subclass
subclasses_symbol = Symbol.__subclasses__() + [Symbol]
subclasses_expression = Expression.__subclasses__() + [Expression]

for subclass in subclasses_symbol:
    symbol_model = create_symbol_model(subclass)
    create_class_endpoint(subclass, symbol_model)

for subclass in subclasses_expression:
    expression_model = create_expression_model(subclass)
    create_class_endpoint(subclass, expression_model)