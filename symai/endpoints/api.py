import importlib
from fastapi import FastAPI, APIRouter, Body
from pydantic import BaseModel, create_model
from typing import Any, Dict, List, Optional, Union, Type
from .. import Symbol, Expression
from ..components import Translate, Compose


app = FastAPI(title="SymbolicAI API", version="1.0")


class CreateSymbolRequest(BaseModel):
    value: Optional[Any] = None
    static_context: Optional[str] = ''


class CreateExpressionRequest(BaseModel):
    value: Optional[Any] = None


@app.get("/")
async def root():
    return {"message": "Hello Symbols"}


@app.post("/symbol/")
def create_symbol(symbol_request: CreateSymbolRequest):
    symbol = Symbol(symbol_request.value, static_context=symbol_request.static_context)
    return symbol.__dict__


@app.post("/expression/")
def create_expression(expression_request: CreateExpressionRequest):
    expression = Expression(expression_request.value)
    return expression.__dict__


@app.get("/symbol/{symbol_id}/")
def get_symbol(symbol_id: int):
    pass  # Implement retrieval logic


@app.get("/expression/{expression_id}/")
def get_expression(expression_id: int):
    pass  # Implement retrieval logic


class SymbolMethodRequest(BaseModel):
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}


@app.post("/symbol/{symbol_id}/call/")
def call_symbol(symbol_id: int, method_request: SymbolMethodRequest):
    symbol = Symbol()  # Replace with retrieval logic
    result = symbol(*method_request.args, **method_request.kwargs)
    return {"result": result}


@app.post("/symbol/{symbol_id}/{method_name}/")
def operate_on_symbol(symbol_id: int, method_name: str, method_request: SymbolMethodRequest):
    symbol = Symbol()  # Replace with retrieval logic
    method = getattr(symbol, method_name, None)
    if method is None or not callable(method):
        return {"error": f"Method {method_name} not found or is not callable"}
    result = method(*method_request.args, **method_request.kwargs)
    return {"result": result.__dict__ if isinstance(result, Symbol) else result}


@app.post("/expression/{expression_id}/{method_name}/")
def operate_on_expression(expression_id: int, method_name: str, method_request: SymbolMethodRequest):
    expression = Expression()  # Replace with retrieval logic
    method = getattr(expression, method_name, None)
    if method is None or not callable(method):
        return {"error": f"Method {method_name} not found or is not callable"}
    result = method(*method_request.args, **method_request.kwargs)
    return {"result": result.__dict__ if isinstance(result, Symbol) else result}


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


router = APIRouter()


# Modify the existing GenericRequest
class GenericRequest(BaseModel):
    class_name: str
    init_args: List[Any]            = {}
    init_kwargs: Dict[str, Any]     = {}
    forward_args: List[ Any]        = {}
    forward_kwargs: Dict[str, Any]  = {}


@app.post("/components/")
async def generic_forward(request: GenericRequest):
    # Dynamically import the class from components module based on request.class_name
    components_module = importlib.import_module('.components', package='symai')
    cls = getattr(components_module, request.class_name)

    # Check if cls is subclass of Expression and instantiate
    if not issubclass(cls, components_module.Expression):
        raise ValueError("The provided class name must be a subclass of Expression")

    # Initialize the class with provided init_args, requiring unpacking **kwargs
    instance = cls(*request.init_args, **request.init_kwargs)

    # Call the forward method with provided forward_args, requiring unpacking **kwargs
    result = instance.forward(*request.forward_args, **request.forward_kwargs)

    # Assume result is a components.Symbol instance, you need to return a serializable format
    if isinstance(result, components_module.Symbol):
        return result.__dict__
    else:
        return result  # If result is already a serializable type


app.include_router(router)