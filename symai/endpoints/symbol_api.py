from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union

from symbol import Symbol, Expression

app = FastAPI(title="SymbolicAI API", version="1.0")


class CreateSymbolRequest(BaseModel):
    value: Optional[Any] = None
    static_context: Optional[str] = ''


class CreateExpressionRequest(BaseModel):
    value: Optional[Any] = None


@app.get("/")
async def root():
    return {"message": "Hello World"}


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
    # This would retrieve a symbol by its ID from storage; however, for this example, it's not implemented.
    pass


@app.get("/expression/{expression_id}/")
def get_expression(expression_id: int):
    # This would retrieve an expression by its ID from storage; however, for this example, it's not implemented.
    pass


class SymbolMethodRequest(BaseModel):
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}


@app.post("/symbol/{symbol_id}/call/")
def call_symbol(symbol_id: int, method_request: SymbolMethodRequest):
    # This endpoint should retrieve the symbol instance first. This example assumes we have the symbol object.
    symbol = Symbol()  # placeholder for actual retrieval logic
    result = symbol(*method_request.args, **method_request.kwargs)
    return {"result": result}


@app.post("/symbol/{symbol_id}/{method_name}/")
def operate_on_symbol(symbol_id: int, method_name: str, method_request: SymbolMethodRequest):
    # This endpoint should retrieve the symbol instance first. This example assumes we have the symbol object.
    symbol = Symbol()  # placeholder for actual retrieval logic
    method = getattr(symbol, method_name, None)
    if method is None or not callable(method):
        return {"error": f"Method {method_name} not found or is not callable"}
    result = method(*method_request.args, **method_request.kwargs)
    return {"result": result.__dict__ if isinstance(result, Symbol) else result}


@app.post("/expression/{expression_id}/{method_name}/")
def operate_on_expression(expression_id: int, method_name: str, method_request: SymbolMethodRequest):
    # This endpoint should retrieve the expression instance first. This example assumes we have the expression object.
    expression = Expression()  # placeholder for actual retrieval logic
    method = getattr(expression, method_name, None)
    if method is None or not callable(method):
        return {"error": f"Method {method_name} not found or is not callable"}
    result = method(*method_request.args, **method_request.kwargs)
    return {"result": result.__dict__ if isinstance(result, Symbol) else result}