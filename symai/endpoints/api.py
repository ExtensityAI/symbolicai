import importlib
import redis
from redis.exceptions import ConnectionError
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Generic, TypeVar, Dict, List, Optional, Type, Union
from .. import Symbol, Expression


# Configure Redis server connection parameters and executable path
HOST = 'localhost'
PORT = 6379


def is_redis_running(host: str, port: int) -> bool:
    """Check if a Redis server is running at the given host and port."""
    try:
        r = redis.Redis(host=host, port=port)
        r.ping()
        print(f"Redis server is running at {host}:{port}")
        return True
    except ConnectionError:
        print(f"Redis server is not running at {host}:{port} or is not reachable - falling back to in-memory storage")
        return False


# Create a generic type variable
T = TypeVar('T')


class GenericRepository(Generic[T]):
    def __init__(self, redis_client: redis.Redis, id_key: str, use_redis: bool = True):
        self.storage: Dict[int, T] = {}  # In-memory dictionary to mock Redis
        self.use_redis     = use_redis
        self.id_key        = id_key  # Key used for storing the incremental ID counter
        self.redis_client  = redis_client
        if self.use_redis:
            self.set       = self._store_redis
            self.get       = self._retrieve_redis
            self.delete    = self._delete_redis
            self.uid       = self._generate_id_redis
        else:
            self.set       = self._store_memory
            self.get       = self._retrieve_memory
            self.delete    = self._delete_memory
            self.uid       = self._generate_id_memory

    def _store_memory(self, item_id: str, item: T) -> None:
        self.storage[item_id] = item

    def _retrieve_memory(self, item_id: str) -> T:
        return self.storage.get(item_id)

    def _delete_memory(self, item_id: str) -> bool:
        if item_id in self.storage:
            del self.storage[item_id]
            return True
        return False

    def _generate_id_memory(self) -> int:
        return len(self.storage) + 1

    def _store_redis(self, item_id: str, item: T) -> None:
        self.redis_client.set(item_id, str(item))

    def _retrieve_redis(self, item_id: str) -> T:
        item = self.redis_client.get(item_id)
        return item.decode('utf-8') if item else None

    def _delete_redis(self, item_id: str) -> bool:
        return self.redis_client.delete(item_id) == 1

    def _generate_id_redis(self) -> int:
        return self.redis_client.incr(self.id_key)


# Initialize the Redis client
redis_client = redis.Redis(
    host=HOST,   # Or use the host where your Redis server is running
    port=PORT,   # The default Redis port number
    db=0         # The default database index
)

# Initialize the FastAPI app and API router
app                           = FastAPI(title="SymbolicAI API", version="1.0")
router                        = APIRouter()
use_redis                     = is_redis_running(HOST, PORT)

# Instantiate the generic repositories with the counter keys
symbol_repository             = GenericRepository[Symbol](redis_client, "symbol_id_counter", use_redis=use_redis)
expression_repository         = GenericRepository[Expression](redis_client, "expression_id_counter", use_redis=use_redis)
component_class_repository    = GenericRepository[Type[Union[Symbol, Expression]]](redis_client, "component_class_id_counter", use_redis=use_redis)
component_instance_repository = GenericRepository[Union[Symbol, Expression]](redis_client, "component_instance_id_counter", use_redis=use_redis)


### Symbols Endpoints ###


class CreateSymbolRequest(BaseModel):
    value: Optional[Any]          = None
    static_context: Optional[str] = ''


class UpdateSymbolRequest(BaseModel):
    value: Any = None
    static_context: str = None


class SymbolMethodRequest(BaseModel):
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}


@app.post("/symbol/")
def create_symbol(symbol_request: CreateSymbolRequest):
    symbol = Symbol(symbol_request.value, static_context=symbol_request.static_context)
    symbol_id = symbol_repository.uid()
    symbol_repository.set(symbol_id, symbol)
    return {"id": symbol_id, **symbol.__dict__}


@app.get("/symbol/{symbol_id}/")
def get_symbol(symbol_id: int):
    symbol = symbol_repository.get(symbol_id)
    if symbol is None:
        raise HTTPException(status_code=404, detail="Symbol not found")
    return symbol.__dict__


@app.patch("/symbol/{symbol_id}/")
def update_symbol(symbol_id: int, update_request: UpdateSymbolRequest):
    symbol = symbol_repository.get(symbol_id)
    if symbol is None:
        raise HTTPException(status_code=404, detail="Symbol not found")
    for attr, value in update_request.model_dump(exclude_unset=True).items():
        setattr(symbol, attr, value)
    symbol_repository.set(symbol_id, symbol)
    return symbol.__dict__


@app.delete("/symbol/{symbol_id}/")
def delete_symbol(symbol_id: int):
    symbol = symbol_repository.delete(symbol_id)
    if symbol is None:
        raise HTTPException(status_code=404, detail="Symbol not found")
    return {"message": "Symbol deleted successfully"}


@app.post("/symbol/{symbol_id}/call/")
def call_symbol(symbol_id: int, method_request: SymbolMethodRequest):
    # Retrieve the symbol instance by ID
    symbol = symbol_repository.get(symbol_id)
    if symbol is None:
        raise HTTPException(status_code=404, detail="Symbol not found")
    result = symbol(*method_request.args, **method_request.kwargs)
    return {"result": result}


@app.post("/symbol/{symbol_id}/{method_name}/")
def operate_on_symbol(symbol_id: int, method_name: str, method_request: SymbolMethodRequest):
    symbol = symbol_repository.get(symbol_id)
    if symbol is None:
        raise HTTPException(status_code=404, detail="Symbol not found")
    method = getattr(symbol, method_name, None)
    if method is None or not callable(method):
        raise HTTPException(status_code=404, detail=f"Method {method_name} not found or is not callable")
    result = method(*method_request.args, **method_request.kwargs)
    return {"result": result.__dict__ if isinstance(result, Symbol) else result}


### Expressions Endpoints ###


class CreateExpressionRequest(BaseModel):
    value: Optional[Any] = None


class UpdateExpressionRequest(BaseModel):
    value: Any = None


@app.post("/expression/")
def create_expression(expression_request: CreateExpressionRequest):
    expression = Expression(expression_request.value)
    expression_id = expression_repository.uid()
    expression_repository.set(expression_id, expression)
    return {"id": expression_id, **expression.__dict__}


@app.get("/expression/{expression_id}/")
def get_expression(expression_id: int):
    expression = expression_repository.get(expression_id)
    if expression is None:
        raise HTTPException(status_code=404, detail="Expression not found")
    return expression.__dict__


@app.post("/expression/{expression_id}/call/")
def call_expression(expression_id: int, method_request: SymbolMethodRequest):
    # Retrieve the expression instance by ID
    expression = expression_repository.get(expression_id)
    if expression is None:
        raise HTTPException(status_code=404, detail="Expression not found")
    result = expression(*method_request.args, **method_request.kwargs)
    return {"result": result}


@app.post("/expression/{expression_id}/{method_name}/")
def operate_on_expression(expression_id: int, method_name: str, method_request: SymbolMethodRequest):
    expression = expression_repository.get(expression_id)
    if expression is None:
        raise HTTPException(status_code=404, detail="Expression not found")
    method = getattr(expression, method_name, None)
    if method is None or not callable(method):
        raise HTTPException(status_code=404, detail=f"Method {method_name} not found or is not callable")
    result = method(*method_request.args, **method_request.kwargs)
    return {"result": result.__dict__ if isinstance(result, Expression) else result}


@app.patch("/expression/{expression_id}/")
def update_expression(expression_id: int, update_request: UpdateExpressionRequest):
    expression = expression_repository.get(expression_id)
    if expression is None:
        raise HTTPException(status_code=404, detail="Expression not found")
    for attr, value in update_request.model_dump(exclude_unset=True).items():
        setattr(expression, attr, value)
    expression_repository.set(expression_id, expression)
    return expression.__dict__


@app.delete("/expression/{expression_id}/")
def delete_expression(expression_id: int):
    expression = expression_repository.delete(expression_id)
    if expression is None:
        raise HTTPException(status_code=404, detail="Expression not found")
    return {"message": "Expression deleted successfully"}


#### Generic Component Endpoints ####


# Endpoint to add a component class to the repository dynamically
class AddComponentRequest(BaseModel):
    module_name: str
    class_name: str


# Endpoint to instantiate a generic component class and get the ID
class CreateComponentGenericRequest(BaseModel):
    class_name: str
    init_args: Dict[str, Any] = {}
    exec_args: Dict[str, Any] = {}


# Modify the existing GenericRequest
class GenericRequest(BaseModel):
    class_name: str                 = ''
    init_args: List[Any]            = {}
    init_kwargs: Dict[str, Any]     = {}
    forward_args: List[ Any]        = {}
    forward_kwargs: Dict[str, Any]  = {}


class UpdateComponentRequest(BaseModel):
    init_args: Dict[str, Any] = {}
    exec_args: Dict[str, Any] = {}


@app.post("/components/add/")
async def add_component(request: AddComponentRequest):
    # Dynamically import the class from components module based on request.module_name and request.class_name
    try:
        module = importlib.import_module(f'..{request.module_name}', package='symai')
        cls = getattr(module, request.class_name)

        # Check if cls is subclass of Symbol or Expression and add to repository
        if not issubclass(cls, (Symbol, Expression)) :
            raise ValueError("The provided class must be a subclass of Symbol or Expression")

        component_class_repository.set(request.class_name, cls)
        return {"message": f"{request.class_name} has been added to the repository."}
    except (ImportError, AttributeError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/components/create/")
async def create_component(request: CreateComponentGenericRequest):
    # Retrieve the class from the repository
    cls = component_class_repository.get(request.class_name)
    if cls is None:
        raise HTTPException(status_code=404, detail="Component class not found")

    # Instantiate the class and execute the command
    instance = cls(**request.init_args)

    # Store the instance with a generated ID and return the ID
    instance_id = component_instance_repository.uid()
    component_instance_repository.set(instance_id, instance)  # Assuming component_instance_repository exists

    return {"id": instance_id}


@app.get("/components/{instance_id}/")
def get_component(instance_id: int):
    # Retrieve an instance by its ID from the repository
    instance = component_instance_repository.get(instance_id)
    if instance is None:
        raise HTTPException(status_code=404, detail="Component instance not found")

    return instance.__dict__  # Assuming __dict__ can be used to serialize the instance


# Endpoint to execute a command on a component instance
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


@app.patch("/components/{instance_id}/")
def update_component(instance_id: int, update_request: UpdateComponentRequest):
    instance = component_instance_repository.get(instance_id)
    if instance is None:
        raise HTTPException(status_code=404, detail="Component instance not found")
    instance.__init__(**update_request.init_args)
    if update_request.exec_args:
        instance.forward(**update_request.exec_args)
    component_instance_repository.set(instance_id, instance)
    return instance.__dict__


@app.delete("/components/{instance_id}/")
def delete_component(instance_id: int):
    instance = component_instance_repository.delete(instance_id)
    if instance is None:
        raise HTTPException(status_code=404, detail="Component instance not found")
    return {"message": "Component instance deleted successfully"}


app.include_router(router)
