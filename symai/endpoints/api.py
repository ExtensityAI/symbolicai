import importlib
import inspect
import pickle
import redis

from redis.exceptions import ConnectionError
from fastapi import FastAPI, APIRouter, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Generic, TypeVar, Dict, List, Optional, Union

from symai.backend import settings

from .. import core_ext
from ..symbol import Symbol, Expression


# Configure Redis server connection parameters and executable path
HOST  = 'localhost'
PORT  = 6379
DEBUG = True
API_KEY = settings.SYMAI_CONFIG.get('FASTAPI_API_KEY', None)

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
        self.storage: Dict[str, T] = {}  # In-memory dictionary to mock Redis
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

    def _deserialize_object(self, serialized_item: bytes) -> T:
        """Deserialize the byte object back into a Python object of type T."""
        return pickle.loads(serialized_item)

    def _serialize_object(self, item_id: str, item: T) -> None:
        return pickle.dumps(item)

    def _store_memory(self, item_id: str, item: T) -> None:
        self.storage[item_id] = item

    def _retrieve_memory(self, item_id: str) -> T:
        return self.storage.get(item_id)

    def _delete_memory(self, item_id: str) -> bool:
        if item_id in self.storage:
            del self.storage[item_id]
            return True
        return False

    def _generate_id_memory(self) -> str:
        id_ = len(self.storage) + 1
        return f'{self.id_key}:{id_}'

    def _store_redis(self, item_id: str, item: T) -> None:
        item = self._serialize_object(item_id, item)
        self.redis_client.set(item_id, item)

    def _retrieve_redis(self, item_id: str) -> T:
        item = self.redis_client.get(item_id)
        if item:
            item = self._deserialize_object(item)
            return item
        return None

    def _delete_redis(self, item_id: str) -> bool:
        return self.redis_client.delete(item_id) == 1

    def _generate_id_redis(self) -> str:
        id_ = self.redis_client.incr(self.id_key)
        return f'{self.id_key}:{id_}'


# Initialize the Redis client
redis_client = redis.Redis(
    host=HOST,   # Or use the host where your Redis server is running
    port=PORT,   # The default Redis port number
    db=0         # The default database index
)

# Initialize the FastAPI app and API router
app                           = FastAPI(title="SymbolicAI API", version="1.0")
api_key_header                = APIKeyHeader(name="X-API-Key")
router                        = APIRouter()
use_redis                     = is_redis_running(HOST, PORT)

# Instantiate the generic repositories with the counter keys
symbol_repository             = GenericRepository[Symbol](redis_client, "sym_id", use_redis=use_redis)
expression_repository         = GenericRepository[Expression](redis_client, "expr_id", use_redis=use_redis)
# Register all types which subclass Expression and offer a get() method with default=None
component_class_types = {
    name: cls for name, cls in inspect.getmembers(importlib.import_module('symai.components'), inspect.isclass) if issubclass(cls, Expression)
}
component_instance_repository = GenericRepository[Union[Symbol, Expression]](redis_client, "comp_id", use_redis=use_redis)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

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

def get_api_key(api_key_header: str = Security(api_key_header) if API_KEY else None) -> str:
    if API_KEY == None:
        return True
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )

@app.post("/symbol/")
def create_symbol(symbol_request: CreateSymbolRequest, api_key: str = Security(get_api_key)):
    symbol = Symbol(symbol_request.value, static_context=symbol_request.static_context)
    symbol_id = symbol_repository.uid()
    symbol_repository.set(symbol_id, symbol)
    return {"id": symbol_id, **symbol.json()}


@app.get("/symbol/{symbol_id}/")
def get_symbol(symbol_id: str, api_key: str = Security(get_api_key)):
    symbol = symbol_repository.get(symbol_id)
    if symbol is None:
        raise HTTPException(status_code=404, detail="Symbol not found")
    return symbol.json()


@app.patch("/symbol/{symbol_id}/")
def update_symbol(symbol_id: str, update_request: UpdateSymbolRequest, api_key: str = Security(get_api_key)):
    symbol = symbol_repository.get(symbol_id)
    if symbol is None:
        raise HTTPException(status_code=404, detail="Symbol not found")
    for attr, value in update_request.model_dump(exclude_unset=True).items():
        setattr(symbol, attr, value)
    symbol_repository.set(symbol_id, symbol)
    return symbol.json()


@app.delete("/symbol/{symbol_id}/")
def delete_symbol(symbol_id: str, api_key: str = Security(get_api_key)):
    symbol = symbol_repository.delete(symbol_id)
    if symbol is None:
        raise HTTPException(status_code=404, detail="Symbol not found")
    return {"message": "Symbol deleted successfully"}


@app.post("/symbol/{symbol_id}/{method_name}/")
@core_ext.error_logging(debug=DEBUG)
def operate_on_symbol(symbol_id: str, method_name: str, method_request: SymbolMethodRequest, api_key: str = Security(get_api_key)):
    symbol = symbol_repository.get(symbol_id)
    if symbol is None:
        raise HTTPException(status_code=404, detail="Symbol not found")
    method = getattr(symbol, method_name, None)
    if method is None or not callable(method):
        raise HTTPException(status_code=404, detail=f"Method {method_name} not found or is not callable")
    result = method(*method_request.args, **method_request.kwargs)
    return {"result": result.json() if isinstance(result, Symbol) else result}


### Expressions Endpoints ###


class CreateExpressionRequest(BaseModel):
    value: Optional[Any] = None


class UpdateExpressionRequest(BaseModel):
    value: Any = None


@app.post("/expression/")
def create_expression(expression_request: CreateExpressionRequest, api_key: str = Security(get_api_key)):
    expression = Expression(expression_request.value)
    expression_id = expression_repository.uid()
    expression_repository.set(expression_id, expression)
    return {"id": expression_id, **expression.json()}


@app.get("/expression/{expression_id}/")
def get_expression(expression_id: str, api_key: str = Security(get_api_key)):
    expression = expression_repository.get(expression_id)
    if expression is None:
        raise HTTPException(status_code=404, detail="Expression not found")
    return expression.json()


@app.post("/expression/{expression_id}/call/")
@core_ext.error_logging(debug=DEBUG)
def call_expression(expression_id: str, method_request: SymbolMethodRequest, api_key: str = Security(get_api_key)):
    # Retrieve the expression instance by ID
    expression = expression_repository.get(expression_id)
    if expression is None:
        raise HTTPException(status_code=404, detail="Expression not found")
    result = expression(*method_request.args, **method_request.kwargs)
    return {"result": result.json() if isinstance(result, Symbol) else result}


@app.post("/expression/{expression_id}/{method_name}/")
@core_ext.error_logging(debug=DEBUG)
def operate_on_expression(expression_id: str, method_name: str, method_request: SymbolMethodRequest, api_key: str = Security(get_api_key)):
    expression = expression_repository.get(expression_id)
    if expression is None:
        raise HTTPException(status_code=404, detail="Expression not found")
    method = getattr(expression, method_name, None)
    if method is None or not callable(method):
        raise HTTPException(status_code=404, detail=f"Method {method_name} not found or is not callable")
    result = method(*method_request.args, **method_request.kwargs)
    return {"result": result.json() if isinstance(result, Symbol) else result}


@app.patch("/expression/{expression_id}/")
def update_expression(expression_id: str, update_request: UpdateExpressionRequest, api_key: str = Security(get_api_key)):
    expression = expression_repository.get(expression_id)
    if expression is None:
        raise HTTPException(status_code=404, detail="Expression not found")
    for attr, value in update_request.model_dump(exclude_unset=True).items():
        setattr(expression, attr, value)
    expression_repository.set(expression_id, expression)
    return expression.json()


@app.delete("/expression/{expression_id}/")
def delete_expression(expression_id: str, api_key: str = Security(get_api_key)):
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
    init_args: List[Any]        = []
    init_kwargs: Dict[str, Any] = {}


# Modify the existing GenericRequest
class GenericRequest(BaseModel):
    class_name: str
    init_args: List[Any]            = {}
    init_kwargs: Dict[str, Any]     = {}
    forward_args: List[ Any]        = {}
    forward_kwargs: Dict[str, Any]  = {}


class UpdateComponentRequest(BaseModel):
    update_kwargs: Dict[str, Any] = {}


@app.post("/components/")
def create_component(request: CreateComponentGenericRequest, api_key: str = Security(get_api_key)):
    # Retrieve the class from the repository
    cls = component_class_types.get(request.class_name)
    if cls is None:
        raise HTTPException(status_code=404, detail="Component class not found")
    # Instantiate the class and execute the command
    instance = cls(*request.init_args, **request.init_kwargs)
    # Store the instance with a generated ID and return the ID
    instance_id = component_instance_repository.uid()
    component_instance_repository.set(instance_id, instance)  # Assuming component_instance_repository exists
    return {"id": instance_id}


@app.get("/components/{instance_id}/")
def get_component(instance_id: str, api_key: str = Security(get_api_key)):
    # Retrieve an instance by its ID from the repository
    instance = component_instance_repository.get(instance_id)
    if instance is None:
        raise HTTPException(status_code=404, detail="Component instance not found")
    return instance.json()  # Assuming __dict__ can be used to serialize the instance


# Endpoint to execute a command on a component instance
@app.post("/components/call/")
@core_ext.error_logging(debug=DEBUG)
def generic_forward(request: GenericRequest, api_key: str = Security(get_api_key)):
    # Dynamically import the class from components module based on request.class_name
    components_module = importlib.import_module('.components', package='symai')
    cls = getattr(components_module, request.class_name)
    # Check if cls is subclass of Expression and instantiate
    if not issubclass(cls, components_module.Expression):
        raise ValueError("The provided class name must be a subclass of Expression")
    # Initialize the class with provided init_args, requiring unpacking **kwargs
    instance = cls(*request.init_args, **request.init_kwargs)
    # Call the forward method with provided forward_args, requiring unpacking **kwargs
    result = instance(*request.forward_args, **request.forward_kwargs)
    # Assume result is a components.Symbol instance, you need to return a serializable format
    if isinstance(result, components_module.Symbol):
        return result.json()
    return result  # If result is already a serializable type


@app.patch("/components/{instance_id}/")
def update_component(instance_id: str, update_request: UpdateComponentRequest, api_key: str = Security(get_api_key)):
    instance = component_instance_repository.get(instance_id)
    if instance is None:
        raise HTTPException(status_code=404, detail="Component instance not found")
    for attr, value in update_request.update_kwargs.items():
        setattr(instance, attr, value)
    component_instance_repository.set(instance_id, instance)
    return instance.json()


@app.delete("/components/{instance_id}/")
def delete_component(instance_id: str, api_key: str = Security(get_api_key)):
    instance = component_instance_repository.delete(instance_id)
    if instance is None:
        raise HTTPException(status_code=404, detail="Component instance not found")
    return {"message": "Component instance deleted successfully"}


### Selectively register the endpoints with the API router ###

# extended imports
from ..extended import Conversation
from ..extended.personas import Persona, Dialogue
from ..extended.personas.student import MaxTenner
from ..extended.personas.sales import ErikJames


extended_types = [Conversation, Persona, Dialogue, MaxTenner, ErikJames]
# Register only the extended types from the extended_types Union
extended_class_types         = {c.__name__: c for c in extended_types}
extended_instance_repository = GenericRepository[extended_types](redis_client, "ext_id", use_redis=use_redis)


# Model definitions for extended classes
class CreateExtendedRequest(BaseModel):
    class_name: str
    init_args: List[Any]        = []
    init_kwargs: Dict[str, Any] = {}


class UpdateExtendedRequest(BaseModel):
    update_kwargs: Dict[str, Any] = {}


# Create endpoints for each of the extended classes
@app.post("/extended/")
def create_extended(request: CreateExtendedRequest, api_key: str = Security(get_api_key)):
    # Dynamically retrieve the extended class
    extended_class = extended_class_types.get(request.class_name)
    if extended_class is None:
        raise HTTPException(status_code=404, detail=f"{request.class_name} class not found")
    # Instantiate the extended class with the provided arguments
    extended_instance = extended_class(*request.init_args, **request.init_kwargs)
    # Store the instance and return the instance ID and details
    instance_id = extended_instance_repository.uid()
    extended_instance_repository.set(instance_id, extended_instance)
    return {"id": instance_id, **extended_instance.json()}


# Endpoint to execute a command on a component instance
@app.post("/extended/call/")
@core_ext.error_logging(debug=DEBUG)
def extended_forward(request: GenericRequest, api_key: str = Security(get_api_key)):
    # Dynamically import the class from components module based on request.class_name
    try:
        # get request.class_name from extended_types if it's a type of extended_types
        cls = extended_class_types.get(request.class_name)
        # look for the class in the extended module if not found in extended_types
        # iterate over the extended_type Union and check if the class name is in the __dict__
        if cls is None:
            raise ImportError(f"Class {request.class_name} not found in extended types")
        # Initialize the class with provided init_args and init_kwargs
        instance = cls(*request.init_args, **request.init_kwargs)
        # Call the 'forward' or similar method with provided forward_args and forward_kwargs
        result = instance(*request.forward_args, **request.forward_kwargs)
        # Assume result needs to be serialized for returning as JSON
        # Check if the type of the result is within extended types or a primitive that can be serialized directly
        if isinstance(result, Symbol):
            return result.json()  # Convert to dictionary if it's a complex type
        else:
            return {"result": result}  # Return as is if it's a primitive type
    except ImportError as e:
        raise HTTPException(status_code=404, detail=f"Module not found: {str(e)}")
    except AttributeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/extended/{instance_id}/")
def get_extended(instance_id: str, api_key: str = Security(get_api_key)):
    # Retrieve an instance by its ID
    extended_instance = extended_instance_repository.get(instance_id)
    if extended_instance is None:
        raise HTTPException(status_code=404, detail="Extended instance not found")
    return extended_instance.json()


@app.patch("/extended/{instance_id}/")
def update_extended(instance_id: str, update_request: UpdateExtendedRequest, api_key: str = Security(get_api_key)):
    # Retrieve the instance by its ID
    extended_instance = extended_instance_repository.get(instance_id)
    if extended_instance is None:
        raise HTTPException(status_code=404, detail="Extended instance not found")
    # Update the instance with the provided arguments
    for attr, value in update_request.update_kwargs.items():
        setattr(extended_instance, attr, value)
    extended_instance_repository.set(instance_id, extended_instance)
    return extended_instance.json()


@app.delete("/extended/{instance_id}/")
def delete_extended(instance_id: str, api_key: str = Security(get_api_key)):
    # Attempt to delete the instance by its ID
    success = extended_instance_repository.delete(instance_id)
    if not success:
        raise HTTPException(status_code=404, detail="Extended instance not found or already deleted")
    return {"message": "Extended instance deleted successfully"}


app.include_router(router)
