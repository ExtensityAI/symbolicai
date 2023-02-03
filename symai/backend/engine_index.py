from .settings import SYMAI_CONFIG
from typing import List
from .base import Engine
import pinecone


class IndexEngine(Engine):
    def __init__(self):
        super().__init__()
        self.documents = None # lazy init
        self.index = None # lazy init
        self.index_name: str = 'data-index'
        self.index_dims = 1536
        self.index_topk = 5
        self.index_inc_values = True
        self.index_metric = 'cosine'
        config = SYMAI_CONFIG
        self.api_key = config['INDEXING_ENGINE_API_KEY']
        self.environment = config['INDEXING_ENGINE_ENVIRONMENT']
        self.old_api_key = self.api_key
        self.old_environment = self.environment
        self.index = None
        
    def command(self, wrp_params):
        super().command(wrp_params)
        if 'INDEXING_ENGINE_API_KEY' in wrp_params:
            self.api_key = wrp_params['INDEXING_ENGINE_API_KEY']
        if 'INDEXING_ENGINE_ENVIRONMENT' in wrp_params:
            self.environment = wrp_params['INDEXING_ENGINE_ENVIRONMENT']

    def forward(self, *args, **kwargs) -> List[str]:
        if self.documents is None or self.old_api_key != self.api_key or self.old_environment != self.environment:
            index_name = kwargs['index_name'] if 'index_name' in kwargs else self.index_name
            index_dims = kwargs['index_dims'] if 'index_dims' in kwargs else self.index_dims
            index_metric = kwargs['index_metric'] if 'index_metric' in kwargs else self.index_metric
            pinecone.init(api_key=self.api_key, environment=self.environment)
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(name=index_name, dimension=index_dims, metric=index_metric)
            self.index = pinecone.Index(index_name=index_name)
            self.old_api_key = self.api_key
            self.old_environment = self.environment
        
        operation = kwargs['operation']
        query = kwargs['prompt']
        input_handler = kwargs['input_handler'] if 'input_handler' in kwargs else None
        if input_handler:
            input_handler((query,))
        
        if operation == 'search':
            top_k = kwargs['index_topk'] if 'index_topk' in kwargs else self.index_topk
            index_inc_values = kwargs['index_inc_values'] if 'index_inc_values' in kwargs else self.index_inc_values
            
            rsp = self.index.query(vector=query, 
                                   top_k=top_k, 
                                   include_values=index_inc_values)
            
        elif operation == 'add':
            rsp = self.index.upsert(vectors=query)
            
        elif operation == 'config':            
            index_name = kwargs['index_name'] if 'index_name' in kwargs else self.index_name

            del_ = kwargs['index_del'] if 'index_del' in kwargs else False
            if self.index is not None and del_:
                pinecone.delete_index(index_name)
                
            get_ = kwargs['index_get'] if 'index_get' in kwargs else False
            if self.index is not None and get_:
                self.index = pinecone.Index(index_name=index_name)
            
        else:
            raise ValueError('Invalid operation')            

        output_handler = kwargs['output_handler'] if 'output_handler' in kwargs else None
        if output_handler:
            output_handler(rsp)
        
        return [rsp]
    
    def prepare(self, args, kwargs, wrp_params):
        wrp_params['prompt'] = wrp_params['prompt']
        