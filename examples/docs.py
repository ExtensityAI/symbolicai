from botdyn.symbol import Symbol, Expression
from botdyn.post_processors import StripPostProcessor
from botdyn.pre_processors import PreProcessor
import botdyn as bd


DOC_CONTEXT = '''General Python Teplate example:
    """_summary_

    Args:
        prompt (_type_, optional): _description_. Defaults to 'Summarize the content of the following text:\n'.
        constraints (List[Callable], optional): _description_. Defaults to [].
        default (object, optional): _description_. Defaults to None.
        pre_processor (List[PreProcessor], optional): _description_. Defaults to [TextMessagePreProcessor()].
        post_processor (List[PostProcessor], optional): _description_. Defaults to [StripPostProcessor()].

    Returns:
        _type_: _description_
    """

    Method Signature Example:
    def summarize(prompt: str = 'Summarize the content of the following text:\n', 
              constraints: List[Callable] = [], 
              default: object = None, 
              pre_processor: List[PreProcessor] = [TextMessagePreProcessor()],
              post_processor: List[PostProcessor] = [StripPostProcessor()],
              **wrp_kwargs):
    return few_shot(prompt,
                    examples=[],
                    constraints=constraints,
                    default=default,
                    limit=1,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    **wrp_kwargs)
                    
    Documentation Example Text:
    """Summarizes the content of a text.

    Args:
        prompt (str, optional): The prompt describing the task. Defaults to 'Summarize the content of the following text:\n'.
        constraints (List[Callable], optional): A list of contrains applied to the model output to verify the output. Defaults to [].
        default (object, optional): The default value to be returned if the task cannot be solved. Defaults to None. Alternatively, one can implement the decorated function.
        pre_processor (List[PreProcessor], optional): A list of pre-processors to be applied to the input and shape the input to the model. Defaults to None.
        post_processor (List[PostProcessor], optional): A list of post-processors to be applied to the model output and before returning the result. Defaults to [StripPostProcessor()].

    Returns:
        str: The summary of the text.
    """ 
    '''


class DocsPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args, **kwds):
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        return 'Method Signature:\n{}\nDocumentation Text:'.format(str(wrp_self))


class Docs(Expression):
    @property
    def static_context(self):
        return DOC_CONTEXT + super().static_context
    
    def forward(self, sym: Symbol, *args, **kwargs):
        @bd.few_shot(prompt="Create only Python style documentation text based on the given template and shown in the example with the provided function signature:", 
                     examples=[],
                     max_tokens=3000,
                     pre_processor=[DocsPreProcessor()],
                     post_processor=[StripPostProcessor()], **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(Docs(sym)))
    
    @property
    def _sym_return_type(self):
        return Docs
    

CPP_DOC_CONTEXT = """Documentation example for a C++ code snippet:

Method:
DLLEXPORT void * research_native_hyperscan_allocate_scratch(void * db_ptr) {

    hs_database_t * db_block = (hs_database_t *) db_ptr;
    hs_scratch_t * scratch = NULL;

    // Allocate scratch space
    hs_error_t err = hs_alloc_scratch(db_block, &scratch);
    if (err != HS_SUCCESS)
        throw std::runtime_error("ERROR: Unable to allocate scratch space. Exiting.\n");

    size_t scratch_size;
    err = hs_scratch_size(scratch, &scratch_size);
    //std::cout << "[HYPERSCAN] Scratch size is: " << scratch_size << "." << std::endl;
    if(err != HS_SUCCESS)
        throw std::runtime_error("ERROR: Could not request scratch size");

    return (void *) scratch;
}

Documentation text:
/**
 * Allocate a Hyperscan scratch given a database block
 *
 * @param db_ptr A pointer to a previously allocated hyperscan database
 * @return A pointer to an allocated scratch for the database
 */
"""

    
class CppDocs(Expression):
    @property
    def static_context(self):
        return CPP_DOC_CONTEXT + super().static_context
    
    def forward(self, sym: Symbol, *args, **kwargs):
        @bd.few_shot(prompt="Create only C++ style documentation text based on the given template and shown in the example with the provided function signature:", 
                     examples=[],
                     max_tokens=3000,
                     pre_processor=[DocsPreProcessor()],
                     post_processor=[StripPostProcessor()], **kwargs)
        def _func(_) -> str:
            pass
        return self._sym_return_type(_func(Docs(sym)))
    
    @property
    def _sym_return_type(self):
        return CppDocs
