import tiktoken
from transformers import AutoTokenizer

def gpt_tokens(model_name: str, string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def model_tokens(model_name: str, string: str) -> int:
    """Returns the number of tokens in a text string."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_tokens = len(tokenizer.encode(string))
    return num_tokens