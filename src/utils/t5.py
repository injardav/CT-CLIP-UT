import torch
from transformers import T5Tokenizer, T5EncoderModel, T5Config, logging
from functools import lru_cache

logging.set_verbosity_error()

# Constants
DEFAULT_T5_NAME = 'google/t5-v1_1-base'
MAX_LENGTH = 256

# Caching model/tokenizer/config to avoid reloading
@lru_cache(maxsize=4)
def get_tokenizer(name):
    return T5Tokenizer.from_pretrained(name)

@lru_cache(maxsize=4)
def get_model(name):
    return T5EncoderModel.from_pretrained(name, ignore_mismatched_sizes=True)

@lru_cache(maxsize=4)
def get_config(name):
    return T5Config.from_pretrained(name)

def get_encoded_dim(name):
    return get_config(name).d_model

# T5 Encoder Wrapper
class T5Encoder:
    def __init__(self, name=DEFAULT_T5_NAME, device='cpu'):
        self.device = torch.device(device)
        self.tokenizer = get_tokenizer(name)
        self.model = get_model(name).to(self.device)
        self.model.eval()
        self.tokens = []

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self

    def get_token_indices(self, keywords):
        """
        Return a dictionary mapping each matched keyword to its corresponding token indices
        in the tokenized report.
    
        If a keyword is not matched, it won't be included in the output.
        """
        def normalize(tokens):
            return [t.lstrip('▁').lower() for t in tokens]
    
        keyword_to_indices = {}
    
        tokens = self.tokens
        tokens_norm = normalize(tokens)
    
        for kw in keywords:
            kw_ids = self.tokenizer(kw, add_special_tokens=False).input_ids
            kw_tokens = self.tokenizer.convert_ids_to_tokens(kw_ids)
            kw_tokens_norm = normalize(kw_tokens)
    
            for i in range(len(tokens_norm) - len(kw_tokens_norm) + 1):
                window = tokens_norm[i:i + len(kw_tokens_norm)]
                if window == kw_tokens_norm:
                    matched_indices = list(range(i, i + len(kw_tokens_norm)))
                    keyword_to_indices[kw] = matched_indices
                    break  # only take the first match
    
        return keyword_to_indices

    def encode(self, texts, max_length=MAX_LENGTH):
        encoded = self.tokenizer(
            texts,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=max_length
        )

        input_ids = encoded.input_ids.to(self.device)
        attn_mask = encoded.attention_mask.to(self.device)
        self.tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        with torch.cuda.amp.autocast(enabled=False):
            output = self.model(input_ids=input_ids, attention_mask=attn_mask)
            hidden_states = output.last_hidden_state

        # Mask padding tokens
        attn_mask = attn_mask.unsqueeze(-1).bool()
        hidden_states = hidden_states.masked_fill(~attn_mask, 0.)

        return hidden_states