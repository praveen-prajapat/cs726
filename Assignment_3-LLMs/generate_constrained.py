import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class ConstrainedTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        eos_id: int, 
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the ConstrainedTextGenerator class.
            
            model: LLM
            tokenizer: LLM's tokenizer.
            eos_id: End-of-sequence token id 
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"], word_list: list
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Word-Constrained decoding technique. (refer assignment document for more details)
            
            `word_list`: contains bag of words for the particular example

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    

        output_ids = input_ids
        word_list_ids = [self.tokenizer.encode(word, add_special_tokens=False) for word in word_list]
        valid_token_ids = [token_id for word in word_list_ids for token_id in word]
        generated_tokens = []

        for _ in range(self.max_output_len):
            outputs = self.model(input_ids=output_ids)
            logits = outputs.logits[:, -1, :]  
            probs = nn.functional.softmax(logits, dim=-1) 
            
            valid_token_mask = torch.zeros_like(probs)  
            for token_id in range(probs.shape[-1]):
                if token_id in valid_token_ids:
                    valid_token_mask[:, token_id] = 1 
            
            masked_probs = probs * valid_token_mask
            masked_probs = masked_probs + (valid_token_mask == 0) * -1e10 
            next_token_id = torch.argmax(masked_probs, dim=-1)
            if next_token_id == self.eos_token_id:
                print("EOS token reached.")
                break

            generated_tokens.append(next_token_id.item())
            output_ids = torch.cat((output_ids, next_token_id.unsqueeze(0)), dim=-1)
        
        return output_ids.squeeze(0)[input_ids.shape[1]:]  

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False 

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word_tokens: List[int]):
        node = self.root
        for token in word_tokens:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.is_word = True
    
    def is_valid_word(self, tokens: List[int]) -> bool:
        node = self.root
        for token in tokens:
            if token not in node.children:
                return False
            node = node.children[token]
        return node.is_word  