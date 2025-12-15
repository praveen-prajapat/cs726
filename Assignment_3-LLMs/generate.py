import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class TextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        decoding_strategy: str, 
        eos_id: int, 
        max_output_len: int = 10,
        tau: int = 1,
        k: int = 10,
        p: int = 0.5
    ) -> None:
        '''
            Initialize the TextGenerator class.
            
            model: LLM
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            max_output_len: Maximum number of tokens to be generated.
            tau: Temperature parameter for random sampling
            k: Top-k parameter for top-k sampling
            p: Cumulative probability threshold for nucleus sampling
            
            Do not edit.
        '''
        self.model = model
        self.decoding_strategy = decoding_strategy
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.tau = tau
        self.k = k 
        self.p = p
        
        if decoding_strategy == "greedy":
            self.generator_func = self.greedy_decoding
        elif decoding_strategy == "random":
            self.generator_func = self.random_sampling
        elif decoding_strategy == "topk":
            self.generator_func = self.topk_sampling
        elif decoding_strategy == "nucleus":
            self.generator_func = self.nucleus_sampling

    def __call__(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"], 
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def greedy_decoding(
        self,
        input_ids: Int[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]: 
        '''
            Implement Greedy decoding technique. (refer assignment document for more details)

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
        for _ in range(self.max_output_len):
            outputs = self.model(input_ids=output_ids)
            logits = outputs.logits[:, -1, :]  
            next_token_id = torch.argmax(logits, dim=-1)  
            if next_token_id == self.eos_token_id:
                break
            output_ids = torch.cat((output_ids, next_token_id.unsqueeze(0)), dim=-1)
        return output_ids.squeeze(0)[input_ids.shape[1]:]

        
    def random_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Random sampling technique. (refer assignment document for more details)

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
        for _ in range(self.max_output_len):
            outputs = self.model(input_ids=output_ids)
            logits = outputs.logits[:, -1, :]  
            logits = logits / self.tau         
            probs = nn.functional.softmax(logits, dim=-1)

            next_token_id = torch.multinomial(probs, 1) 
            if next_token_id.item() == self.eos_token_id:
                break

            output_ids = torch.cat((output_ids, next_token_id), dim=-1) 
        
        return output_ids.squeeze(0)[input_ids.shape[1]:] 
    
    def topk_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Top-k sampling technique. (refer assignment document for more details)

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
        for _ in range(self.max_output_len):
            outputs = self.model(input_ids=output_ids)
            logits = outputs.logits[:, -1, :]
            top_k_values, top_k_indices = torch.topk(logits, self.k, dim=-1)
            top_k_probs = nn.functional.softmax(top_k_values, dim=-1)
            next_token_id = torch.multinomial(top_k_probs, 1) 
            next_token_id = top_k_indices.gather(-1, next_token_id) 
            if next_token_id == self.eos_token_id:
                break
            output_ids = torch.cat((output_ids, next_token_id), dim=-1)
        return output_ids.squeeze(0)[input_ids.shape[1]:]
    
    def nucleus_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Nucleus sampling technique. (refer assignment document for more details)

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

        for _ in range(self.max_output_len):
            outputs = self.model(input_ids=output_ids)
            logits = outputs.logits[:, -1, :] 
            probs = nn.functional.softmax(logits, dim=-1) 
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            nucleus_mask = cumulative_probs <= self.p
            nucleus_mask[..., 0] = True
            top_p_probs = sorted_probs[nucleus_mask]
            top_p_indices = sorted_indices[nucleus_mask]
            top_p_probs = top_p_probs / top_p_probs.sum()
            sampled_idx = torch.multinomial(top_p_probs, 1)
            next_token_id = top_p_indices[sampled_idx]
            if next_token_id.item() == self.eos_token_id:
                break
            output_ids = torch.cat((output_ids, next_token_id.unsqueeze(0)), dim=-1)

        return output_ids.squeeze(0)[input_ids.shape[1]:]