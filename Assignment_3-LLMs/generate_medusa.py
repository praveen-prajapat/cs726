import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class MedusaTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        decoding_strategy: str, 
        eos_id: int, 
        use_no_medusa_heads: int = 5,
        beam_width: int = 2,
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the MedusaTextGenerator class.
            
            model: LLM
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            use_no_medusa_heads: Number of medusa heads to be used (maximum:5) (denoted as S).
            beam_width: Maximum number of candidates that can be present in the beam (denoted as W).
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        self.decoding_strategy = decoding_strategy
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.beam_width = beam_width
        
        assert use_no_medusa_heads <= 5, "The current medusa model supports at max 5 heads"
        self.no_heads = use_no_medusa_heads + 1
        
        if decoding_strategy == "single-head":
            self.generator_func = self.single_head_decoding
        elif decoding_strategy == "multi-head":
            self.generator_func = self.multi_head_decoding
        
    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def single_head_decoding(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        '''
            Implement Single-head decoding technique. Use only LM head for decoding here.
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''
#Edited
        generated_tokens = input_ids 
        
        for _ in range(self.max_output_len):
            output = self.model.base_model(
                torch.as_tensor(generated_tokens).cuda(), 
                past_key_values=None, 
            )
            logits = output.logits
            next_token = logits[:, -1, :].argmax(-1) 
            generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=-1)
            if next_token.item() == self.eos_token_id:
                break
        return generated_tokens.squeeze(0)[len(input_ids[0]):]
#Edited


    def multi_head_decoding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Implement multi-head decoding technique with beam search and scoring as per given instructions.

        Input:
            input_ids: tensor of shape (1, P)
        
        Returns:
            tensor of shape (T,), where T <= self.max_output_len
        """

        generated_tokens = input_ids
        
        step = 0
        while step < self.max_output_len:
            #Step - 1 (Getting medusa logits)
            input_tokens = torch.as_tensor(generated_tokens)
            medusa_logits, outputs, logits = self.model(input_tokens.cuda().unsqueeze(0), past_key_values=None)
            log_probs = nn.functional.log_softmax(medusa_logits[:, -1, :], dim=-1)  

            # Step - 2 (TopW candidate sequences)
            candidate_sequences, candidate_scores = beam_search(log_probs, generated_tokens, self.beam_width,self.no_heads)
            candidate_scores = torch.tensor(candidate_scores)
            
            #Step - 3 (Sequence selection out of all sequences in candidate_sequences (next_sequence))
            best_sequence_idx = torch.argmax(candidate_scores, dim=0)
            best_sequence = candidate_sequences[best_sequence_idx] 
            generated_tokens = best_sequence
            if generated_tokens[-1] == self.eos_token_id:
                break
            step += len(best_sequence) - len(generated_tokens)
        return generated_tokens.squeeze(0)[len(input_ids[0]):]



def beam_search(pt, past_tokens, W, S):
    candidates = [(tuple(past_tokens),)]  
    scores = [0]  

    for s in range(S):
        logpt_s = pt[0] 
        top_W_scores, top_W_indices = torch.topk(logpt_s, W, dim=-1, largest=True, sorted=False)
        new_candidates = []
        new_scores = []
        for c, candidate in enumerate(candidates):
            for idx in range(W):
                y_hat = top_W_indices[idx].item()
                new_score = scores[c] + top_W_scores[idx].item()
                new_candidate = candidate + (y_hat,)
                new_scores.append(new_score)
                new_candidates.append(new_candidate)

        sorted_new_candidates = sorted(zip(new_candidates, new_scores), key=lambda x: x[1], reverse=True)
        candidates, scores = zip(*sorted_new_candidates[:W])

    return zip(*sorted_new_candidates[:W])