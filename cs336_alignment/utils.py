from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import List, Dict
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from vllm import LLM, SamplingParams
from typing import Callable
from cs336_alignment.baseline import evaluate_vllm

def tokenize_prompt_and_output(prompt_strs: List[str], output_strs: List[str], tokenizer: PreTrainedTokenizer, device: str = 'cuda') -> Dict[str, List[int]]:
    n_examples = len(prompt_strs)
    tokenized_prompts = tokenizer(prompt_strs, return_tensors = None, padding = False, truncation = False)
    tokenized_outputs = tokenizer(output_strs, return_tensors = None, padding = False, truncation = False)

    # 0: prompt length, 1: output length, 2: total length
    token_tensors = []
    attention_tensors = []
    for i in range(n_examples):
        token_tensors.append(torch.cat([torch.tensor(tokenized_prompts['input_ids'][i]), 
                                        torch.tensor(tokenized_outputs['input_ids'][i])]))
        # 0 for prompt, 1 for output
        attention_tensors.append(torch.cat([torch.zeros_like(torch.tensor(tokenized_prompts['attention_mask'][i])), 
                                            torch.tensor(tokenized_outputs['attention_mask'][i])]))

    padded_token_tensors = pad_sequence(token_tensors, batch_first = True, padding_value = tokenizer.pad_token_id)
    input_ids = padded_token_tensors[:, :-1].to(device)
    labels = padded_token_tensors[:, 1:].clone().to(device)
    response_mask = pad_sequence(attention_tensors, batch_first = True, padding_value = 0)[:, 1:].to(device)
    # view response mask as a boolean tensor
    response_mask = response_mask.bool().to(device)

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # input dim: batch_size, seq_len, vocab_size
    logprobs = logits - torch.logsumexp(logits, dim = -1, keepdim = True)
    entropy = -torch.sum(torch.exp(logprobs) * logprobs, dim = -1) # sum over vocab dim
    return entropy # shape: batch_size, seq_len

def get_response_log_probs(model: PreTrainedModel,
                 input_ids: torch.Tensor,
                 labels: torch.Tensor,
                 return_token_entropy: bool = False):
    return_dict = {}
    logits = model(input_ids).logits # get logits for all next-tokens; batch_size x seq_len
    logprobs = F.log_softmax(logits, dim = -1)
    token_logprobs = torch.gather(logprobs, dim = -1, index = labels.unsqueeze(-1)).squeeze(-1) # faster than logprobs[:, :, labels]
    return_dict["log_probs"] = token_logprobs
    if return_token_entropy:
        return_dict["token_entropy"] = compute_entropy(logits)
    return return_dict

def masked_normalize(tensor: torch.Tensor,
                     mask: torch.Tensor,
                     normalize_constant: float,
                     dim: int | None = None) -> torch.Tensor:
    masked = tensor * mask
    tsum = torch.sum(masked, dim = dim)
    return tsum / normalize_constant

def sft_microbatch_train_step(policy_log_probs: torch.Tensor,
                              response_mask: torch.Tensor,
                              gradient_accumulation_steps: int,
                              normalize_constant: float = 1.0,
                              do_backward: bool = True) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    ce = -policy_log_probs
    loss = masked_normalize(ce, mask = response_mask, dim = -1, normalize_constant = normalize_constant)
    loss = loss.mean() # averaging over batch size
    scaled_loss = loss / gradient_accumulation_steps
    if do_backward:
        scaled_loss.backward()
    metadata = {'loss': loss.item(),
                'scaled_loss': scaled_loss.item(),
                'response_lengths': response_mask.sum(dim = -1)} 

    return scaled_loss, metadata

def log_generations(vllm_model: LLM,
        reward_fn: Callable[[str, str], dict[str, float]],
        prompts: List[str],
        answers: List[str],
        sampling_params: SamplingParams,
        log_file: str = None,
        iter_idx: int = 0):
    """
    Log the generations of the model.
    """
    results = evaluate_vllm(vllm_model, reward_fn, prompts, answers, sampling_params, save_path = None)

    with open(log_file, 'a') as f:
        f.write("-" * 100 + "\n")
        f.write(f"ITERATION {iter_idx}\n")
        for result in results:
            f.write(f"\nprompt: {result['prompt']}\nresponse: {result['model_output']}\nanswer: {result['expected_answer']}\nformat_reward: {result['format_reward']}\nanswer_reward: {result['answer_reward']}\nreward: {result['reward']}\n\n")
        
        f.write("-" * 100 + "\n")