from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
import torch.nn as nn
import cs336_alignment.utils as utils
import json
import random
import torch

# load alpaca prompt
with open('/afs/cs.stanford.edu/u/cye/assignment5-alignment/cs336_alignment/prompts/alpaca_sft.prompt', 'r') as f:
    ALPACA_PROMPT = f.read()

class ITDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle):
        self.tokenizer = tokenizer # from transformers
        self.dataset_path = dataset_path
        self.seq_length = seq_length
        self.shuffle = shuffle

        self.load_data()
        self.tokenize()
        self.break_into_chunks()
    
    def load_data(self):
        self.data = []
        # read from jsonl file, each line is a json object
        with open(self.dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.data.append(ALPACA_PROMPT.format(instruction=data['prompt'], 
                                                      response=data['response']))
    
    def tokenize(self):
        print(f"Tokenizing {len(self.data)} items")
        encodings = self.tokenizer(self.data, truncation=False, padding=False)
        self.encodings = encodings['input_ids']
        if self.shuffle:
            random.shuffle(self.encodings)
    
    def break_into_chunks(self):
        # concatenate all encodings with tokenizer.pad_token_id in between
        self.stream = []
        for i in range(0, len(self.encodings)):
            encoding = self.encodings[i]
            self.stream.extend(encoding)
            self.stream.append(self.tokenizer.eos_token_id) 
        # convert to tensor
        self.stream = torch.tensor(self.stream)
        self.chunks = []
        n_chunks = len(self.stream - 1) // self.seq_length
        for i in range(n_chunks):
            self.chunks.append({'input_ids': self.stream[i * self.seq_length: (i + 1) * self.seq_length],
                                'labels': self.stream[i * self.seq_length + 1: (i + 1) * self.seq_length + 1]})
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx]

def iterate_batches(dataset: Dataset, batch_size: int, shuffle: bool):
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

    return iter(dataloader)

def DPOLoss(lm: nn.Module, lm_ref: nn.Module, tokenizer: PreTrainedTokenizerBase, beta: float, prompt: str, response_chosen: str, response_rejected: str):
    # format w/ alpaca prompt
    win = ALPACA_PROMPT.format(instruction = prompt, response = response_chosen)
    lose = ALPACA_PROMPT.format(instruction = prompt, response = response_rejected)
    print(win)
    print(lose)

    # get logits for all tokens; must make it look batch_size x seq_len
    w_tokens = torch.tensor(tokenizer(win)['input_ids'] + [tokenizer.eos_token_id]).unsqueeze(0)
    l_tokens = torch.tensor(tokenizer(lose)['input_ids'] + [tokenizer.eos_token_id]).unsqueeze(0)

    print(w_tokens.shape, l_tokens.shape)
    
    w_theta_lp = torch.sum(utils.get_response_log_probs(lm, w_tokens[:, :-1], w_tokens[:, 1:])["log_probs"])
    w_ref_lp = torch.sum(utils.get_response_log_probs(lm_ref, w_tokens[:, :-1], w_tokens[:, 1:])["log_probs"])
    l_theta_lp = torch.sum(utils.get_response_log_probs(lm, l_tokens[:, :-1], l_tokens[:, 1:])["log_probs"])
    l_ref_lp = torch.sum(utils.get_response_log_probs(lm_ref, l_tokens[:, :-1], l_tokens[:, 1:])["log_probs"])

    h = beta * (w_theta_lp - w_ref_lp - l_theta_lp + l_ref_lp)

    return -nn.functional.logsigmoid(h)