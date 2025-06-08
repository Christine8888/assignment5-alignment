from vllm import SamplingParams

QWEN_25 = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
PROMPT_PATH = '/home/c-cye/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt'
MATH_TRAIN_PATH = '/data/a5-alignment/MATH/train.jsonl'
MATH_SFT_PATH = '/data/a5-alignment/MATH/sft.jsonl'
MATH_VAL_PATH = '/data/a5-alignment/MATH/validation.jsonl'

EVAL_SAMPLING_PARAMS = SamplingParams(
        temperature = 1.0, 
        top_p = 1.0, 
        max_tokens = 1024, 
        stop = ["</answer>"], 
        include_stop_str_in_output = True,
        min_tokens = 4,
    )