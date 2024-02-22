import torch
from tools.inference import modified_forward
from tools.utils import show_outputs

def patch_scope(
    model, tokenizer, hiddens, 
    target_prompt='cat->cat\nsad->sad\nA->A\nx->', 
    target_token_pos = -2,
    target_layer_id = 0,
    gen_len = 1):

    batch_size = hiddens.size(0)
    prompts_lens = [target_prompt for _ in range(batch_size)]

    outputs = modified_forward(model, tokenizer, prompts_lens, hiddens, target_layer_id, token_pos = target_token_pos, gen_len=gen_len)
    result = show_outputs(tokenizer, outputs, without_prompt=prompts_lens)
    
    return result