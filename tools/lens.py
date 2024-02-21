from tools.inference import modified_forward
from tools.utils import show_outputs

def id_lens(model, tokenizer, hiddens, from_layer, gen_len=1):
    batch_size=hiddens.size(0)
    prompts_lens=['cat->cat\nsad->sad\nA->A\nx->' for _ in range(batch_size)]
    # prompts_lens=['The capital of x:' for _ in range(batch_size)]

    outputs = modified_forward(model, tokenizer, prompts_lens, hiddens, from_layer, token_pos=-2, gen_len=gen_len)
    result = show_outputs(tokenizer, outputs, without_prompt=prompts_lens)
    return result

def patch_scope(
  model, tokenizer, hiddens, 
  target_prompts='cat->cat\nsad->sad\nA->A\nx->', 
  target_token_id = -2,
  target_layer_id = 0,
  gen_len = 1):

  batch_size = hiddens.size(0)
  prompts_lens = [target_prompt for _ in range(batch_size)]

  outputs = modified_forward(model, tokenizer, prompts_lens, hiddens, target_layer_id, token_pos=target_token_id, gen_len=gen_len)
  result = show_outputs(tokenizer, outputs, without_prompt=prompts_lens)
  return result