from tools.inference import modified_forward
from tools.utils import show_outputs

def id_lens(model, tokenizer, hiddens, from_layer):
    batch_size=hiddens.size(0)
    prompts_lens=['cat->cat\nsad->sad\nA->A\nx->' for _ in range(batch_size)]

    outputs = modified_forward(model, tokenizer, prompts_lens, hiddens, from_layer, token_pos=-2, gen_len=1)
    result = show_outputs(tokenizer, outputs, without_prompt=prompts_lens)
    return result