from tools.utils import HiddenInjector, HiddenInjector_New, modified_forward_context_manager
import torch

from transformers import GenerationConfig
generation_config = GenerationConfig(
    do_sample=False,
)

# test

def forward(model, tokenizer, prompts, with_hiddens=False, with_attns=False, gen_len=1):
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].cuda()

    outputs = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores = True,
        output_hidden_states = with_hiddens,
        output_attentions = with_attns,
        max_new_tokens=gen_len,
        pad_token_id=tokenizer.eos_token_id
    )
    return outputs

"""
layer_id denotes the layer where after(!) it the hidden state is hooked.
For hidden state patching, layer_id should be in 0-30. Set it to 31 will result in incorrect outcomes.
"""

def modified_forward(model, tokenizer, prompts, hiddens_to_inject, layer_pos, token_pos=-1, with_hiddens=False, with_attns=False, gen_len=1):
  
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].cuda()

    intermediate_layer = torch.tensor(layer_pos).repeat(len(input_ids)) #batch_id, layer_id
    injection_positions = token_pos * torch.ones_like(intermediate_layer, dtype=torch.long)

    forward_modifiers = [
        HiddenInjector_New(
            model,
            injection_layers = intermediate_layer,
            injection_positions = injection_positions,
            hiddens_to_inject = hiddens_to_inject,
        )
    ]
    context_manager = modified_forward_context_manager(model, forward_modifiers=forward_modifiers)
    
    with context_manager:
        outputs = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores = True,
            output_hidden_states = with_hiddens,
            output_attentions = with_attns,
            max_new_tokens=gen_len,
            pad_token_id=tokenizer.eos_token_id
        )
    return outputs
