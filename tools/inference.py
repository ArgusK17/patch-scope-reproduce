import torch
from torch import nn
from typing import ContextManager, Dict, Iterable, List, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from contextlib import AbstractContextManager, ExitStack

from transformers import GenerationConfig

generation_config = GenerationConfig(
    do_sample=False,
)

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

def modified_forward(model, tokenizer, prompts, hiddens_to_inject, layer_pos, token_pos=-1, with_hiddens=False, with_attns=False, gen_len=1):
  
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].cuda()

    intermediate_layer = torch.tensor(layer_pos).repeat(len(input_ids)) #batch_id, layer_id
    injection_positions = token_pos * torch.ones_like(intermediate_layer, dtype=torch.long)

    forward_modifiers = [
        HiddenInjector(
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

# class HiddenInjector:
#     def __init__(
#         self,
#         model: PreTrainedModel,
#         injection_layers: torch.Tensor,  # (batch_size)
#         injection_positions: torch.Tensor,  # (batch_size)
#         hiddens_to_inject: torch.Tensor,  # (batch_size, hidden_size)
#     ):
#         """
#         Args:
#             model: The model to inject hidden states into
#             injection_layer: the layer to inject hidden states into, for each example in the batch (batch_size)
#             injection_position: the position to inject hidden states into, for each example in the batch (batch_size)
#             hidden_to_inject: the hidden states to inject, for each example in the batch (batch_size, hidden_size)
#         """

#         self._model = model
#         self._injection_layer = injection_layers
#         self._injection_position = injection_positions
#         self._hidden_to_inject = hiddens_to_inject

#         self._hooks = []

#     def __enter__(self):
#         self._register_forward_hooks()

#     def __exit__(self, exc_type, exc_value, traceback):
#         for hook in self._hooks:
#             hook.remove()

#     def _register_forward_hooks(self):
#         def inject_hidden_hook(layer_idx):
#             def inject_hidden(mod, inp, out):
#                 hidden_states = out[0] if isinstance(out, tuple) else out

#                 mask = self._injection_layer == layer_idx
#                 if mask.any():
#                     hidden_to_inject = self._hidden_to_inject.to(hidden_states.device).type(hidden_states.dtype)
#                     idx_to_inject = torch.arange(hidden_states.shape[0], device=hidden_states.device)[mask]
#                     hidden_states[idx_to_inject, self._injection_position[mask]] = hidden_to_inject[mask]

#                 return out

#             return inject_hidden

#         for i, layer in enumerate(get_layers(self._model)):
#             hook = layer.register_forward_hook(inject_hidden_hook(i))
#             self._hooks.append(hook)

class HiddenInjector:
    def __init__(
        self,
        model: PreTrainedModel,
        injection_layers: torch.Tensor,  # (batch_size)
        injection_positions: torch.Tensor,  # (batch_size)
        hiddens_to_inject: torch.Tensor,  # (batch_size, hidden_size)
    ):
        self._model = model
        self._injection_layer = injection_layers
        self._injection_position = injection_positions
        self._hidden_to_inject = hiddens_to_inject
        self._hooks = []
        self._used = False  # Flag to indicate if the hook has been used

    def __enter__(self):
        self._register_forward_hooks()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []  # Clear hooks

    def _register_forward_hooks(self):
        def inject_hidden_hook(layer_idx):
            def inject_hidden(mod, inp, out):
                if self._used:
                    return out  # Do nothing if the hook has been used

                hidden_states = out[0] if isinstance(out, tuple) else out

                mask = self._injection_layer == layer_idx
                if mask.any():
                    hidden_to_inject = self._hidden_to_inject.to(hidden_states.device).type(hidden_states.dtype)
                    idx_to_inject = torch.arange(hidden_states.shape[0], device=hidden_states.device)[mask]
                    hidden_states[idx_to_inject, self._injection_position[mask]] = hidden_to_inject[mask]
                    self._used = True  # Set the flag after the hook is used

                return out

            return inject_hidden

        for i, layer in enumerate(get_layers(self._model)):
            hook = layer.register_forward_hook(inject_hidden_hook(i))
            self._hooks.append(hook)

class CombinedContextManager(AbstractContextManager):
    def __init__(self, context_managers):
        self.context_managers = context_managers
        self.stack = None

    def __enter__(self):
        self.stack = ExitStack()
        for cm in self.context_managers:
            self.stack.enter_context(cm)
        return self.stack

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stack is not None:
            self.stack.__exit__(exc_type, exc_val, exc_tb)

def modified_forward_context_manager(
    model: PreTrainedModel, forward_modifiers: Optional[Iterable[ContextManager]] = ()
) -> ContextManager:
    context_manager = CombinedContextManager([*forward_modifiers])
    return context_manager

def get_nested_attr(obj, attr_path):
    attrs = attr_path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj

def set_nested_attr(obj, attr_path, value):
    attrs = attr_path.split(".")
    parent = get_nested_attr(obj, ".".join(attrs[:-1]))
    setattr(parent, attrs[-1], value)

def find_longest_modulelist(model, path=""):
    """
    Recursively find the longest nn.ModuleList in a PyTorch model.
    Args:
        model: PyTorch model.
        path: Current path in the model (used for recursion).
    Returns:
        Tuple with path and length of the longest nn.ModuleList found.
    """
    longest_path = path
    longest_len = 0

    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > longest_len:
            longest_len = len(child)
            longest_path = f"{path}.{name}" if path else name

        # Recursively check the child's children
        child_path, child_len = find_longest_modulelist(child, f"{path}.{name}" if path else name)
        if child_len > longest_len:
            longest_len = child_len
            longest_path = child_path

    return longest_path, longest_len


def find_module(block, keywords):
    """
    Try to find a module in a transformer block.
    Args:
        block: Transformer block (nn.Module).
        keywords: List of possible module names (str).
    Returns:
        The found module if found, else None.
    """
    for name, module in block.named_modules():
        if any(keyword in name for keyword in keywords):
            return module
    submodule_names = [name for name, _ in block.named_modules()]
    raise ValueError(f"Could not find keywords {keywords} in: {submodule_names}")


def get_embedding_layer(model: PreTrainedModel):
    # model_type = model.__class__.__name__
    # if model_type == "LlamaForCausalLM":
    #     return model.model.embed_tokens
    # elif model_type == "RWForCausalLM":
    #     return model.transformer.word_embeddings

    keywords = ["emb", "wte"]
    return find_module(model, keywords)

def get_lm_head(model: PreTrainedModel):
    return nn.Sequential(model.model.norm, model.lm_head)

def get_layers_path(model: PreTrainedModel):
    longest_path, longest_len = find_longest_modulelist(model)
    return longest_path

def get_layers(model: PreTrainedModel):
    # model_type = model.__class__.__name__
    # if model_type == "LlamaForCausalLM":
    #     return model.model.layers
    # elif model_type == "RWForCausalLM":
    #     return model.transformer.h

    longest_path = get_layers_path(model)
    return get_nested_attr(model, longest_path)


def get_attention_layers(model: PreTrainedModel):
    # model_type = model.__class__.__name__
    # if model_type == "LlamaForCausalLM":
    #     return [layer.self_attn for layer in layers]
    # elif model_type == "RWForCausalLM":
    #     return [layer.self_attention for layer in layers]

    layers = get_layers(model)
    keywords = ["attention", "attn"]
    attention_layers = [find_module(layer, keywords) for layer in layers]
    return attention_layers


def get_mlp_layers(model: PreTrainedModel):
    # model_type = model.__class__.__name__
    # if model_type == "LlamaForCausalLM":
    #     return [layer.mlp for layer in layers]
    # elif model_type == "RWForCausalLM":
    #     return [layer.mlp for layer in layers]

    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    mlp_layers = [find_module(layer, mlp_keywords) for layer in layers]
    return mlp_layers