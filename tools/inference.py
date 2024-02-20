import torch
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

class HiddenInjector:
    def __init__(
        self,
        model: PreTrainedModel,
        injection_layers: torch.Tensor,  # (batch_size)
        injection_positions: torch.Tensor,  # (batch_size)
        hiddens_to_inject: torch.Tensor,  # (batch_size, hidden_size)
    ):
        """
        Args:
            model: The model to inject hidden states into
            injection_layer: the layer to inject hidden states into, for each example in the batch (batch_size)
            injection_position: the position to inject hidden states into, for each example in the batch (batch_size)
            hidden_to_inject: the hidden states to inject, for each example in the batch (batch_size, hidden_size)
        """

        self._model = model
        self._injection_layer = injection_layers
        self._injection_position = injection_positions
        self._hidden_to_inject = hiddens_to_inject

        self._hooks = []

    def __enter__(self):
        self._register_forward_hooks()

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self._hooks:
            hook.remove()

    def _register_forward_hooks(self):
        def inject_hidden_hook(layer_idx):
            def inject_hidden(mod, inp, out):
                hidden_states = out[0] if isinstance(out, tuple) else out

                mask = self._injection_layer == layer_idx
                if mask.any():
                    hidden_to_inject = self._hidden_to_inject.to(hidden_states.device).type(hidden_states.dtype)
                    idx_to_inject = torch.arange(hidden_states.shape[0], device=hidden_states.device)[mask]
                    hidden_states[idx_to_inject, self._injection_position[mask]] = hidden_to_inject[mask]

                return out

            return inject_hidden

        for i, layer in enumerate(get_layers(self._model)):
            hook = layer.register_forward_hook(inject_hidden_hook(i))
            self._hooks.append(hook)



class HiddenInjector_New:
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