import torch

def show_outputs(tokenizer, outputs, without_prompt=None):
    if without_prompt==None:
        start_pos=0
    else:
        inputs = tokenizer(without_prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        start_pos = input_ids.size(1)

    responses=[]
    for s in outputs.sequences:
        output = tokenizer.decode(s[start_pos:])
        responses.append(output)
    return responses

# def extract_hidden(outputs, layer_id, call_id = 0, token_pos = -1) -> torch.Tensor:
#     # IMPORTANT: layer_id denotes the layer where after(!) it the hidden state is hooked.
#     hiddens = outputs.hidden_states[call_id][layer_id+1][:,token_pos,:]
#     return hiddens

# def extract_hidden_list(outputs, layer_id, token_pos_list, call_id=0) -> torch.Tensor:
#     if isinstance(token_pos_list, list):
#         token_pos_list = torch.tensor(token_pos_list, dtype=torch.long)
#     layer_hiddens = outputs.hidden_states[call_id][layer_id + 1]

#     extracted_hiddens = []
#     for idx, token_pos in enumerate(token_pos_list):
#         hiddens = layer_hiddens[idx, token_pos, :]
#         extracted_hiddens.append(hiddens)
#     extracted_hiddens_tensor = torch.stack(extracted_hiddens)
#     return extracted_hiddens_tensor

def global2local(token_pos, prompt_len):
    if token_pos<prompt_len:
        call_id=0
        token_pos_loc=token_pos
    else:
        call_id=token_pos-prompt_len
        token_pos_loc=0      
    return call_id, token_pos_loc

def extract_hidden(outputs, layer_id, token_pos) -> torch.Tensor:
    prompt_len = outputs.hidden_states[0][0].size(0)
    if isinstance(token_pos, int):
        call_id, token_pos_loc = global2local(token_pos, prompt_len)
        extracted_hiddens = outputs.hidden_states[call_id][layer_id+1][:,token_pos_loc,:]
    elif isinstance(token_pos, list):
        extracted_hiddens_list = []
        for idx, token_pos_ in enumerate(token_pos):
            call_id, token_pos_loc = global2local(token_pos_, prompt_len)
            hiddens = outputs.hidden_states[call_id][layer_id+1][:,token_pos_loc,:]
            extracted_hiddens_list.append(hiddens)
        extracted_hiddens = torch.stack(extracted_hiddens_list)
    else:
        raise ValueError(f"Wrong type for token_pos ({type(token_pos)}), only int or list are supported")
    return extracted_hiddens

def break2tokens(tokenizer, promtps):
    inputs = tokenizer(promtps, return_tensors="pt", padding=True)
    prompt_list=[]
    for ss in inputs["input_ids"]:
        token_list=[]
        for s in ss.tolist():
            output = tokenizer.decode(s)
            token_list.append(output)
        prompt_list.append(token_list)
    return prompt_list