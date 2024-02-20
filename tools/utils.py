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

def extract_hidden(outputs, layer_id, call_id = 0, token_pos = -1) -> torch.Tensor:
    hiddens = outputs.hidden_states[call_id][layer_id+1][:,token_pos,:]
    return hiddens


def extract_hidden_list(outputs, layer_id, token_pos_list, call_id=0) -> torch.Tensor:
    if isinstance(token_pos_list, list):
        token_pos_list = torch.tensor(token_pos_list, dtype=torch.long)
    layer_hiddens = outputs.hidden_states[call_id][layer_id + 1]

    extracted_hiddens = []
    for idx, token_pos in enumerate(token_pos_list):
        hiddens = layer_hiddens[idx, token_pos, :]
        extracted_hiddens.append(hiddens)
    extracted_hiddens_tensor = torch.stack(extracted_hiddens)

    return extracted_hiddens_tensor

def break2tokens(tokenizer, promtp: str):
    inputs = tokenizer(promtp, return_tensors="pt", padding=True)
    
    task_list=[]
    for ss in inputs["input_ids"]:
        token_list=[]
        for s in ss.tolist():
            output = tokenizer.decode(s)
            token_list.append(output)
        task_list.append(token_list)
    return task_list
