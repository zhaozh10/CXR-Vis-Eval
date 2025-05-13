def extract_backbone(ckpt,prefix: str='backbone')->callable:
    state_dict=ckpt
    if prefix==None:
        for k in list(state_dict.keys()):
            if k.startswith('fc'):
                del state_dict[k]
        return state_dict
    else:
        for k in list(state_dict.keys()):
            if k.startswith(f'{prefix}.'):
                # print(k)
                if k.startswith('') and not k.startswith(f'{prefix}.fc'):
                    # remove prefix
                    state_dict[k[len(f"{prefix}."):]] = state_dict[k]
            # del掉不是backbone的部分
            del state_dict[k]
        return state_dict
    

def extract_vit_backbone(ckpt, source: str='mae',prefix=None)->callable:
    # prefix='encoder.layers
    state_dict=ckpt
    if prefix !=None:
         for k in list(state_dict.keys()):
            if k.startswith(f'{prefix}.'):
                # print(k)
                if not k.startswith(f'{prefix}.fc'):
                    # remove prefix
                    state_dict[k[len(f"{prefix}."):]] = state_dict[k]
            # del掉不是backbone的部分
            del state_dict[k]


    if source == None:

        for k in list(state_dict.keys()):
            if k.startswith('head'):
                del state_dict[k]
        return state_dict
    elif source=='mae':
        for k in list(state_dict.keys()):
            if k.startswith('patch_embed'):
                state_dict[k.replace('projection','proj')]=state_dict[k]
                del state_dict[k]
            elif k.startswith('layers'):
                layer_num=eval(k.split('.')[1])
                new_key='blocks'+k[len("layers"):]
                new_key=new_key.replace('.ln','.norm').replace('.ffn.layers.0.0.','.mlp.fc1.').replace('.ffn.layers.1','.mlp.fc2')
                state_dict[new_key]=state_dict[k]
                del state_dict[k]
            elif k.startswith('ln1'):
                state_dict[k.replace('ln1','norm')]=state_dict[k]
                del state_dict[k]
        return state_dict

