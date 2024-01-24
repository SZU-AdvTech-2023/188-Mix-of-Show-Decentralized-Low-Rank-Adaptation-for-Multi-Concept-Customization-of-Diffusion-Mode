import argparse
import copy
import itertools
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline
from tqdm import tqdm

from mixofshow.models.edlora import revise_edlora_unet_attention_forward
from mixofshow.pipelines.pipeline_edlora import bind_concept_prompt
from mixofshow.utils.util import set_logger

TEMPLATE_SIMPLE = 'photo of a {}'


def chunk_compute_mse(K_target, V_target, W, W1, device, chunk_size=5000):
    num_chunks = (K_target.size(0) + chunk_size - 1) // chunk_size

    loss = 0

    for i in range(num_chunks):
        # Extract the current chunk
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, K_target.size(0))
        loss += F.mse_loss(
            F.linear(K_target[start_idx:end_idx].to(device), W + W1),
            V_target[start_idx:end_idx].to(device)) * (end_idx - start_idx)
    loss /= K_target.size(0)
    return loss

def update_kv_ziplora(K_target, V_target, W, iters, device):
    '''
    Args:
        K: torch.Tensor, size [n_samples, n_features]
        V: torch.Tensor, size [n_samples, n_targets]
        K_target: torch.Tensor, size [n_constraints, n_features]
        V_target: torch.Tensor, size [n_constraints, n_targets]
        W: torch.Tensor, size [n_targets, n_features]

    Returns:
        Wnew: torch.Tensor, size [n_targets, n_features]
                    ziplora_set_forward_type(unet, type="merge")
    loss_1 = F.mse_loss(
        model_pred_mc.float(), model_pred_cc.float(), reduction="mean"
    )
    loss_2 = F.mse_loss(
        model_pred_ms.float(), model_pred_ss.float(), reduction="mean"
    )
    loss_3 = args.similarity_lambda * ziplora_compute_mergers_similarity(
        unet
    )
    loss = loss_1 + loss_2 + loss_3
    '''

    W1 = torch.zeros_like(W)
    W1 = W1.detach()

    W = W.detach()
    V_target = V_target.detach()
    K_target = K_target.detach()

    W.requires_grad = False
    W1.requires_grad = True
    K_target.requires_grad = False
    V_target.requires_grad = False

    best_loss = np.Inf
    best_W1 = None

    def closure():
        nonlocal best_W1, best_loss
        optimizer.zero_grad()

        if len(W.shape) == 4:
            loss = F.mse_loss(F.conv2d(K_target.to(device), W + W1),
                              V_target.to(device))
        else:
            loss = chunk_compute_mse(K_target, V_target, W, W1, device)

        if loss < best_loss:
            best_loss = loss
            best_W1 = W1.clone().cpu()
        loss.backward()
        return loss

    optimizer = optim.LBFGS([W1],
                            lr=1,
                            max_iter=iters,
                            history_size=25,
                            line_search_fn='strong_wolfe',
                            tolerance_grad=1e-16,
                            tolerance_change=1e-16)
    optimizer.step(closure)

    with torch.no_grad():
        if len(W.shape) == 4:
            loss = torch.norm(
                F.conv2d(K_target.to(torch.float32), best_W1.to(torch.float32)) - V_target.to(torch.float32), 2, dim=1)
        else:
            loss = torch.norm(
                F.linear(K_target.to(torch.float32), best_W1.to(torch.float32)) - V_target.to(torch.float32), 2, dim=1)

    logging.info('new_concept loss: %e' % loss.mean().item())
    return best_W1


def update_quasi_newton(K_target, V_target, W, iters, device):
    '''
    Args:
        K: torch.Tensor, size [n_samples, n_features]
        V: torch.Tensor, size [n_samples, n_targets]
        K_target: torch.Tensor, size [n_constraints, n_features]
        V_target: torch.Tensor, size [n_constraints, n_targets]
        W: torch.Tensor, size [n_targets, n_features]

    Returns:
        Wnew: torch.Tensor, size [n_targets, n_features]
    '''

    W1 = torch.zeros_like(W)
    W1 = W1.detach()

    W = W.detach()
    V_target = V_target.detach()
    K_target = K_target.detach()

    W.requires_grad = False
    W1.requires_grad = True
    K_target.requires_grad = False
    V_target.requires_grad = False

    best_loss = np.Inf
    best_W1 = None

    def closure():
        nonlocal best_W1, best_loss
        optimizer.zero_grad()

        if len(W.shape) == 4:
            loss = F.mse_loss(F.conv2d(K_target.to(device), W + W1),
                              V_target.to(device))
        else:
            loss = chunk_compute_mse(K_target, V_target, W, W1, device)

        if loss < best_loss:
            best_loss = loss
            best_W1 = W1.clone().cpu()
        loss.backward()
        return loss

    optimizer = optim.LBFGS([W1],
                            lr=1,
                            max_iter=iters,
                            history_size=25,
                            line_search_fn='strong_wolfe',
                            tolerance_grad=1e-16,
                            tolerance_change=1e-16)
    optimizer.step(closure)

    with torch.no_grad():
        if len(W.shape) == 4:
            loss = torch.norm(
                F.conv2d(K_target.to(torch.float32), best_W1.to(torch.float32)) - V_target.to(torch.float32), 2, dim=1)
        else:
            loss = torch.norm(
                F.linear(K_target.to(torch.float32), best_W1.to(torch.float32)) - V_target.to(torch.float32), 2, dim=1)

    logging.info('new_concept loss: %e' % loss.mean().item())
    return best_W1


def merge_lora_into_weight(original_state_dict, lora_state_dict, modification_layer_names, model_type, alpha, device):
    def get_lora_down_name(original_layer_name):
        if model_type == 'text_encoder':
            lora_down_name = original_layer_name.replace('q_proj.weight', 'q_proj.lora_down.weight') \
                .replace('k_proj.weight', 'k_proj.lora_down.weight') \
                .replace('v_proj.weight', 'v_proj.lora_down.weight') \
                .replace('out_proj.weight', 'out_proj.lora_down.weight') \
                .replace('fc1.weight', 'fc1.lora_down.weight') \
                .replace('fc2.weight', 'fc2.lora_down.weight')
        else:
            lora_down_name = k.replace('to_q.weight', 'to_q.lora_down.weight') \
                .replace('to_k.weight', 'to_k.lora_down.weight') \
                .replace('to_v.weight', 'to_v.lora_down.weight') \
                .replace('to_out.0.weight', 'to_out.0.lora_down.weight') \
                .replace('ff.net.0.proj.weight', 'ff.net.0.proj.lora_down.weight') \
                .replace('ff.net.2.weight', 'ff.net.2.lora_down.weight') \
                .replace('proj_out.weight', 'proj_out.lora_down.weight') \
                .replace('proj_in.weight', 'proj_in.lora_down.weight')

        return lora_down_name

    assert model_type in ['unet', 'text_encoder']
    new_state_dict = copy.deepcopy(original_state_dict)
    load_cnt = 0

    for k in modification_layer_names:
        lora_down_name = get_lora_down_name(k)
        lora_up_name = lora_down_name.replace('lora_down', 'lora_up')

        if lora_up_name in lora_state_dict:
            load_cnt += 1
            original_params = new_state_dict[k]
            lora_down_params = lora_state_dict[lora_down_name].to(device)
            lora_up_params = lora_state_dict[lora_up_name].to(device)
            if len(original_params.shape) == 4:
                lora_param = lora_up_params.squeeze(
                ) @ lora_down_params.squeeze()
                lora_param = lora_param.unsqueeze(-1).unsqueeze(-1)
            else:
                lora_param = lora_up_params @ lora_down_params
            merge_params = original_params + alpha * lora_param
            new_state_dict[k] = merge_params

    logging.info(f'load {load_cnt} LoRAs of {model_type}')
    return new_state_dict


module_io_recoder = {}
record_feature = False  # remember to set record feature


def get_hooker(module_name):
    def hook(module, feature_in, feature_out):
        if module_name not in module_io_recoder:
            module_io_recoder[module_name] = {'input': [], 'output': []}
        if record_feature:
            module_io_recoder[module_name]['input'].append(feature_in[0].cpu())
            if module.bias is not None:
                if len(feature_out.shape) == 4:
                    bias = module.bias.unsqueeze(-1).unsqueeze(-1)
                else:
                    bias = module.bias
                module_io_recoder[module_name]['output'].append(
                    (feature_out - bias).cpu())  # remove bias
            else:
                module_io_recoder[module_name]['output'].append(
                    feature_out.cpu())

    return hook


def init_stable_diffusion(pretrained_model_path, device):
    # step1: get w0 parameters
    model_id = pretrained_model_path
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    train_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder='scheduler')
    test_scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder='scheduler')
    pipe.safety_checker = None
    pipe.scheduler = test_scheduler
    return pipe, train_scheduler, test_scheduler


@torch.no_grad()
def get_text_feature(prompts, tokenizer, text_encoder, device, return_type='category_embedding'):
    text_features = []

    if return_type == 'category_embedding':
        for text in prompts:
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_length=True,
                return_overflowing_tokens=False,
                padding='do_not_pad',
            ).input_ids

            new_token_position = torch.where(torch.tensor(tokens) >= 49407)[0]
            # >40497 not include end token | >=40497 include end token
            concept_feature = text_encoder(
                torch.LongTensor(tokens).reshape(
                    1, -1).to(device))[0][:,
                              new_token_position].reshape(-1, 768)
            text_features.append(concept_feature)
        return torch.cat(text_features, 0).float()
    elif return_type == 'full_embedding':
        text_input = tokenizer(prompts,
                               padding='max_length',
                               max_length=tokenizer.model_max_length,
                               truncation=True,
                               return_tensors='pt')
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        return text_embeddings
    else:
        raise NotImplementedError


def merge_new_concepts_(embedding_list, concept_list, tokenizer, text_encoder):
    def add_new_concept(concept_name, embedding):
        new_token_names = [
            f'<new{start_idx + layer_id}>'
            for layer_id in range(NUM_CROSS_ATTENTION_LAYERS)
        ]
        num_added_tokens = tokenizer.add_tokens(new_token_names)
        assert num_added_tokens == NUM_CROSS_ATTENTION_LAYERS
        new_token_ids = [
            tokenizer.convert_tokens_to_ids(token_name)
            for token_name in new_token_names
        ]

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data

        token_embeds[new_token_ids] = token_embeds[new_token_ids].copy_(
            embedding[concept_name])

        embedding_features.update({concept_name: embedding[concept_name]})
        logging.info(
            f'concept {concept_name} is bind with token_id: [{min(new_token_ids)}, {max(new_token_ids)}]'
        )

        return start_idx + NUM_CROSS_ATTENTION_LAYERS, new_token_ids, new_token_names

    embedding_features = {}
    new_concept_cfg = {}

    start_idx = 0

    NUM_CROSS_ATTENTION_LAYERS = 16

    for idx, (embedding,
              concept) in enumerate(zip(embedding_list, concept_list)):
        concept_names = concept['concept_name'].split(' ')

        for concept_name in concept_names:
            if not concept_name.startswith('<'):
                continue
            else:
                assert concept_name in embedding, 'check the config, the provide concept name is not in the lora model'
            start_idx, new_token_ids, new_token_names = add_new_concept(
                concept_name, embedding)
            new_concept_cfg.update({
                concept_name: {
                    'concept_token_ids': new_token_ids,
                    'concept_token_names': new_token_names
                }
            })
    return embedding_features, new_concept_cfg

def merge_lora_weights(
    tensors: torch.Tensor, key: str, prefix: str = "unet.unet."
) -> Dict[str, torch.Tensor]:
    """
    Args:
        tensors (torch.Tensor): state dict of lora weights
        key (str): target attn layer's key
        prefix (str, optional): prefix for state dict. Defaults to "unet.unet.".
    """
    target_key = prefix + key
    out = {}
    for part in ["to_q", "to_k", "to_v", "to_out.0"]:
        down_key = target_key + f".{part}.lora.down.weight"
        up_key = target_key + f".{part}.lora.up.weight"
        merged_weight = tensors[up_key] @ tensors[down_key]
        out[part] = merged_weight
    return out

def parse_new_concepts(concept_cfg, unet):
    with open(concept_cfg, 'r') as f:
        concept_list = json.load(f)

    model_paths = [concept['lora_path'] for concept in concept_list]

    embedding_list = []
    text_encoder_list = []
    unet_crosskv_list = []
    unet_spatial_attn_list = []

    for model_path in model_paths:

        #修改
        i = 0
        lora_weights = []
        from safetensors import safe_open
        tensors = {}
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        lora_weights[i] = tensors
        i += 1
        #修改

        model = torch.load(model_path)['params']

        if 'new_concept_embedding' in model and len(
                model['new_concept_embedding']) != 0:
            embedding_list.append(model['new_concept_embedding'])
        else:
            embedding_list.append(None)

        if 'text_encoder' in model and len(model['text_encoder']) != 0:
            text_encoder_list.append(model['text_encoder'])
        else:
            text_encoder_list.append(None)

        if 'unet' in model and len(model['unet']) != 0:
            crosskv_matches = ['attn2.to_k.lora', 'attn2.to_v.lora']
            crosskv_dict = {
                k: v
                for k, v in model['unet'].items()
                if any([x in k for x in crosskv_matches])
            }

            if len(crosskv_dict) != 0:
                unet_crosskv_list.append(crosskv_dict)
            else:
                unet_crosskv_list.append(None)

            spatial_attn_dict = {
                k: v
                for k, v in model['unet'].items()
                if all([x not in k for x in crosskv_matches])
            }

            if len(spatial_attn_dict) != 0:
                unet_spatial_attn_list.append(spatial_attn_dict)
            else:
                unet_spatial_attn_list.append(None)
        else:
            unet_crosskv_list.append(None)
            unet_spatial_attn_list.append(None)

    #以下为修改
    unet_lora_parameters = []
    for attn_processor_name, attn_processor in unet.attn_processors.items():
        # Parse the attention module.
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)
        # Get prepared for ziplora
        attn_name = ".".join(attn_processor_name.split(".")[:-1])
        merged_lora_weights_dict = merge_lora_weights(lora_weights[0], attn_name)
        merged_lora_weights_dict_2 = merge_lora_weights(lora_weights[1], attn_name)
        kwargs = {
            "state_dict": merged_lora_weights_dict,
            "state_dict_2": merged_lora_weights_dict_2,
        }
        from ziplora import ZipLoRALinearLayer, ZipLoRALinearLayerInference

        def initialize_ziplora_layer(state_dict, state_dict_2, part, **model_kwargs):
            ziplora_layer = ZipLoRALinearLayer(**model_kwargs)
            ziplora_layer.load_state_dict(
                {
                    "weight_1": state_dict[part],
                    "weight_2": state_dict_2[part],
                },
                strict=False,
            )
            return ziplora_layer
    
        # Set the `lora_layer` attribute of the attention-related matrices.
        attn_module.to_q.set_lora_layer(
            initialize_ziplora_layer(
                part="to_q",
                in_features=attn_module.to_q.in_features,
                out_features=attn_module.to_q.out_features,
                init_merger_value=args.init_merger_value,
                init_merger_value_2=args.init_merger_value_2,
                **kwargs,
            )
        )
        attn_module.to_k.set_lora_layer(
            initialize_ziplora_layer(
                part="to_k",
                in_features=attn_module.to_k.in_features,
                out_features=attn_module.to_k.out_features,
                init_merger_value=args.init_merger_value,
                init_merger_value_2=args.init_merger_value_2,
                **kwargs,
            )
        )
        attn_module.to_v.set_lora_layer(
            initialize_ziplora_layer(
                part="to_v",
                in_features=attn_module.to_v.in_features,
                out_features=attn_module.to_v.out_features,
                init_merger_value=args.init_merger_value,
                init_merger_value_2=args.init_merger_value_2,
                **kwargs,
            )
        )
        attn_module.to_out[0].set_lora_layer(
            initialize_ziplora_layer(
                part="to_out.0",
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
                init_merger_value=args.init_merger_value,
                init_merger_value_2=args.init_merger_value_2,
                **kwargs,
            )
        )

        # Accumulate the LoRA params to optimize.
        unet_lora_parameters.extend(
            [p for p in attn_module.to_q.lora_layer.parameters() if p.requires_grad]
        )
        unet_lora_parameters.extend(
            [p for p in attn_module.to_k.lora_layer.parameters() if p.requires_grad]
        )
        unet_lora_parameters.extend(
            [p for p in attn_module.to_v.lora_layer.parameters() if p.requires_grad]
        )
        unet_lora_parameters.extend(
            [
                p
                for p in attn_module.to_out[0].lora_layer.parameters()
                if p.requires_grad
            ]
        )
    #以上为修改

    return embedding_list, text_encoder_list, unet_crosskv_list, unet_spatial_attn_list, concept_list, unet_lora_parameters

#新增
def merge_in_ziplora():
    optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
#新增

def merge_kv_in_cross_attention(concept_list, optimize_iters, new_concept_cfg,
                                tokenizer, text_encoder, unet, 
                                unet_crosskv_list, device):
    # crosskv attention layer names
    matches = ['attn2.to_k', 'attn2.to_v']

    cross_attention_idx = -1
    cross_kv_layer_names = []

    # the crosskv name should match the order down->mid->up, and record its layer id
    for name, _ in unet.down_blocks.named_parameters():
        if any([x in name for x in matches]):
            if 'to_k' in name:
                cross_attention_idx += 1
                cross_kv_layer_names.append(
                    (cross_attention_idx, 'down_blocks.' + name))
                cross_kv_layer_names.append(
                    (cross_attention_idx,
                     'down_blocks.' + name.replace('to_k', 'to_v')))
            else:
                pass

    for name, _ in unet.mid_block.named_parameters():
        if any([x in name for x in matches]):
            if 'to_k' in name:
                cross_attention_idx += 1
                cross_kv_layer_names.append(
                    (cross_attention_idx, 'mid_block.' + name))
                cross_kv_layer_names.append(
                    (cross_attention_idx,
                     'mid_block.' + name.replace('to_k', 'to_v')))
            else:
                pass

    for name, _ in unet.up_blocks.named_parameters():
        if any([x in name for x in matches]):
            if 'to_k' in name:
                cross_attention_idx += 1
                cross_kv_layer_names.append(
                    (cross_attention_idx, 'up_blocks.' + name))
                cross_kv_layer_names.append(
                    (cross_attention_idx,
                     'up_blocks.' + name.replace('to_k', 'to_v')))
            else:
                pass

    logging.info(
        f'Unet have {len(cross_kv_layer_names)} linear layer (related to text feature) need to optimize'
    )

    original_unet_state_dict = unet.state_dict()  # original state dict

    new_concept_input_dict = {}
    new_concept_output_dict = {}

    # step 1: construct prompts for new concept -> extract input/target features
    for concept, tuned_state_dict in zip(concept_list, unet_crosskv_list):
        concept_prompt = [
            TEMPLATE_SIMPLE.format(concept['concept_name']),
            concept['concept_name']
        ]
        concept_prompt = bind_concept_prompt(concept_prompt, new_concept_cfg)

        n = len(concept_prompt) // 16
        layer_prompts = [
            tuple(concept_prompt[j * 16 + i] for j in range(n))
            for i in range(16)
        ]

        for layer_idx, layer_name in cross_kv_layer_names:

            # merge params
            original_params = original_unet_state_dict[layer_name]

            # hard coded here: in unet, self/crosskv attention disable bias parameter
            lora_down_name = layer_name.replace('to_k.weight', 'to_k.lora_down.weight').replace('to_v.weight', 'to_v.lora_down.weight')
            lora_up_name = lora_down_name.replace('lora_down', 'lora_up')

            alpha = concept['unet_alpha']

            lora_down_params = tuned_state_dict[lora_down_name].to(device)
            lora_up_params = tuned_state_dict[lora_up_name].to(device)

            merge_params = original_params + alpha * lora_up_params @ lora_down_params

            layer_concept_prompt = list(layer_prompts[layer_idx])

            prompt_feature = get_text_feature(
                layer_concept_prompt,
                tokenizer,
                text_encoder,
                device,
                return_type='category_embedding').cpu()

            if layer_name not in new_concept_input_dict:
                new_concept_input_dict[layer_name] = []

            if layer_name not in new_concept_output_dict:
                new_concept_output_dict[layer_name] = []

            # print(merge_params.shape, prompt_feature.shape)
            # torch.Size([320, 768]) torch.Size([6, 768])
            new_concept_input_dict[layer_name].append(prompt_feature)
            new_concept_output_dict[layer_name].append(
                (merge_params.cpu() @ prompt_feature.T).T)

    for k, v in new_concept_input_dict.items():
        new_concept_input_dict[k] = torch.cat(v, 0)  # torch.Size([14, 768])

    for k, v in new_concept_output_dict.items():
        new_concept_output_dict[k] = torch.cat(v, 0)  # torch.Size([14, 768])

    new_kv_weights = {}
    # step 3: begin update model
    for idx, (layer_idx, layer_name) in enumerate(cross_kv_layer_names):
        W = original_unet_state_dict[layer_name].to(torch.float32)  # origin params

        new_concept_input = new_concept_input_dict[layer_name]
        new_concept_output = new_concept_output_dict[layer_name]

        logging.info(
            f'[{(idx + 1)}/{len(cross_kv_layer_names)}] optimizing {layer_name}')

        Wnew = update_quasi_newton(
            new_concept_input.to(W.dtype),  # our concept
            new_concept_output.to(W.dtype),  # our concept
            W.clone(),
            iters=optimize_iters,
            device=device)

        new_kv_weights[layer_name] = Wnew + W

    return new_kv_weights


def merge_text_encoder(concept_list, optimize_iters, new_concept_cfg,
                       tokenizer, text_encoder, text_encoder_list, device):
    def process_extract_features(input_feature_list, output_feature_list):
        text_input_features = [
            feat.reshape(-1, feat.shape[-1]) for feat in input_feature_list
        ]
        text_output_features = [
            feat.reshape(-1, feat.shape[-1]) for feat in output_feature_list
        ]
        text_input_features = torch.cat(text_input_features, 0)
        text_output_features = torch.cat(text_output_features, 0)
        return text_input_features, text_output_features

    LoRA_keys = []
    for textenc_lora in text_encoder_list:
        LoRA_keys += list(textenc_lora.keys())
    LoRA_keys = set([
        key.replace('.lora_down', '').replace('.lora_up', '')
        for key in LoRA_keys
    ])
    text_encoder_layer_names = LoRA_keys

    candidate_module_name = [
        'q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'
    ]
    candidate_module_name = [
        name for name in candidate_module_name
        if any([name in key for key in LoRA_keys])
    ]

    logging.info(f'text_encoder have {len(text_encoder_layer_names)} linear layer need to optimize')

    global module_io_recoder, record_feature
    hooker_handlers = []
    for name, module in text_encoder.named_modules():
        if any([item in name for item in candidate_module_name]):
            hooker_handlers.append(module.register_forward_hook(hook=get_hooker(name)))

    logging.info(f'add {len(hooker_handlers)} hooker to text_encoder')

    original_state_dict = copy.deepcopy(text_encoder.state_dict())  # original state dict

    new_concept_input_dict = {}
    new_concept_output_dict = {}

    for concept, lora_state_dict in zip(concept_list, text_encoder_list):

        merged_state_dict = merge_lora_into_weight(
            original_state_dict,
            lora_state_dict,
            text_encoder_layer_names,
            model_type='text_encoder',
            alpha=concept['text_encoder_alpha'],
            device=device)
        text_encoder.load_state_dict(merged_state_dict)  # load merged parameters

        concept_prompt = [
            TEMPLATE_SIMPLE.format(concept['concept_name']),
            concept['concept_name']
        ]
        concept_prompt = bind_concept_prompt(concept_prompt, new_concept_cfg)

        # reinit module io recorder
        module_io_recoder = {}
        record_feature = True
        _ = get_text_feature(concept_prompt,
                             tokenizer,
                             text_encoder,
                             device,
                             return_type='category_embedding')

        # we use different model to compute new concept feature
        for layer_name in text_encoder_layer_names:
            input_feature_list = module_io_recoder[layer_name.replace('.weight', '')]['input']
            output_feature_list = module_io_recoder[layer_name.replace('.weight', '')]['output']

            text_input_features, text_output_features = \
                process_extract_features(input_feature_list, output_feature_list)

            if layer_name not in new_concept_output_dict:
                new_concept_input_dict[layer_name] = []
                new_concept_output_dict[layer_name] = []

            new_concept_input_dict[layer_name].append(text_input_features)
            new_concept_output_dict[layer_name].append(text_output_features)

    for k, v in new_concept_input_dict.items():
        new_concept_input_dict[k] = torch.cat(v, 0)  # torch.Size([14, 768])

    for k, v in new_concept_output_dict.items():
        new_concept_output_dict[k] = torch.cat(v, 0)  # torch.Size([14, 768])

    new_text_encoder_weights = {}
    # step 3: begin update model
    for idx, layer_name in enumerate(text_encoder_layer_names):
        W = original_state_dict[layer_name].to(torch.float32)

        new_concept_input = new_concept_input_dict[layer_name]
        new_concept_target = new_concept_output_dict[layer_name]

        logging.info(f'[{(idx + 1)}/{len(text_encoder_layer_names)}] optimizing {layer_name}')

        Wnew = update_quasi_newton(
            new_concept_input.to(W.dtype),  # our concept
            new_concept_target.to(W.dtype),  # our concept
            W.clone(),
            iters=optimize_iters,
            device=device)
        new_text_encoder_weights[layer_name] = Wnew + W

    logging.info(f'remove {len(hooker_handlers)} hooker from text_encoder')

    # remove forward hooker
    for hook_handle in hooker_handlers:
        hook_handle.remove()

    return new_text_encoder_weights


@torch.no_grad()
def decode_to_latents(concept_prompt, new_concept_cfg, tokenizer, text_encoder,
                      unet, test_scheduler, num_inference_steps, device,
                      record_nums, batch_size):

    concept_prompt = bind_concept_prompt([concept_prompt], new_concept_cfg)
    text_embeddings = get_text_feature(
        concept_prompt,
        tokenizer,
        text_encoder,
        device,
        return_type='full_embedding').unsqueeze(0)

    text_embeddings = text_embeddings.repeat((batch_size, 1, 1, 1))

    # sd 1.x
    height = 512
    width = 512

    latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8), )
    latents = latents.to(device, dtype=text_embeddings.dtype)

    test_scheduler.set_timesteps(num_inference_steps)
    latents = latents * test_scheduler.init_noise_sigma

    global record_feature
    step = (test_scheduler.timesteps.size(0)) // record_nums
    record_timestep = test_scheduler.timesteps[torch.arange(0, test_scheduler.timesteps.size(0), step=step)[:record_nums]]

    for t in tqdm(test_scheduler.timesteps):

        if t in record_timestep:
            record_feature = True
        else:
            record_feature = False

        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = latents
        latent_model_input = test_scheduler.scale_model_input(latent_model_input, t)

        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # compute the previous noisy sample x_t -> x_t-1
        latents = test_scheduler.step(noise_pred, t, latents).prev_sample

    return latents, text_embeddings


def merge_spatial_attention(concept_list, optimize_iters, new_concept_cfg, tokenizer, text_encoder, unet, unet_spatial_attn_list, test_scheduler, device):
    LoRA_keys = []
    for unet_lora in unet_spatial_attn_list:
        LoRA_keys += list(unet_lora.keys())
    LoRA_keys = set([
        key.replace('.lora_down', '').replace('.lora_up', '')
        for key in LoRA_keys
    ])
    spatial_attention_layer_names = LoRA_keys

    candidate_module_name = [
        'attn2.to_q', 'attn2.to_out.0', 'attn1.to_q', 'attn1.to_k',
        'attn1.to_v', 'attn1.to_out.0', 'ff.net.2', 'ff.net.0.proj',
        'proj_out', 'proj_in'
    ]
    candidate_module_name = [
        name for name in candidate_module_name
        if any([name in key for key in LoRA_keys])
    ]

    logging.info(
        f'unet have {len(spatial_attention_layer_names)} linear layer need to optimize'
    )

    global module_io_recoder
    hooker_handlers = []
    for name, module in unet.named_modules():
        if any([x in name for x in candidate_module_name]):
            hooker_handlers.append(
                module.register_forward_hook(hook=get_hooker(name)))

    logging.info(f'add {len(hooker_handlers)} hooker to unet')

    original_state_dict = copy.deepcopy(unet.state_dict())  # original state dict
    revise_edlora_unet_attention_forward(unet)

    new_concept_input_dict = {}
    new_concept_output_dict = {}

    for concept, tuned_state_dict in zip(concept_list, unet_spatial_attn_list):
        # set unet
        module_io_recoder = {}  # reinit module io recorder

        merged_state_dict = merge_lora_into_weight(
            original_state_dict,
            tuned_state_dict,
            spatial_attention_layer_names,
            model_type='unet',
            alpha=concept['unet_alpha'],
            device=device)
        unet.load_state_dict(merged_state_dict)  # load merged parameters

        concept_name = concept['concept_name']
        concept_prompt = TEMPLATE_SIMPLE.format(concept_name)

        decode_to_latents(concept_prompt,
                          new_concept_cfg,
                          tokenizer,
                          text_encoder,
                          unet,
                          test_scheduler,
                          num_inference_steps=20,
                          device=device,
                          record_nums=20,
                          batch_size=1)
        # record record_num * batch size feature for one concept

        for layer_name in spatial_attention_layer_names:
            input_feature_list = module_io_recoder[layer_name.replace('.weight', '')]['input']
            output_feature_list = module_io_recoder[layer_name.replace('.weight', '')]['output']

            text_input_features, text_output_features = \
                torch.cat(input_feature_list, 0), torch.cat(output_feature_list, 0)

            if layer_name not in new_concept_output_dict:
                new_concept_input_dict[layer_name] = []
                new_concept_output_dict[layer_name] = []

            new_concept_input_dict[layer_name].append(text_input_features)
            new_concept_output_dict[layer_name].append(text_output_features)

    for k, v in new_concept_input_dict.items():
        new_concept_input_dict[k] = torch.cat(v, 0)

    for k, v in new_concept_output_dict.items():
        new_concept_output_dict[k] = torch.cat(v, 0)

    new_spatial_attention_weights = {}

    # step 5: begin update model
    for idx, layer_name in enumerate(spatial_attention_layer_names):
        new_concept_input = new_concept_input_dict[layer_name]
        if len(new_concept_input.shape) == 4:
            new_concept_input = new_concept_input
        else:
            new_concept_input = new_concept_input.reshape(-1, new_concept_input.shape[-1])

        new_concept_output = new_concept_output_dict[layer_name]
        if len(new_concept_output.shape) == 4:
            new_concept_output = new_concept_output
        else:
            new_concept_output = new_concept_output.reshape(-1, new_concept_output.shape[-1])

        logging.info(f'[{(idx + 1)}/{len(spatial_attention_layer_names)}] optimizing {layer_name}')

        W = original_state_dict[layer_name].to(torch.float32)  # origin params
        Wnew = update_quasi_newton(
            new_concept_input.to(W.dtype),  # our concept
            new_concept_output.to(W.dtype),  # our concept
            W.clone(),
            iters=optimize_iters,
            device=device,
        )
        new_spatial_attention_weights[layer_name] = Wnew + W

    logging.info(f'remove {len(hooker_handlers)} hooker from unet')

    for hook_handle in hooker_handlers:
        hook_handle.remove()

    return new_spatial_attention_weights


def compose_concepts(concept_cfg, optimize_textenc_iters, optimize_unet_iters, pretrained_model_path, save_path, suffix, device):
    logging.info('------Step 1: load stable diffusion checkpoint------')
    pipe, train_scheduler, test_scheduler = init_stable_diffusion(pretrained_model_path, device)
    tokenizer, text_encoder, unet, vae = pipe.tokenizer, pipe.text_encoder, pipe.unet, pipe.vae
    for param in itertools.chain(text_encoder.parameters(), unet.parameters(), vae.parameters()):
        param.requires_grad = False

    logging.info('------Step 2: load new concepts checkpoints------')
    embedding_list, text_encoder_list, unet_crosskv_list, unet_spatial_attn_list, concept_list, unet_lora_parameters = parse_new_concepts(concept_cfg, unet)

    # step 1: inplace add new concept to tokenizer and embedding layers of text encoder
    if any([item is not None for item in embedding_list]):
        logging.info('------Step 3: merge token embedding------')
        _, new_concept_cfg = merge_new_concepts_(embedding_list, concept_list, tokenizer, text_encoder)
    else:
        _, new_concept_cfg = {}, {}
        logging.info('------Step 3: no new embedding, skip merging token embedding------')

    # step 2: construct reparameterized text_encoder
    if any([item is not None for item in text_encoder_list]):
        logging.info('------Step 4: merge text encoder------')
        new_text_encoder_weights = merge_text_encoder(
            concept_list, optimize_textenc_iters, new_concept_cfg, tokenizer,
            text_encoder, text_encoder_list, device)
        # update the merged state_dict in text_encoder
        text_encoder_state_dict = text_encoder.state_dict()
        text_encoder_state_dict.update(new_text_encoder_weights)
        text_encoder.load_state_dict(text_encoder_state_dict)
    else:
        new_text_encoder_weights = {}
        logging.info('------Step 4: no new text encoder, skip merging text encoder------')

    # step 3: merge unet (k,v in crosskv-attention) params, since they only receive input from text-encoder

    if any([item is not None for item in unet_crosskv_list]):
        logging.info('------Step 5: merge kv of cross-attention in unet------')
        new_kv_weights = merge_kv_in_cross_attention(
            concept_list, optimize_textenc_iters, new_concept_cfg,
            tokenizer, text_encoder, unet, unet_crosskv_list, device)
        # update the merged state_dict in kv of crosskv-attention in Unet
        unet_state_dict = unet.state_dict()
        unet_state_dict.update(new_kv_weights)
        unet.load_state_dict(unet_state_dict)
    else:
        new_kv_weights = {}
        logging.info('------Step 5: no new kv of cross-attention in unet, skip merging kv------')

    #修改
    #定义数据集类
    from torchvision import transforms
    from torch.utils.data import Dataset
    from pathlib import Path
    from PIL import Image
    def prepare_instance_images(instance_data_root: str, repeats: int):
        instance_data_root = Path(instance_data_root)
        if not instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        instance_images = [
            Image.open(path) for path in list(Path(instance_data_root).iterdir())
        ]

        res = []
        for img in instance_images:
            res.extend(itertools.repeat(img, repeats))
        return res
    class DreamBoothDataset(Dataset):
        """
        A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
        It pre-processes the images.
        """

        def __init__(
            self,
            instance_data_root,
            instance_prompt,
            instance_data_root_2,
            instance_prompt_2,
            size=1024,
            repeats=1,
            center_crop=False,
        ):
            self.size = size
            self.center_crop = center_crop

            self.instance_prompt = instance_prompt
            self.instance_prompt_2 = instance_prompt_2

            # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
            # we load the training data using load_dataset
            if args.dataset_name is not None:
                raise NotImplementedError
            self.instance_images = prepare_instance_images(instance_data_root, repeats)
            self.instance_images_2 = prepare_instance_images(instance_data_root_2, repeats)
            self.num_instance_images = max(
                len(self.instance_images), len(self.instance_images_2)
            )
            if len(self.instance_images) != self.num_instance_images:
                # repeat
                self.instance_images = self.instance_images * math.ceil(
                    self.num_instance_images / len(self.instance_images)
                )
                self.instance_images = self.instance_images[: self.num_instance_images]

            if len(self.instance_images_2) != self.num_instance_images:
                # repeat
                self.instance_images_2 = self.instance_images_2 * math.ceil(
                    self.num_instance_images / len(self.instance_images_2)
                )
                self.instance_images_2 = self.instance_images_2[: self.num_instance_images]
            self._length = self.num_instance_images
            self.image_transforms = transforms.Compose(
                [
                    transforms.Resize(
                        size, interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.CenterCrop(size)
                    if center_crop
                    else transforms.RandomCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

        def __len__(self):
            return self._length

        def _transform_image(self, image):
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            return self.image_transforms(image)

        def __getitem__(self, index):
            example = {}
            instance_image = self.instance_images[index % self.num_instance_images]
            example["instance_images"] = self._transform_image(instance_image)
            example["instance_prompt"] = self.instance_prompt
            instance_image_2 = self.instance_images_2[index % self.num_instance_images]
            example["instance_images_2"] = self._transform_image(instance_image_2)
            example["instance_prompt_2"] = self.instance_prompt_2
            return example
    # 准备instance images数据集:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        instance_data_root_2=args.instance_data_dir_2,
        instance_prompt_2=args.instance_prompt_2,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
    )

    #创建优化器
    unet_lora_parameters_with_lr = {
        "params": unet_lora_parameters,
        "lr": args.learning_rate,
    }
    params_to_optimize = [unet_lora_parameters_with_lr]
    logger = get_logger(__name__)
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warn(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warn(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
        
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError(
                "To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`"
            )

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warn(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warn(
                f"Learning rates were provided both for the unet and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one and text_encoder_parameters_two to be
            # --learning_rate
            params_to_optimize[1]["lr"] = args.learning_rate
            params_to_optimize[2]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Scheduler and math around the number of training steps.（意义暂时不明？)
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )
    #将以上准备放入acceleratar
    from accelerate import Accelerator
    from accelerate.logging import get_logger
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        ProjectConfiguration,
        set_seed,
    )
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    import math
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # 有疑问
    if accelerator.is_main_process:
        accelerator.init_trackers("mixofshow-sd", config=vars(args))

    #进度条
    initial_global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
    def tokenize_prompt(tokenizer, prompt):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        return text_input_ids
    def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
        prompt_embeds_list = []

        for i, text_encoder in enumerate(text_encoders):
            if tokenizers is not None:
                tokenizer = tokenizers[i]
                text_input_ids = tokenize_prompt(tokenizer, prompt)
            else:
                assert text_input_ids_list is not None
                text_input_ids = text_input_ids_list[i]

            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds
    def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoders, tokenizers, prompt
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds
    if not args.train_text_encoder:
        (
            instance_prompt_hidden_states,
            instance_pooled_prompt_embeds,
        ) = compute_text_embeddings(args.instance_prompt, text_encoders, tokenizers)
        (
            instance_prompt_hidden_states_2,
            instance_pooled_prompt_embeds_2,
        ) = compute_text_embeddings(args.instance_prompt_2, text_encoders, tokenizers)
    # Clear the memory here
    import gc
    if not args.train_text_encoder:
        del tokenizers, text_encoders
        gc.collect()
        torch.cuda.empty_cache()
    prompt_embeds = instance_prompt_hidden_states
    unet_add_text_embeds = instance_pooled_prompt_embeds
    prompt_embeds_2 = instance_prompt_hidden_states_2
    unet_add_text_embeds_2 = instance_pooled_prompt_embeds_2

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                model_inputs = []
                for i in range(2):
                    pixel_values_key = "pixel_values" if i == 0 else "pixel_values_2"
                    prompts_key = "prompts" if i == 0 else "prompts_2"
                    pixel_values = batch[pixel_values_key].to(dtype=vae.dtype)
                    prompts = batch[prompts_key]
                    # Convert images to latent space
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = model_input * vae.config.scaling_factor
                    if args.pretrained_vae_model_name_or_path is None:
                        model_input = model_input.to(weight_dtype)
                    model_inputs.append(model_input)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    train_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=model_input.device,
                )
                timesteps = timesteps.long()
                # Predict the noise residual
                unet_added_conditions = {
                    "time_ids": add_time_ids.repeat(bsz, 1),
                    "text_embeds": unet_add_text_embeds.repeat(bsz, 1),
                }
                prompt_embeds_input = prompt_embeds.repeat(bsz, 1, 1)

                unet_added_conditions_2 = {
                    "time_ids": unet_added_conditions["time_ids"],
                    "text_embeds": unet_add_text_embeds_2.repeat(bsz, 1),
                }
                prompt_embeds_input_2 = prompt_embeds_2.repeat(bsz, 1, 1)

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                for i in range(2):
                    model_inputs[i] = train_scheduler.add_noise(
                        model_inputs[i], noise, timesteps
                    )

                # 1. merged weights + concept
                model_pred_mc = unet(
                    model_inputs[0],
                    timesteps,
                    prompt_embeds_input,
                    added_cond_kwargs=unet_added_conditions,
                ).sample
                # 2. merged weights + style
                model_pred_ms = unet(
                    model_inputs[1],
                    timesteps,
                    prompt_embeds_input_2,
                    added_cond_kwargs=unet_added_conditions_2,
                ).sample

                # 3. concept weights + concept
                ziplora_set_forward_type(unet, type="weight_1")
                with torch.no_grad():
                    model_pred_cc = unet(
                        model_inputs[0],
                        timesteps,
                        prompt_embeds_input,
                        added_cond_kwargs=unet_added_conditions,
                    ).sample
                # 4. style weights + style
                ziplora_set_forward_type(unet, type="weight_2")
                with torch.no_grad():
                    model_pred_ss = unet(
                        model_inputs[1],
                        timesteps,
                        prompt_embeds_input_2,
                        added_cond_kwargs=unet_added_conditions_2,
                    ).sample
                # compute losses
                from diffusers import UNet2DConditionModel
                def ziplora_set_forward_type(unet: UNet2DConditionModel, type: str = "merge"):
                    assert type in ["merge", "weight_1", "weight_2"]

                    for name, module in unet.named_modules():
                        if hasattr(module, "set_lora_layer"):
                            lora_layer = getattr(module, "lora_layer")
                            if lora_layer is not None:
                                assert hasattr(lora_layer, "set_forward_type"), lora_layer
                                lora_layer.set_forward_type(type)
                    return unet
                ziplora_set_forward_type(unet, type="merge")
                loss_1 = F.mse_loss(
                    model_pred_mc.float(), model_pred_cc.float(), reduction="mean"
                )
                loss_2 = F.mse_loss(
                    model_pred_ms.float(), model_pred_ss.float(), reduction="mean"
                )
                def ziplora_compute_mergers_similarity(unet):
                    similarities = []
                    for name, module in unet.named_modules():
                        if hasattr(module, "set_lora_layer"):
                            lora_layer = getattr(module, "lora_layer")
                            if lora_layer is not None:
                                assert hasattr(lora_layer, "compute_mergers_similarity"), lora_layer
                                similarities.append(lora_layer.compute_mergers_similarity())
                    similarity = torch.stack(similarities).sum(dim=0)
                    return similarity
                loss_3 = args.similarity_lambda * ziplora_compute_mergers_similarity(
                    unet
                )
                loss = loss_1 + loss_2 + loss_3

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        unet_lora_parameters, args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    #每checkpointing_steps保存一次checkpoint
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "loss": loss.detach().item(),
                "loss_1": loss_1.detach().item(),
                "loss_2": loss_2.detach().item(),
                "loss_3": loss_3.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps or loss_3.detach().item() == 0.0:
                break

    #修改

    # step 4: merge unet (q,k,v in self-attention, q in crosskv-attention)
    if any([item is not None for item in unet_spatial_attn_list]):
        logging.info('------Step 6: merge spatial attention (q in cross-attention, qkv in self-attention) in unet------')
        new_spatial_attention_weights = merge_spatial_attention(
            concept_list, optimize_unet_iters, new_concept_cfg, tokenizer,
            text_encoder, unet, unet_spatial_attn_list, test_scheduler, device)
        unet_state_dict = unet.state_dict()
        unet_state_dict.update(new_spatial_attention_weights)
        unet.load_state_dict(unet_state_dict)
    else:
        new_spatial_attention_weights = {}
        logging.info('------Step 6: no new spatial-attention in unet, skip merging spatial attention------')

    checkpoint_save_path = f'{save_path}/combined_model_{suffix}'
    pipe.save_pretrained(checkpoint_save_path)
    with open(os.path.join(checkpoint_save_path, 'new_concept_cfg.json'), 'w') as json_file:
        json.dump(new_concept_cfg, json_file)


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--concept_cfg', help='json file for multi-concept', required=True, type=str)
    parser.add_argument('--save_path', help='folder name to save optimized weights', required=True, type=str)
    parser.add_argument('--suffix', help='suffix name', default='base', type=str)
    parser.add_argument('--pretrained_models', required=True, type=str)
    parser.add_argument('--optimize_unet_iters', default=50, type=int)
    parser.add_argument('--optimize_textenc_iters', default=500, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # s1: set logger
    exp_dir = f'{args.save_path}'
    os.makedirs(exp_dir, exist_ok=True)
    log_file = f'{exp_dir}/combined_model_{args.suffix}.log'
    set_logger(log_file=log_file)
    logging.info(args)

    compose_concepts(args.concept_cfg,
                     args.optimize_textenc_iters,
                     args.optimize_unet_iters,
                     args.pretrained_models,
                     args.save_path,
                     args.suffix,
                     device='cuda')
