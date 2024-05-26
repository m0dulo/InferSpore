import argparse
import configparser
import os
from pathlib import Path
import numpy as np
import torch

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"

if __name__ == "__main__":     
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-saved_dir', '-o', type=str, help='file name of output file', required=True)
    parser.add_argument('-in_file', '-i', type=str, help='file name of input checkpoint file', required=True)
    # parser.add_argument('-trained_gpu_num', '-t_g', type=int, help='How many gpus for inference', default=1)
    # parser.add_argument('-infer_gpu_num', '-i_g', type=int, help='How many gpus for inference', required=True)
    # parser.add_argument("-processes", "-p", type=int, help="How many processes to spawn for conversion (default: 4)", default=4)
    parser.add_argument("-weight_data_type", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--load-model-on-cpu", action="store_true")
    parser.add_argument("--convert-model-on-cpu", action="store_true")
    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))#vars(args)返回args的字典形式
    print("========================================")
    torch_dtype = torch.float16 if args.weight_data_type == 'fp16' else torch.float32
    model = LlamaForCausalLM.from_pretrained(args.in_file,
                                                torch_dtype=torch_dtype,
                                                device_map="auto",
                                                trust_remote_code=True)
    if args.load_model_on_cpu:
        model = model.float()
        model = model.cpu()
        torch.cuda.empty_cache()
        
    saved_dir = args.saved_dir + "/1-gpu"
    hf_config = vars(model.config)
    print("\n=============== HF model config ===============")
    print(hf_config)
    print("\n===============================================")
    # 根据config来写入config.ini
    import pdb;pdb.set_trace()
    config = configparser.ConfigParser()
    config["llama"] = {}
    config["llama"]["model_name"] = "llama-2-7b-chat" if hf_config["_name_or_path"] == '' else hf_config["_name_or_path"]
    config["llama"]["head_num"] = str(hf_config["num_attention_heads"])
    config["llama"]["kv_head_num"] = str(hf_config["num_key_value_heads"])
    config["llama"]["hidden_size"] = str(hf_config["hidden_size"])
    config["llama"]["head_size"] = str(hf_config["hidden_size"] // hf_config["num_attention_heads"])
    config["llama"]["inter_size"] = str(hf_config["intermediate_size"]) #11008
    #config['llama']['max_pos_seq_len'] = str(hf_config['n_positions'])
    config["llama"]["num_layer"] = str(hf_config["num_hidden_layers"])
    config["llama"]["vocab_size"] = str(hf_config["vocab_size"]) #32000
    config["llama"]["bos_token_id"] = str(hf_config["bos_token_id"]) #1
    config["llama"]["eos_token_id"] = str(hf_config["eos_token_id"]) #2
    config['llama']['weight_data_type'] = args.weight_data_type
    config['llama']['max_position_embeddings'] = str(hf_config["max_position_embeddings"])
    config['llama']['rope_theta'] = str(hf_config["rope_theta"])
    config['llama']['rms_norm_eps'] = str(hf_config["rms_norm_eps"])
    config['llama']['attention_bias'] = str(hf_config["attention_bias"]) #false
    config['llama']['top_k'] = str(hf_config["top_k"])#50
    with open(saved_dir + "/config.ini", 'w') as configfile:
        config.write(configfile)
    # except:
    #     print(f"Fail to save the config in config.ini.")
    np_weight_data_type = get_weight_data_type(args.weight_data_type)
    cur_layer = 0
    q = 0
    k = 0
    for name, param in model.named_parameters():
        # model.embed_tokens.weight [32000, 4096]
        # import pdb;pdb.set_trace()
        # if name.find("weight") == -1 and name.find("bias") == -1:
        #     continue
#        import pdb;pdb.set_trace()
        if name.find('model.embed_tokens.weight') != -1:
            param.detach().cpu().float().numpy().astype(np_weight_data_type).tofile(f"model.embed_tokens.weight.bin")
        elif name.find('model.norm.weight') != -1:
            param.detach().cpu().float().numpy().astype(np_weight_data_type).tofile(f"model.norm.weight.bin")
        elif name.find('lm_head.weight') != -1:
            param.detach().cpu().float().numpy().astype(np_weight_data_type).tofile(f"lm_head.weight.bin")
        elif name.find('self_attn.q_proj.weight') != -1 or name.find('self_attn.k_proj.weight') != -1 or name.find('self_attn.v_proj.weight') != -1:
            layer = name.split(".")[2]
            if name.find('self_attn.q_proj.weight') != -1:
                q = param.detach().cpu().float().numpy()
            elif name.find('self_attn.k_proj.weight') != -1:
                k = param.detach().cpu().float().numpy()
            elif name.find('self_attn.v_proj.weight') != -1:
                v = param.detach().cpu().float().numpy()
                qkv = np.hstack((q, k, v))
                qkv.astype(np_weight_data_type).tofile(f"model.layers.{layer}.self_attn.qkv.weight.bin")
                print("qkv shape: ", qkv.shape)
            # if cur_layer == layer:
            #     np.concat(param.detach().cpu().float().numpy())
            # else:
            #     cur_layer = layer
                
        elif name.find('self_attn.o_proj.weight') != -1:
            layer = name.split(".")[2]
            param.detach().cpu().float().numpy().astype(np_weight_data_type).tofile(f"model.layers.{layer}.self_attn.o_proj.weight.bin")
        
        elif name.find('mlp.gate_proj.weight') != -1 or name.find('mlp.up_proj.weight') != -1:
            layer = name.split(".")[2]
            if name.find('mlp.gate_proj.weight') != -1:
                gate = param.detach().cpu().float().numpy()
            elif name.find('mlp.up_proj.weight') != -1:
                up = param.detach().cpu().float().numpy()
                gate_up = np.hstack((gate, up))
                gate_up.astype(np_weight_data_type).tofile(f"model.layers.{layer}.mlp.gate_up_proj.weight.bin")
                print("fused gate_up shape: ", gate_up.shape)
        # elif name.find('mlp.up_proj.weight') != -1:
        #     layer = name.split(".")[2]
        #     param.detach().cpu().float().numpy().astype(np_weight_data_type).tofile(f"model.layers.{layer}.mlp.up_proj.weight.bin")
        elif name.find('mlp.down_proj.weight') != -1:
            layer = name.split(".")[2]
            param.detach().cpu().float().numpy().astype(np_weight_data_type).tofile(f"model.layers.{layer}.mlp.down_proj.weight.bin")
        elif name.find('input_layernorm.weight') != -1:
            layer = name.split(".")[2]
            param.detach().cpu().float().numpy().astype(np_weight_data_type).tofile(f"model.layers.{layer}.input_layernorm.weight.bin")
        elif name.find('post_attention_layernorm.weight') != -1:
            layer = name.split(".")[2]
            param.detach().cpu().float().numpy().astype(np_weight_data_type).tofile(f"model.layers.{layer}.post_attention_layernorm.weight.bin")

        # else:
            
        #     for i in range(len(huggingface_model_name_pattern)):
        #         if name.find(huggingface_model_name_pattern[i]) != -1:
        #             new_name = name.replace("h.", "layers.").replace(huggingface_model_name_pattern[i], ft_model_name_pattern[i])
        #             param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "model..bin")
