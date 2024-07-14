import torch
import torch.nn as nn
import numpy as np
# from llm2vec import LLM2Vec
from llm2vec import *
from torch import Tensor, device, nn

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model
from typing import Any, Dict, List, Optional, Tuple, Union

def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

class EncodingModel(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        if config.model == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained(
                "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
            )
            # config = AutoConfig.from_pretrained(
            #     "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", trust_remote_code=True
            # )
            # model = AutoModel.from_pretrained(
            #     "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            #     trust_remote_code=True,
            #     config=config,
            #     torch_dtype=torch.bfloat16,
            #     device_map="cuda" if torch.cuda.is_available() else "cpu",
            # )


            # model.enable_input_require_grads()
            # model = PeftModel.from_pretrained(
            #     model,
            #     "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
            #     is_trainable=True,
            # )
            # model = model.merge_and_unload()  # This can take several minutes on cpu
            # self.encoder = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=256)
            self.encoder = LLM2Vec.from_pretrained(
                "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
                peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16,
                merge_peft=True,
                pooling_mode="mean",
                max_length=256,
            )
            self.encoder.model = self.initialize_peft(
                self.encoder.model,
            )

            # model.enable_input_require_grads()
            # # Loading supervised model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + supervised (LoRA).
            # model = PeftModel.from_pretrained(
            #     model, "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
            # )

            # Wrapper for encoding and pooling operations
            # for name, param in self.encoder.named_parameters():
            #     if 'lora_A' in name or 'lora_B' in name:
            #         param.requires_grad = True
            
    def initialize_peft(
        self,
        model,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_modules: Optional[List[str]] = None,
    ):
        if lora_modules is None and model.config.__class__.__name__ in [
            "LlamaConfig",
            "MistralConfig",
            "GemmaConfig",
            "Qwen2Config",
        ]:
            lora_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        elif lora_modules is None:
            raise ValueError("lora_modules must be specified for this model.")

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=None,
        )

        model = get_peft_model(model, config)
        print(f"Model's Lora trainable parameters:")
        model.print_trainable_parameters()
        return model

        # if config.tune == 'prompt':
        #     for param in self.encoder.parameters():
        #         param.requires_grad = False


        

        # self.bert_word_embedding = self.encoder.get_input_embeddings()
        # self.embedding_dim = self.bert_word_embedding.embedding_dim
        # self.prompt_lens = config.prompt_len * config.prompt_num
        # self.softprompt_encoder = nn.Embedding(self.prompt_lens, self.embedding_dim).to(config.device)
        # # initialize prompt embedding
        # self._init_prompt()
        # self.prompt_ids = torch.LongTensor(list(range(self.prompt_lens))).to(self.config.device)

    # def _init_prompt(self):
    #     # is is is [e1] is is is [MASK] is is is [e2] is is is
    #     if self.config.prompt_init == 1:
    #         prompt_embedding = torch.zeros_like(self.softprompt_encoder.weight).to(self.config.device)
    #         token_embedding = self.bert_word_embedding.weight[2003]
    #         prompt_embedding[list(range(self.prompt_lens)), :] = token_embedding.clone().detach()
    #         for param in self.softprompt_encoder.parameters():
    #             param.data = prompt_embedding # param.data
       
    #     # ! @ # [e1] he is as [MASK] * & % [e2] just do it  
    #     elif self.config.prompt_init == 2:
    #         prompt_embedding = torch.zeros_like(self.softprompt_encoder.weight).to(self.config.device)
    #         ids = [999, 1030, 1001, 2002, 2003, 2004, 1008, 1004, 1003, 2074, 2079,  2009]
    #         for i in range(self.prompt_lens):
    #             token_embedding = self.bert_word_embedding.weight[ids[i]]
    #             prompt_embedding[i, :] = token_embedding.clone().detach()
    #         for param in self.softprompt_encoder.parameters():
    #             param.data = prompt_embedding # param.data


    # def embedding_input(self, input_ids): # (b, max_len)
    #     input_embedding = self.bert_word_embedding(input_ids) # (b, max_len, h)
    #     prompt_embedding = self.softprompt_encoder(self.prompt_ids) # (prompt_len, h)

    #     for i in range(input_ids.size()[0]):
    #         p = 0
    #         for j in range(input_ids.size()[1]):
    #             if input_ids[i][j] == self.config.prompt_token_ids:
    #                 input_embedding[i][j] = prompt_embedding[p]
    #                 p += 1

    #     return input_embedding


    # def forward(self, inputs): # (b, max_length)
    #     batch_size = inputs['ids'].size()[0]
    #     tensor_range = torch.arange(batch_size) # (b)     
    #     pattern = self.config.pattern
    #     if pattern == 'softprompt' or pattern == 'hybridprompt':
    #         input_embedding = self.embedding_input(inputs['ids'])
    #         outputs_words = self.encoder(inputs_embeds=input_embedding, attention_mask=inputs['mask'])[0]
    #     else:
    #         outputs_words = self.encoder(inputs['ids'], attention_mask=inputs['mask'])[0] # (b, max_length, h)

    #     # return [CLS] hidden
    #     if pattern == 'cls' or pattern == 'softprompt':
    #         clss = torch.zeros(batch_size, dtype=torch.long)
    #         return outputs_words[tensor_range ,clss] # (b, h)

    #     # return [MASK] hidden
    #     elif pattern == 'hardprompt' or pattern == 'hybridprompt':
    #         masks = []
    #         for i in range(batch_size):
    #             ids = inputs['ids'][i].cpu().numpy()
    #             mask = np.argwhere(ids == self.config.mask_token_ids)[0][0]
    #             masks.append(mask)
    #         mask_hidden = outputs_words[tensor_range, torch.tensor(masks)] # (b, h)
    #         return mask_hidden

    #     # return e1:e2 hidden
    #     elif pattern == 'marker':
    #         h1, t1 = [], []
    #         for i in range(batch_size):
    #             ids = inputs['ids'][i].cpu().numpy()
    #             h1_index, t1_index = np.argwhere(ids == self.config.h_ids), np.argwhere(ids == self.config.t_ids)
    #             h1.append(0) if h1_index.size == 0 else h1.append(h1_index[0][0])
    #             t1.append(0) if t1_index.size == 0 else t1.append(t1_index[0][0])

    #         h_state = outputs_words[tensor_range, torch.tensor(h1)] # (b, h)
    #         t_state = outputs_words[tensor_range, torch.tensor(t1)]

    #         concerate_h_t = (h_state + t_state) / 2 # (b, h)
    #         return concerate_h_t


    def forward(self, inputs): # (b, max_length)
        # batch_size = inputs['input'].size()[0]
        # tensor_range = torch.arange(batch_size) # (b)     
        pattern = self.config.pattern
        if pattern == 'softprompt' or pattern == 'hybridprompt':
            input_embedding = self.embedding_input(inputs['ids'])
            # outputs_words = self.encoder.encode((inputs_embeds=input_embedding, attention_mask=inputs['mask'])[0]
        else:

            features = self.tokenize(
            [LLM2Vec.prepare_for_tokenization(sentence) for sentence in inputs['input']]
            )
            features = batch_to_device(features, self.config.device)

            # with torch.no_grad():
            embeddings = self.forward(features)

            # outputs_words = self.encoder.encode_train((inputs['input'])) # (b, h)
        # outputs_words = torch.nn.functional.normalize(outputs_words, p=2, dim=1)
        return embeddings
        # # return [CLS] hidden
        # if pattern == 'cls' or pattern == 'softprompt':
        #     clss = torch.zeros(batch_size, dtype=torch.long)
        #     return outputs_words[tensor_range ,clss] # (b, h)

        # # return [MASK] hidden
        # elif pattern == 'hardprompt' or pattern == 'hybridprompt':
        #     masks = []
        #     for i in range(batch_size):
        #         ids = inputs['ids'][i].cpu().numpy()
        #         mask = np.argwhere(ids == self.config.mask_token_ids)[0][0]
        #         masks.append(mask)
        #     mask_hidden = outputs_words[tensor_range, torch.tensor(masks)] # (b, h)
        #     return mask_hidden

        # # return e1:e2 hidden
        # elif pattern == 'marker':
        #     h1, t1 = [], []
        #     for i in range(batch_size):
        #         ids = inputs['ids'][i].cpu().numpy()
        #         h1_index, t1_index = np.argwhere(ids == self.config.h_ids), np.argwhere(ids == self.config.t_ids)
        #         h1.append(0) if h1_index.size == 0 else h1.append(h1_index[0][0])
        #         t1.append(0) if t1_index.size == 0 else t1.append(t1_index[0][0])

        #     h_state = outputs_words[tensor_range, torch.tensor(h1)] # (b, h)
        #     t_state = outputs_words[tensor_range, torch.tensor(t1)]

        #     concerate_h_t = (h_state + t_state) / 2 # (b, h)
        #     return concerate_h_t

