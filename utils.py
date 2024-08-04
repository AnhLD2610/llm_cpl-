import torch
import openai
import random
import time
import numpy as np
import torch.nn.functional as F
from data_loader import get_data_loader_BERT
from nltk import word_tokenize
from retry import retry
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from collections import deque
api_key = ['AIzaSyBhhtj0QazGHw-rC-UT2DaNe31ewxX1H8A',
 'AIzaSyDbOVZpt_IPnLGAj0lefBgM2gqsSnmHHRQ',
 'AIzaSyDDFA8lyMRwKBYI59WESPwURy-No6hmK8o',
 'AIzaSyBH6akgsXP4-5Lp96DAeepBkcXlIsBtZTo',
 'AIzaSyCs38oyPylC1sGXcX7IQYaltdHcKdAm9_g',
 'AIzaSyBbTifY_E3QwWha3Z274JXzhsWeYSPIvcE',
 'AIzaSyABcVXcKZwNiHUr4rK4JIGHLSxnqjv03_Q',
 'AIzaSyBP_KLOmJJWFAbIHdJ4VKGTrznSrPmvHfg',
 'AIzaSyC_lW_DujJAOvhFhnxvxeQVj0VthhBsGjo',
 'AIzaSyAjNdcFWOYN-1UhM2Q4tocy9T0KfImEEcM',
 'AIzaSyC6XNcP87RPNzF7Txvl4oB2ZPcFHRlfxp8',
 'AIzaSyDzqZtys4QRT3aokL-qTbkCkMBCzfJmVLc',
 'AIzaSyDW40W0mJdWP9pcUhlydBT4gUa-OhV_8VE',
 'AIzaSyBixha-TDt3XzpfYIbZhg53xcS8ydVPbBo',
 'AIzaSyAovqCq5YxI6JXXqHPPP51YQBrAFGR4Id4',
 'AIzaSyDWXCp3DqHkzRF7LpFmnDRqrR8tFDhN6vk',
 'AIzaSyBYpN3w3ZrpAlxOik-6eRHtCiP-GWEZvjA',
 'AIzaSyA7V6tUutUqxPdGzIGVWggqdVXKZJjclhI',
 'AIzaSyAOl78H3f_2e0o9Bydlfc_OmmSnbqgvoYo',
 'AIzaSyAkapAtynasS_BTogwUW74razK35O_6cp4']
api_queue = deque(api_key)


class Moment:
    def __init__(self, config) -> None:
        self.config = config
        self.features = None
        self.labels = None
        self.mem_samples = None
        self.mem_features = None
        self.mem_labels = None
        self.sample_k = config.sample_k
        self.temperature = config.contrastive_temp
        self.m = config.margin
    
    def init_moment(self, encoder, dataset, is_memory=False):
        encoder.eval()
        datalen = len(dataset)
        if not is_memory:
            # self.features = torch.zeros(datalen, self.config.encoder_output_size, dtype=torch.bfloat16)
            self.features = torch.zeros(datalen, self.config.encoder_output_size)

            data_loader = get_data_loader_BERT(self.config, dataset) # shuffle=False
            lbs = []
            for step, (instance, labels, ind) in enumerate(data_loader):
                # for k in instance.keys():
                #     instance[k] = instance[k].to(self.config.device)
                hidden  = encoder(instance)
                fea = hidden.detach().cpu().data
                self.update(ind, fea)
                lbs.append(labels) # shuffle=False
            lbs = torch.cat(lbs)
            self.labels = lbs
        else:
            self.mem_samples = dataset
            # self.mem_features = torch.zeros(datalen, self.config.encoder_output_size,  dtype=torch.bfloat16)
            self.mem_features = torch.zeros(datalen, self.config.encoder_output_size)

            data_loader = get_data_loader_BERT(self.config, dataset) # shuffle=False
            lbs = []
            for step, (instance, labels, ind) in enumerate(data_loader):
                # for k in instance.keys():
                #     instance[k] = instance[k].to(self.config.device)
                hidden  = encoder(instance)
                fea = hidden.detach().cpu().data
                self.update(ind, fea, is_memory)
                lbs.append(labels) # shuffle=False
            lbs = torch.cat(lbs)
            self.mem_labels = lbs            

    def update(self, ind, feature, is_memory=False):
        if not is_memory:
            self.features[ind] = feature
        else:
            self.mem_features[ind] = feature
    
    def update_allmem(self, encoder):
            data_loader = get_data_loader_BERT(self.config, self.mem_samples, batch_size=64) # shuffle=False
            for step, (instance, labels, ind) in enumerate(data_loader):
                # for k in instance.keys():
                #     instance[k] = instance[k].to(self.config.device)
                hidden  = encoder(instance)
                fea = hidden.detach().cpu().data
                self.update(ind, fea, is_memory=True)
        

    def get_mem_proto(self):
        cinds = []
        for x in self.mem_labels:
            if x.item() not in cinds:
                cinds.append(x.item())

        num = len(cinds)
        feats = self.mem_features
        centroids = torch.zeros((num, feats.size(1)), dtype=torch.float32, device=feats.device)
        for i, c in enumerate(cinds):
            ind = np.where(self.mem_labels.cpu().numpy() == c)[0]
            centroids[i, :] = feats[ind, :].mean(dim=0)
        return centroids
    
    def contrastive_loss(self, x, labels, is_memory=False):
        '''
        x (B, H)
        '''
        if is_memory:
            ct_x = self.mem_features.to(self.config.device)
            ct_y = self.mem_labels
        else:
            idx = list(range(len(self.features)))
            if len(idx) > self.sample_k:
                sample_id = random.sample(idx, self.sample_k)
            else:  # sample number > total feature
                sample_id = idx
            ct_x = self.features[sample_id].to(self.config.device) # (N, H)
            ct_y = self.labels[sample_id] # (N)

        # l2 normalize
        x = F.normalize(x, p=2, dim=1)
        ct_x = F.normalize(ct_x, p=2, dim=1)
        
        t1 = torch.mm(x, ct_x.T) + 1 # 0 <= cos + 1 <= 2
        zeros = (torch.zeros_like(t1)).to(self.config.device)
        pos = self.m + 0.5 * t1
        neg = 1 - self.m + 0.5 * t1
        dot_product_tempered_pos = torch.where(pos > 0, pos * t1 / self.temperature, zeros)
        dot_product_tempered_neg = torch.where(neg > 0, neg * t1 / self.temperature, zeros)
        
        exp_dot_tempered_pos = (
            torch.exp(dot_product_tempered_pos - \
                torch.max(dot_product_tempered_pos, dim=1, keepdim=True)[0].detach()) + 1e-5
        )
        exp_dot_tempered_neg = (
            torch.exp(dot_product_tempered_neg - \
                torch.max(dot_product_tempered_pos, dim=1, keepdim=True)[0].detach()) + 1e-5
        ) 
        mask_combined_pos = (labels.unsqueeze(1).repeat(1, ct_y.shape[0]) == ct_y).to(self.config.device)
        mask_combined_neg = ~mask_combined_pos
        cardinality_per_samples = torch.sum(mask_combined_pos, dim=1)

        sum_temp = torch.sum(exp_dot_tempered_pos * mask_combined_pos, dim=1, keepdim=True) \
            + torch.sum(exp_dot_tempered_neg * mask_combined_neg, dim=1, keepdim=True)
        log_prob = -torch.log(exp_dot_tempered_pos / sum_temp)
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined_pos, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss

    # def contrastive_loss(self, x, labels, is_memory=False):
    #     '''
    #     x (B, H)
    #     '''
    #     if is_memory:
    #         ct_x = self.mem_features.to(self.config.device)
    #         # ct_x_des = self.mem_features_des.to(self.config.device)

    #         ct_y = self.mem_labels
    #     else:
    #         idx = list(range(len(self.features)))
    #         if len(idx) > self.sample_k:
    #             sample_id = random.sample(idx, self.sample_k)
    #         else:  # sample number > total feature
    #             sample_id = idx
    #         ct_x = self.features[sample_id].to(self.config.device)  # (N, H)
    #         # ct_x_des = self.features_des[sample_id].to(self.config.device)  # (N, H)

    #         ct_y = self.labels[sample_id]  # (N)

    #     # l2 normalize
    #     x = F.normalize(x, p=2, dim=1) # (B, H)
    #     ct_x = F.normalize(ct_x, p=2, dim=1) # (N, H)
    #     # ct_x_des = F.normalize(ct_x_des, p=2, dim=1) # (B, H)
    #     # x = torch.cat((x, ct_x_des), dim=1) 

    #     # ct_x = torch.cat((ct_x, ct_x_des), dim=1) # (N, 2*H)

    #     t1 = torch.mm(x, ct_x.T) + 1  # 0 <= cos + 1 <= 2
    #     # zeros = (torch.zeros_like(t1)).to(self.config.device)


    #     mask_combined_pos = (labels.unsqueeze(1).repeat(1, ct_y.shape[0]) == ct_y).to(self.config.device)
    #     mask_combined_neg = ~mask_combined_pos

    #     # Calculate beta for negative samples
    #     exp_t1_neg = torch.exp(t1 / self.temperature) * mask_combined_neg
    #     sum_exp_t1_neg = torch.sum(exp_t1_neg, dim=1, keepdim=True) + 1e-6
    #     cardinality_neg = torch.sum(mask_combined_neg, dim=1, keepdim=True)
    #     # beta = (exp_t1_neg / sum_exp_t1_neg) * cardinality_neg

    #     exp_t1_pos = torch.exp(t1 / self.temperature) * mask_combined_pos
    #     sum_exp_t1_pos = torch.sum(exp_t1_pos, dim=1, keepdim=True) + 1e-6



    #     # dot_product_tempered_pos = torch.where(pos > 0, pos * t1 / self.temperature, zeros)
    #     # dot_product_tempered_neg = torch.where(neg > 0, neg * t1 / self.temperature, zeros)

    #     # exp_dot_tempered_pos = (
    #     #         torch.exp(dot_product_tempered_pos - \
    #     #                   torch.max(dot_product_tempered_pos, dim=1, keepdim=True)[0].detach()) + 1e-5
    #     # )
    #     # exp_dot_tempered_neg = (
    #     #         torch.exp(dot_product_tempered_neg - \
    #     #                   torch.max(dot_product_tempered_pos, dim=1, keepdim=True)[0].detach()) + 1e-5
    #     # )
    #     # mask_combined_pos = (labels.unsqueeze(1).repeat(1, ct_y.shape[0]) == ct_y).to(self.config.device)
    #     # mask_combined_neg = ~mask_combined_pos
    #     cardinality_per_samples = torch.sum(mask_combined_pos, dim=1)

    #     sum_temp = sum_exp_t1_pos + sum_exp_t1_neg
    #     log_prob = -torch.log(sum_exp_t1_pos / sum_temp)
    #     supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined_pos,
    #                                                        dim=1) / cardinality_per_samples
    #     supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

    #     return supervised_contrastive_loss
    # def contrastive_loss(self, x, labels, is_memory=False):
    #     '''
    #     x (B, H)
    #     '''
    #     if is_memory:
    #         ct_x = self.mem_features.to(self.config.device)
    #         ct_y = self.mem_labels
    #     else:
    #         idx = list(range(len(self.features)))
    #         if len(idx) > self.sample_k:
    #             sample_id = random.sample(idx, self.sample_k)
    #         else:  # sample number > total feature
    #             sample_id = idx
    #         ct_x = self.features[sample_id].to(self.config.device) # (N, H)
    #         ct_y = self.labels[sample_id] # (N)

    #     # l2 normalize
    #     x = F.normalize(x, p=2, dim=1)
    #     ct_x = F.normalize(ct_x, p=2, dim=1)
        
    #     t1 = torch.mm(x, ct_x.T) + 1 # 0 <= cos + 1 <= 2
    #     zeros = (torch.zeros_like(t1)).to(self.config.device)
    #     pos = self.m + 0.5 * t1 
    #     neg = 1 - self.m + 0.5 * t1 
    #     dot_product_tempered_pos = torch.where(pos > 0, pos * t1 / self.temperature, zeros)
    #     dot_product_tempered_neg = torch.where(neg > 0, neg * t1 / self.temperature, zeros)
        
    #     exp_dot_tempered_pos = (
    #         torch.exp(dot_product_tempered_pos - \
    #             torch.max(dot_product_tempered_pos, dim=1, keepdim=True)[0].detach()) + 1e-5
    #     )
    #     exp_dot_tempered_neg = (
    #         torch.exp(dot_product_tempered_neg - \
    #             torch.max(dot_product_tempered_pos, dim=1, keepdim=True)[0].detach()) + 1e-5
    #     ) 
    #     mask_combined_pos = (labels.unsqueeze(1).repeat(1, ct_y.shape[0]) == ct_y).to(self.config.device)
    #     mask_combined_neg = ~mask_combined_pos
    #     cardinality_per_samples = torch.sum(mask_combined_pos, dim=1)

    #     exp_t1_neg = exp_dot_tempered_neg * mask_combined_neg
    #     sum_exp_t1_neg = torch.sum(exp_t1_neg, dim=1, keepdim=True)
    #     cardinality_neg = torch.sum(mask_combined_neg, dim=1, keepdim=True)
    #     beta = (exp_t1_neg / sum_exp_t1_neg) * cardinality_neg

    #     sum_temp = torch.sum(exp_dot_tempered_pos * mask_combined_pos, dim=1, keepdim=True) \
    #         + beta*torch.sum(exp_dot_tempered_neg * mask_combined_neg, dim=1, keepdim=True)
    #     # sum_temp = torch.sum(exp_dot_tempered_pos * mask_combined_pos, dim=1, keepdim=True) \
    #     #     + torch.sum(exp_dot_tempered_neg * mask_combined_neg, dim=1, keepdim=True)
    #     log_prob = -torch.log(exp_dot_tempered_pos / sum_temp)
    #     supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined_pos, dim=1) / cardinality_per_samples
    #     supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

    #     return supervised_contrastive_loss


# for openai
# @retry(tries=10, delay=1)
from openai import OpenAI

def gpt(input, t=0, key=None):
    MAX_TRIES = 15
    # client = OpenAI()
    
    while MAX_TRIES > 0:
        try:
            time.sleep(3)
            cur_api_key = api_queue.popleft()
            api_queue.append(cur_api_key)
            genai.configure(api_key=cur_api_key)
            # genai.configure(api_key= 'AIzaSyDBECQnpdlHjyw0m90b8nMRBsA_oaE0WXU')
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(input)
            return response.text
            # completion = client.chat.completions.create(
            # model="gpt-4o-mini",
            # messages=[
            #     {"role": "user", "content": input}
            # ]
            # ,
            # temperature=t

            # )
            # return completion.choices[0].message.content
        except Exception as e:
            print(e)
            MAX_TRIES -= 1
    print('gen failed')
    return ''
        


@retry(tries=10, delay=1)
def gemini(input, t=0, key=None):
    time.sleep(5)
    cur_api_key = api_queue.popleft()
    api_queue.append(cur_api_key)
    genai.configure(api_key=cur_api_key)
    # genai.configure(api_key= 'AIzaSyDBECQnpdlHjyw0m90b8nMRBsA_oaE0WXU')
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content(input)
    return response.text


def parse(rel2id, text):
    cons = ['Relation:', 'Context:', 'Head Entity:', 'Tail Entity:']
    lens = [len(item) for item in cons]
    parse_text = []

    temp = text
    while True:
        parse_item = {}

        i = temp.find(cons[0])
        temp = temp[i + lens[0]:]
        i = temp.find(cons[1])
        r = temp[:i].strip()
        temp = temp[i + lens[1]:]
        i = temp.find(cons[2])
        c = temp[:i].strip()
        temp = temp[i + lens[2]:]
        i = temp.find(cons[3])
        h = temp[:i].strip()
        temp = temp[i + lens[3]:]
        i = temp.find('\n')
        t = temp[:i].strip()
        i = temp.find(cons[0])

        r = r.split('\n')[0]
        r = r.replace('**', '')
        r = r.replace('\n', '')
        r = r.strip()

        parse_item['relation'] = rel2id[r]
        parse_item['index'] = 0
        tokens = word_tokenize(c.lower())
        parse_item['tokens'] = tokens

        headent, tailent = h.lower(), t.lower()
        h_tokens, t_tokens = word_tokenize(headent), word_tokenize(tailent)
        try:
            h1 = tokens.index(h_tokens[0])
        except Exception:
            h1 = 0
        try:
            h2 = tokens.index(h_tokens[-1])
        except Exception:
            h2 = h1
        try:
            t1 = tokens.index(t_tokens[0])
        except Exception:
            t1 = h2
        try:
            t2 = tokens.index(t_tokens[-1])
        except Exception:
            t2 = t1
        parse_item['h'] = [headent, '0', [[h1, h2]]]
        parse_item['t'] = [tailent, '0', [[t1, t2]]]

        parse_text.append(parse_item)

        if i == -1:
            break
        temp = temp[i:]

    return parse_text


def prompt_input(rname, rdesc, sample=None, n=10):
    pre_input = 'You are a data scientist working on a relation extraction task. Please do the following task and do not give output in the markdown format.'
    input = ''
    if sample == None:
        input = 'One sample in relation extraction datasets consists of a relation, a context, a pair of head and tail entities in the context.The head entity has the relation with the tail entity. Generate ' \
                + str(
            n) + ' diversity samples (must have full : Relation , Context , Head Entity , Tail Entity) for the relation "' + rname \
                + '" which means ' + rdesc \
                + ', and indicate the head entity and tail entity in the following format:\n' \
                + 'Relation: xxx\nContext: xxx\nHead Entity: xxx\nTail Entity: xxx'
    else:
        input = 'One sample in relation extraction datasets consists of a relation, a context, a pair of head and tail entities in the context.The head entity has the relation with the tail entity.\n' \
                + 'Relation "' + rname + '" means ' + rdesc + '.\nHere is an example:\n' \
                + 'Relation: ' + rname + '\nContext: ' + sample['tokens'] + '\nHead Entity: ' + sample[
                    'h'] + '\nTail Entity: ' + sample['t'] + '\n' \
                + 'Please generate ' + str(
            n) + ' diversity samples (must have full : Relation , Context , Head Entity , Tail Entity) like the above example for the relation "' + rname + '":'
    return pre_input + input


def gen_data(r2desc, rel2id, sample, n=10, t=0, key=None):
    import random
    MAX_TRIES = 15
    rname = sample['relation']
    rdesc = r2desc[rname]
    print('####', rname, '####')
    input = prompt_input(rname, rdesc, sample=sample, n=n)
    print(input)
    output = gpt(input=input, t=t, key=key)
    print(output)
    while MAX_TRIES > 0:
        try:
            parse_output = parse(rel2id, output)
            return parse_output
        except Exception as e:
            print(e)
            t = random.uniform(0.5, 1)
            output = gpt(input=input + "\nRelation: ", t=t, key=key)
            MAX_TRIES -= 1

    return ''


if __name__ == "__main__":
    s = """## Relation: person countries of residence

**Sample 1:**

Context: The renowned author, Salman Rushdie, was born in Bombay, India, and now resides in the United Kingdom. 
Head Entity: Salman Rushdie
Tail Entity: United Kingdom

**Sample 2:**

Context: During his visit to Japan, President Biden met with Prime Minister Kishida to discuss economic cooperation.
Head Entity: President Biden
Tail Entity: Japan

**Sample 3:**

Context:  After her successful career in Hollywood, actress Angelina Jolie decided to move to Cambodia and dedicate her time to humanitarian work.
Head Entity: Angelina Jolie
Tail Entity: Cambodia

**Sample 4:**

Context: The renowned scientist, Albert Einstein, emigrated to the United States in 1933 to escape the Nazi regime in Germany.
Head Entity: Albert Einstein
Tail Entity: United States

**Sample 5:**

Context: The musician, Bob Marley, was born in Jamaica and spent most of his life there before moving to Miami, Florida, in the 1970s.
Head Entity: Bob Marley
Tail Entity: Jamaica """
    # print(parse(s))



