# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
import sys 
import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
from model import Model
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaModel, RobertaTokenizer)

logger = logging.getLogger(__name__)
from opendelta import AdapterModel
from collections import defaultdict, deque
import itertools
from collections import Counter

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        
def convert_examples_to_features(js,tokenizer,args):
    """convert examples to token ids"""
    code = js['code_intro'] + "The user's review and the corresponding code must have the same meaning to match, so the code \"" + js['method_content'] + "\" can be represented as \"" 
    code_tokens = tokenizer.tokenize(code)[:args.code_length-4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    

    nl = js['review_intro'] + "The user's review and the corresponding code must have the same meaning to match, so the user's review \"" + js['review_raw'] + "\" can be represented as \"" 
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-4]
    nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length    
    
    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        with open(file_path) as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
            elif "codebase"in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)
            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js) 

        for js in data:
            self.examples.append(convert_examples_to_features(js,tokenizer,args))
                
        # if "train" in file_path:
        #     for idx, example in enumerate(self.examples[:3]):
        #         logger.info("*** Example ***")
        #         logger.info("idx: {}".format(idx))
        #         logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
        #         logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
        #         logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
        #         logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))                             
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        code = torch.tensor(self.examples[i].code_ids)
        nl = torch.tensor(self.examples[i].nl_ids)
        return (code,nl)
            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer):
    """ Train the model """
    #get training dataset
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)

    #get optimizer and scheduler
    # params = []
    # for n, p in model.named_parameters():
    #     if 'adapter' in n:
    #         params.append(p)
    # optimizer = AdamW(params, lr=args.learning_rate, eps=1e-8)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.num_train_epochs)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    
    model.train()
    tr_num,tr_loss,best_mrr = 0,0,0 
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            #get inputs
            code_inputs = batch[0].to(args.device)    
            nl_inputs = batch[1].to(args.device)
            # logger.info(code_inputs.shape)
            # logger.info(nl_inputs.shape)
            #get code and nl vectors
            code_vec = model(code_inputs=code_inputs)
            nl_vec = model(nl_inputs=nl_inputs)

            #calculate scores and loss
            scores = torch.einsum("ab,cb->ac",nl_vec,code_vec)
            loss_fct = CrossEntropyLoss()

        

            rows_as_tuples = [tuple(row.tolist()) for row in nl_vec]


            row_indices = defaultdict(list)


            for index, row in enumerate(rows_as_tuples):
                row_indices[row].append(index)


            additional_indices = []
            for indices in row_indices.values():
                if len(indices) > 1:  
                    additional_indices.extend(list(itertools.combinations(indices, 2)))

            row_indices.clear()


            identity_matrix = np.eye(code_inputs.size(0))


            for i1, i2 in additional_indices:
                identity_matrix[i1, i2] = 1
                identity_matrix[i2, i1] = 1 

            tensor_matrix = torch.tensor(identity_matrix, device=scores.device)

            loss = loss_fct(scores*20, tensor_matrix)

            
            #report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step+1)%100 == 0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss = 0
                tr_num = 0
            
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
            
        #evaluate    
        method_MRR = evaluate(args, model, tokenizer,args.eval_data_file, eval_when_training=True)
        # for key, value in results.items():
        #     logger.info("  %s = %s", key, round(value,4))    
            
        #save best model
        if method_MRR > best_mrr:
            best_mrr = method_MRR
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s",round(best_mrr,3))
            logger.info("  "+"*"*20)                         

            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)


def sort_by_frequency(lst):

    count = Counter(lst)

    sorted_list = [item for item, freq in count.most_common()]
    return sorted_list



import numpy as np

def insert_elements(np_array, lst, alpha_k):

    fixed_part = np_array[:alpha_k].tolist()  
    modifiable_part = np_array[alpha_k:].tolist()

    lst_idx = 0
    i = 0


    while lst_idx < len(lst):
        if i >= len(modifiable_part):
            modifiable_part.append(lst[lst_idx])
        elif lst[lst_idx] != modifiable_part[i]:
            modifiable_part.insert(i, lst[lst_idx])
        else:
            i += 1

        lst_idx += 1
        i += 1

    result_array = np.array(fixed_part + modifiable_part)
    
    return result_array



def evaluate(args, model, tokenizer,file_name,eval_when_training=False):

    model.eval()
    project_name = args.project_name
    alpha_k = args.alpha_k


    review_file = open(args.eval_data_file, "rb")
    reviews = json.load(review_file)

    code_file = open(args.codebase_file, "rb")
    codes = json.load(code_file)

    if alpha_k > 0:
        with open(project_name + '_frequent_groups.json', 'r') as file:
            frequent_groups_data = json.load(file)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(reviews))
    logger.info("  Num codes = %d", len(codes))

    query_vec_list = torch.zeros(len(reviews), 768)
    query_method_list = []
    for idx, review_dic in enumerate(reviews):
        review_content = review_dic['review_raw']
        real_methods_list = review_dic['method_list']
        review_content = "This user review belongs to the " + project_name + " project. " + "The user's review and the corresponding code must have the same meaning to match, so the user's review \"" + review_dic['review_raw'] + "\" can be represented as \"" 
        query_vec = model(tokenizer(review_content, return_tensors='pt',
                                          max_length=128, truncation=True)['input_ids'].to(args.device))
        query_vec_list[idx] = query_vec.detach().cpu()
        query_method_list.append(real_methods_list)

    code_vec_list = torch.zeros(len(codes), 768)
    code_information_list = []
    code_id_list = []
    code_id_dict = {}
    method2path = dict()
    codes_idx = 0
    for idx, method_dic in enumerate(codes):
        code_path = method_dic['method_path']
        code_name = method_dic['method_name']
        code_content = method_dic['method_content']
        code_id = code_path + '#' + code_name
        if code_id not in code_id_list:
            code_content = "This code belongs to the " + project_name + " project. " + "The user's review and the corresponding code must have the same meaning to match, so the code \"" + method_dic['method_content'] + "\" can be represented as \"" 
            code_vec = model(tokenizer(code_content, return_tensors='pt',
                                            max_length=256, truncation=True)['input_ids'].to(args.device))
            code_vec_list[codes_idx]=code_vec.detach().cpu()
            code_information_list.append((code_id, code_path, code_name, code_content))
            code_id_list.append(code_id)
            method2path[code_id] = code_path
            codes_idx += 1
            code_id_dict[code_id] = codes_idx
    
    model.train()
    scores = torch.einsum("ab,cb->ac", query_vec_list, code_vec_list)
    scores = scores.detach().cpu().numpy()
    sort_id_list = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]



    file_set = set()
    for key in method2path.keys():
        file_set.add(method2path[key])

    top1_hitting = 0
    top3_hitting = 0
    top5_hitting = 0
    top10_hitting = 0
    top20_hitting = 0
    review_num = len(query_method_list)
    ranks = []
    result_list = []
    intersection_count_list = []
    real_method_count_list = []


    for review_dic, real_methods_list, sort_id, score in zip(reviews, query_method_list, sort_id_list, scores):

        new_real_method_set = set()

        new_real_file_set = set()
        for method in real_methods_list:
            method_name = method.split('#')[1] + '#' + method.split('#')[2]
            new_real_method_set.add(method_name)
            new_real_file_set.add(method.split('#')[1])
        new_real_method_list = list(new_real_method_set)
        new_real_file_list = list(new_real_file_set)



        review_result_dict = {}
        review_result_dict['review_id'] = review_dic['review_id']
        review_result_dict['sorted_0_idx'] = str(sort_id[0])
        review_result_dict['sorted_0_path_name'] = code_information_list[sort_id[0]][1] + '#' + code_information_list[sort_id[0]][2]


        if alpha_k > 0:
            sort_ids_alpha_k = sort_id[:alpha_k]
            matching_item_list = []
            for ids in sort_ids_alpha_k:
                method_path = code_information_list[ids][1]
                method_name = code_information_list[ids][2]
                method_path_name = method_path + '#' + method_name
                if method_path_name in frequent_groups_data.keys():
                    matching_item = frequent_groups_data[method_path_name][:alpha_k]
                    # print(matching_item)
                    # matching_item_list.append(matching_item)
                    matching_item_list.extend(matching_item)
            # matching_result = most_frequent_element(matching_item_list)
            matching_result = sort_by_frequency(matching_item_list)
            matching_result_repeat = matching_result

            for index in range(alpha_k):
                idx = sort_id[index]
                method_path = code_information_list[idx][1]
                method_name = code_information_list[idx][2]
                code_id = method_path + '#' + method_name
                if code_id in matching_result_repeat:
                    matching_result.remove(code_id)
            matching_result_idx_list = []
            for item_path_name in matching_result:
                fre_idx = code_id_dict[item_path_name] - 1
                matching_result_idx_list.append(fre_idx)
            sort_id = insert_elements(sort_id, matching_result_idx_list, alpha_k)

        rank = 0
        find = False
        for idx in sort_id[:1000]:
            method_path = code_information_list[idx][1]
            method_name = code_information_list[idx][2]
            code_content = code_information_list[idx][3]

            
            if find is False:
                rank += 1
            if method_path + '#' + method_name in new_real_method_list:
                find = True
                ranks.append(1 / rank)
                break
        if not find:
            ranks.append(0)

        pre_methods_list = []
        code_content_list = []
        visited_method_set = set()
        for i in sort_id:
            method_path = code_information_list[i][1]
            method_name = code_information_list[i][2]
            code_content = code_information_list[i][3]
            if method_path + '#' + method_name in visited_method_set:
                continue
            pre_methods_list.append(method_path + '#' + method_name)
            code_content_list.append(code_content)
            visited_method_set.add(method_path + '#' + method_name)
            if len(pre_methods_list) == 100:
                break

        result_dict = review_dic
        result_dict['method_list'] = new_real_method_list
        result_dict['predict_code'] = []

        intersection_count_item = len(set(new_real_method_list) & set(pre_methods_list[:len(new_real_method_list)]))
        intersection_count_list.append(intersection_count_item)
        real_method_count_list.append(len(new_real_method_list))

        if len(set(new_real_method_list) & set(pre_methods_list[:1])) != 0:
            top1_hitting += 1
        if len(set(new_real_method_list) & set(pre_methods_list[:3])) != 0:
            top3_hitting += 1
        if len(set(new_real_method_list) & set(pre_methods_list[:5])) != 0:
            top5_hitting += 1
        if len(set(new_real_method_list) & set(pre_methods_list[:10])) != 0:
            top10_hitting += 1
        if len(set(new_real_method_list) & set(pre_methods_list[:20])) != 0:
            top20_hitting += 1

        pre_method_count = 0
        for pre_method, code_content in zip(pre_methods_list, code_content_list):
            pre_method_count += 1
            cur_pre_result = dict()
            cur_pre_result['pre_method_count'] = pre_method_count
            cur_pre_result['predict_method'] = pre_method
            cur_pre_result['predict_content'] = code_content
            pre_file = pre_method.split('#')[0]
            if pre_file in new_real_file_list:
                isFilePredict = True
            else:
                isFilePredict = False

            if pre_method in new_real_method_list:
                isMethodPredict = True
            else:
                isMethodPredict = False
            cur_pre_result['isFilePredict'] = isFilePredict
            cur_pre_result['isMethodPredict'] = isMethodPredict
            result_dict['predict_code'].append(cur_pre_result)
            if isMethodPredict and isFilePredict:
                break
        result_list.append(result_dict)


    method_MRR = np.mean(ranks)
    print('Method_ranks:', ranks)
    print('Method_MRR:', str(np.mean(ranks)))
    print('Method_top1_hitting:', str(top1_hitting / review_num))
    print('Method_top3_hitting:', str(top3_hitting / review_num))
    print('Method_top5_hitting:', str(top5_hitting / review_num))
    print('Method_top10_hitting:', str(top10_hitting / review_num))
    print('Method_top20_hitting:', str(top20_hitting / review_num))

    intersection_count_res = str(sum(intersection_count_list) / sum(real_method_count_list))
    print('intersection_count_result:', str(intersection_count_res))


    top1_hitting = 0
    top3_hitting = 0
    top5_hitting = 0
    top10_hitting = 0
    top20_hitting = 0
    ranks = []
    for real_methods_list, sort_id in zip(query_method_list, sort_id_list):

        real_code_path_set = set()
        for method in real_methods_list:
            real_code_path_set.add(method.split('#')[1])
        real_code_path_list = list(real_code_path_set)

        if alpha_k > 0:
            sort_ids_alpha_k = sort_id[:alpha_k]
            matching_item_list = []
            for ids in sort_ids_alpha_k:
                method_path = code_information_list[ids][1]
                method_name = code_information_list[ids][2]
                method_path_name = method_path + '#' + method_name
                if method_path_name in frequent_groups_data.keys():
                    matching_item = frequent_groups_data[method_path_name][:alpha_k]
                    # print(matching_item)
                    # matching_item_list.append(matching_item)
                    matching_item_list.extend(matching_item)
            # matching_result = most_frequent_element(matching_item_list)
            matching_result = sort_by_frequency(matching_item_list)
            matching_result_repeat = matching_result


            for index in range(alpha_k):
                idx = sort_id[index]
                method_path = code_information_list[idx][1]
                method_name = code_information_list[idx][2]
                code_id = method_path + '#' + method_name
                if code_id in matching_result_repeat:
                    matching_result.remove(code_id)
            matching_result_idx_list = []
            for item_path_name in matching_result:
                fre_idx = code_id_dict[item_path_name] - 1
                matching_result_idx_list.append(fre_idx)

            sort_id = insert_elements(sort_id, matching_result_idx_list, alpha_k)



        rank = 0
        find = False
        for idx in sort_id[:1000]:
            method_path = code_information_list[idx][1]
            if find is False:
                rank += 1
            if method_path in real_code_path_list:
                find = True
                ranks.append(1 / rank)
                break
        if not find:
            ranks.append(0)

        predict_code_path_list = []
        for id in sort_id:
            pre_method_id = code_information_list[id][0]
            if method2path[pre_method_id] not in predict_code_path_list:
                predict_code_path_list.append(method2path[pre_method_id])
            if len(predict_code_path_list) == 20:
                break

        if len(set(real_code_path_list) & set(predict_code_path_list[:1])) != 0:
            top1_hitting += 1
        if len(set(real_code_path_list) & set(predict_code_path_list[:3])) != 0:
            top3_hitting += 1
        if len(set(real_code_path_list) & set(predict_code_path_list[:5])) != 0:
            top5_hitting += 1
        if len(set(real_code_path_list) & set(predict_code_path_list[:10])) != 0:
            top10_hitting += 1
        if len(set(real_code_path_list) & set(predict_code_path_list[:20])) != 0:
            top20_hitting += 1


    print('File_MRR:', str(np.mean(ranks)))
    print('File_top1_hitting:', str(top1_hitting / review_num))
    print('File_top3_hitting:', str(top3_hitting / review_num))
    print('File_top5_hitting:', str(top5_hitting / review_num))
    print('File_top10_hitting:', str(top10_hitting / review_num))
    print('File_top20_hitting:', str(top20_hitting / review_num))

    return method_MRR

                        
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    parser.add_argument("--do_zero_shot", action='store_true',
                        help="Whether to run eval on the test set.")     
    parser.add_argument("--do_F2_norm", action='store_true',
                        help="Whether to run eval on the test set.")      

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    parser.add_argument('--result_file',default="", type=str,
                        help="random seed for initialization")
    parser.add_argument('--alpha_k', type=int, default=0,
                        help="rerank")
    parser.add_argument('--project_name', type=str, default="",
                        help="project_name")

    
    
    #print arguments
    args = parser.parse_args()
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    #build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path) 


 
    model = Model(model)
    logger.info("Training/evaluation parameters %s", args)
    
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  
            
    # Training
    if args.do_train:    
        model.to(args.device)
        train(args, model, tokenizer)
      
    # Evaluation
    results = {}
    if args.do_eval:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model_to_load = model.module if hasattr(model, 'module') else model  
            model_to_load.load_state_dict(torch.load(output_dir))      
        model.to(args.device)
        result = evaluate(args, model, tokenizer,args.eval_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],3)))
            
    if args.do_test:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model_to_load = model.module if hasattr(model, 'module') else model  
            model_to_load.load_state_dict(torch.load(output_dir))      
        model.to(args.device)
        method_MRR = evaluate(args, model, tokenizer,args.test_data_file)
        logger.info("***** Eval results *****")
        logger.info("  method_MRR = %s", str(round(method_MRR,3)))


if __name__ == "__main__":
    main()