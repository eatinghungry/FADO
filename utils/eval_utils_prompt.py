#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
import torch
import logging
from torch import Tensor
import numpy as np
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


def cal_entropy(generated):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) / total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) / total
    return etp_score, div_score


def eval_model_loss(model, dqn, toker, eval_dataloader, epoch_id, infer, args):
    # use the same signature with eval_model_generation
    logger.info('compute eval model loss, using eval mode, '
                'please change it back to train after calling this function')
    model.eval()

    def process_global_type(emo_path='./_reformat/emo_type.json', prob_path='./_reformat/prob_type.json'):
        with open(emo_path, 'r') as f:
            emo_dict = json.load(f)
        with open(prob_path, 'r') as f2:
            prob_dict = json.load(f2)

        #emo_length, prob_length = len(emo_dict), len(prob_dict)
        emo_dict = dict(map(lambda x:(x[1], x[0]), emo_dict.items()))
        prob_dict = dict(map(lambda x: (x[1], x[0]), prob_dict.items()))


        return emo_dict, prob_dict 

    tot_loss = []
    tot_sample = []
    pointwise_loss = []
    pointwise_sample = []
    max_input_length = 160
    strat_acc = []
    strategies = ['Question', 'Restatement or Paraphrasing', 'Reflection of feelings', 'Self-disclosure',
                   'Affirmation and Reassurance', 'Providing Suggestions', 'Information', 'Others' ]
    strats_in_nlg = ['asking questions', 'restatement or paraphrasing', 'reflection of feelings', 'self-disclosure', 
            'affirmation and reassurance', 'providing suggestions', 'providing information', 'others strategies']
    strat_dict = dict(zip(strategies, strats_in_nlg))
   
    emo_dict, prob_dict = process_global_type()
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = {k: v.to(args.device) if isinstance(v, Tensor) else v for k, v in batch.items()}
            strat_preds = dqn.choose_action(batch['input_ids_og'], batch['attention_mask_og'], 
                            batch['strat_hist'], batch['sentiment_hist'], 
                            batch['utterance_num'], batch['emotion'], batch['problem'])

            process = lambda x: toker.convert_tokens_to_ids(toker.tokenize(x))
            
            prompt_list = []
            for i in range(batch['problem'].shape[0]):
                #prob_idx, emo_idx = int(torch.argmax(batch['problem'][i]).cpu().numpy()), int(torch.argmax(batch['emotion'][i]).cpu().numpy())
                strat_in_nlg = strat_dict[strategies[strat_preds[i]]]
                prompt_txt = f'p: please generate a gentle response with the strategy of {strat_in_nlg}.'
                #prompt_list.append(process(prompt_txt))
                prompt_txt = process(prompt_txt)
                #len_gap = len(batch['prompt'][i]) - len(prompt_txt)
                prompt_txt = torch.tensor(prompt_txt, dtype=torch.long).to('cuda')
                input_ids = batch['input_ids_og'][i]
                input_ids = input_ids[-(max_input_length-len(prompt_txt)):]
                input_ids = torch.cat((input_ids, prompt_txt))#input_ids + prompt_txt
                #print('lolllllololo',input_ids.shape, batch['input_ids'][i].shape)
                #print('lsdfadfoo', len(prompt_txt), prompt_txt.shape, )
                c_id = process('c: ')
                c_id = torch.tensor(c_id, dtype=torch.long).to('cuda')
                input_ids[:len(c_id)] = c_id
                # if len_gap >= 0:
                #     batch['input_ids'][i, len_gap : len(prompt_txt)+len_gap] = prompt_txt
                # elif len_gap < 0:
                #     batch['input_ids'][i, :len(prompt_txt)] = prompt_txt
                batch['input_ids'][i, :] = input_ids

            #prompt_txt = process()
            #strat_preds += (len(toker) - 9) #strat_preds max value is 8
            strat_ground_truth = batch['strat_id']#batch['decoder_input_ids'][:,1]
            tmp = (strat_preds == strat_ground_truth).float()
            #print(f'strat_preds: {strat_preds}')
            #batch['decoder_input_ids'][:,1] = strat_preds
            #batch['input_ids'][:, :len(prompt_txt)] = prompt_list
            strat_acc.append(torch.mean(tmp).detach().cpu().numpy())
            loss_sample, n_sample = model(
                validation=True,
                **batch
            )
            if torch.isnan(loss_sample).sum().cpu().long().numpy() > 0:
                print(loss_sample)
                exit()
            tot_loss.append(loss_sample.sum().cpu().float().numpy())
            tot_sample.append(n_sample.sum().cpu().float().numpy())
            if infer:
                pointwise_loss.extend(loss_sample.sum(dim=-1).cpu().tolist())
                pointwise_sample.extend(n_sample.cpu().tolist())
    #exit()
    tot_loss = np.sum(tot_loss)
    tot_sample = np.sum(tot_sample)
    mean_loss = tot_loss / tot_sample
    mean_ppl = np.exp(mean_loss)
    mean_strat_acc = np.mean(strat_acc)
    print(f"\n Epoch {epoch_id}: Val loss {mean_loss} Val ppl {mean_ppl}  Strat_acc {mean_strat_acc}")
    return mean_loss, mean_ppl, tot_sample, pointwise_loss, pointwise_sample
