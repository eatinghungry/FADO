#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
import torch
import logging
from torch import Tensor
import numpy as np
from collections import defaultdict
import json
from torch.nn.utils.rnn import pad_sequence

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

def load_strat_def(path='/ziyuanqin/projects/nlp/comet/codes_zcj/models/strat_definition.json'):
    with open(path, 'r') as f:
        loaded_dict = json.load(f)
    strat_def_dict = dict()
    for key, item in loaded_dict.items():
        strat_def_dict[key.lower()] = item
    
    return strat_def_dict

def _norm(s):
    return ' '.join(s.strip().split())

def eval_model_loss(model, dqn, toker, model_emo, eval_dataloader, epoch_id, infer, args):
    # use the same signature with eval_model_generation
    strat_dict = {
    0: "Question",
    1: "Restatement or Paraphrasing",
    2: "Reflection of feelings",
    3: "Self-disclosure",
    4: "Affirmation and Reassurance",
    5: "Providing Suggestions",
    6: "Information",
    7: "Others",
    }  
    strat_def_dict = load_strat_def()
    process = lambda x: toker.convert_tokens_to_ids(toker.tokenize(x))
    eos = toker.eos_token_id
    if eos is None:
        eos = toker.sep_token_id
    logger.info('compute eval model loss, using eval mode, '
                'please change it back to train after calling this function')
    model.eval()
    tot_loss = []
    tot_sample = []
    pointwise_loss = []
    pointwise_sample = []
    strat_acc = []
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = {k: v.to(args.device) if isinstance(v, Tensor) else v for k, v in batch.items()}
            batch['emo_encoding'] = model_emo(input_ids=batch['input_emo_ids'],
                             attention_mask=batch['attention_mask_emo'])['last_hidden_state']
            strat_preds, preds, embed = dqn.choose_action(batch['input_ids'], batch['attention_mask'], 
                            batch['strat_hist'], batch['sentiment_hist'], 
                            batch['utterance_num'], batch['emotion'], 
                            batch['problem'],context_emo=batch['emo_encoding'])
            strat_preds_2 = strat_preds + (len(toker) - 9) #strat_preds max value is 8
            strat_defs = []
            for i, strat_num in enumerate(strat_preds):
                strat_num = int(strat_num.cpu().numpy())
                num = strat_dict[strat_num]
                strat_def = strat_def_dict[num.lower()]
                strat_def = process(_norm(strat_def)) + [eos]
                strat_defs.append(strat_def)

            pad = toker.pad_token_id
            if pad is None:
                pad = toker.eos_token_id
                assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
            
            strat_def_batch = pad_sequence([torch.tensor(s, dtype=torch.long) for s in strat_defs],
                          batch_first=True, padding_value=pad).to('cuda')
            strat_mask = pad_sequence([torch.tensor([1.] * len(s), dtype=torch.float) for s in strat_defs],
                          batch_first=True, padding_value=0.).to('cuda')
            
            batch['strat_def'] = strat_def_batch
            batch['strat_mask'] = strat_mask
            batch['rl_branch'] = embed
                
            strat_ground_truth = batch['decoder_input_ids'][:,1]
            tmp = (strat_preds_2 == strat_ground_truth).float()
            #print(f'strat_preds: {strat_preds}')
            batch['decoder_input_ids'][:,1] = strat_preds_2
            batch['strat_id'] = strat_preds
            batch['preds'] = preds
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
