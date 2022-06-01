#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
import torch
import logging
from torch import Tensor
import numpy as np
from collections import defaultdict

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


def eval_model_loss(model, toker, eval_dataloader, epoch_id, infer, args):
    # use the same signature with eval_model_generation
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
            loss_sample, n_sample= model(
                validation=True,
                **batch
            )
            #print('!!!!'*5, logits.shape)
            # logits = logits[:, 0, -8:]
            # strat_preds = torch.argmax(logits, dim=-1)

            return_dict = model.config.use_return_dict
            
            encoder_outputs = model.model.encoder(
            input_ids= batch['input_ids'],
            attention_mask= batch['attention_mask'],
            return_dict=return_dict,
        )
            
            decoder_outputs = model.model.decoder(
                input_ids=batch['decoder_input_ids'],
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=batch['attention_mask'],
                return_dict=return_dict,
            )
            lm_logits = model.lm_head(decoder_outputs.last_hidden_state) + model.final_logits_bias
            print('!!!!'*5, lm_logits.shape)

            encoded_info = {
            'pred_strat_id':0,
            'pred_strat_id_top1': 0,
            'pred_strat_id_top3': 0,
            'pred_strat_id_dist': 0
        }
            strat_preds=model.predict_strategy(lm_logits, encoded_info)
            print('asdfasdfadfaeds',strat_preds)
            strat_preds += (len(toker) - 8)
            strat_ground_truth = batch['decoder_input_ids'][:,1]
            print('!!!!'*10, strat_ground_truth)
            tmp = (strat_preds == strat_ground_truth).float()
            strat_acc.append(torch.mean(tmp).detach().cpu().numpy())
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
