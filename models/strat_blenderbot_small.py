# coding=utf-8
# copied from bart

from base64 import encode
import torch
import torch.nn as nn
import torch.nn.functional as F
#from codes_zcj.inputters import strat
from models.model_utils import BaseModel
from transformers.generation_utils import top_k_top_p_filtering
from transformers.models.blenderbot_small import (BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration,)
from transformers.modeling_outputs import (BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput,)
from .PARAMS import SAMPLE, TEMPERATURE


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


def linear_combination(x, y, epsilon):
    return epsilon*x + (1-epsilon)*y


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='none'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)



# loss_func = LabelSmoothingCrossEntropy(epsilon=args.label_smoothing)
# loss_func(outputs, labels)

class Model(BaseModel, BlenderbotSmallForConditionalGeneration):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        validation=False,
        **kwargs
    ):
        assert self.toker is not None
        
        encoded_info = kwargs
        assert (self.training or validation) == (labels is not None)
        if validation:
            labels[:, 0] = -100
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if not self.training and not validation: # inference
            use_cache = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
            #strat_id = kwargs['strat_id'],
            #preds = kwargs['preds'],
            **kwargs,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        # lm_logits2 = self.lm_head(outputs['last_hidden_stat']) + self.final_logits_bias
        # assert lm_logits == lm_logits2, 'logit outputs[0] does not equal to outputs last_hidden_stat'

        # encode_logits = outputs['encoder_last_hidden_state'] 
        # encode_logits = torch.mean(encode, 1)
        # encode_loss = F.cross_entropy()
        loss_func = LabelSmoothingCrossEntropy(epsilon=0.1)
        if validation:
            lm_logits = lm_logits[..., :self.toker.vocab_size].contiguous() #?

        masked_lm_loss = None
        if labels is not None:
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), reduction='none')
            #loss = loss_func(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            loss = loss.view(labels.size(0), labels.size(1))
            #labels.ne(-100) = [True, True ...] if no elements equal to -100
            #-100 here is a value of mask,sum(labels.ne(-100)) will give u the size of unmasked tokens
            label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
            masked_lm_loss = torch.sum(loss) / torch.sum(label_size)
            ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))

        if not self.training and not validation: # inference
            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

        elif self.training: # training
            assert not validation
            res = {'all': masked_lm_loss, 'ppl': ppl_value}
            return res

        else: # validation
            assert not self.training
            return loss, label_size#, lm_logits

    
    def forward2(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        validation=False,
        **kwargs
    ):
        assert self.toker is not None
        
        encoded_info = kwargs
        assert (self.training or validation) == (labels is not None)
        if validation:
            labels[:, 0] = -100
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if not self.training and not validation: # inference
            use_cache = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
            **kwargs,
        )
        tmp = outputs[0]
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        #lm_logits2 = self.lm_head(outputs['last_hidden_stat']) + self.final_logits_bias
        #assert lm_logits == lm_logits2, 'logit outputs[0] does not equal to outputs last_hidden_stat'

        encode_logits = outputs['last_hidden_state']  #outputs = decoder_outputs + encoder_outputs
        #print(encode_logits, '!!!!!!!!')
        encode_logits = torch.mean(encode_logits, 1)
        encode_logits = F.relu(self.encode_head(encode_logits))
        encode_loss = F.cross_entropy(encode_logits, kwargs['strat_id'])
        
        if validation:
            lm_logits = lm_logits[..., :self.toker.vocab_size].contiguous() #?

        masked_lm_loss = None
        if labels is not None:
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), reduction='none')
            loss = loss.view(labels.size(0), labels.size(1))
            #labels.ne(-100) = [True, True ...] if no elements equal to -100
            #-100 here is a value of mask,sum(labels.ne(-100)) will give u the size of unmasked tokens
            label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
            masked_lm_loss = torch.sum(loss) / torch.sum(label_size)
            ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))

        if not self.training and not validation: # inference
            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

        elif self.training: # training
            assert not validation
            res = {'all': masked_lm_loss, 'ppl': ppl_value, 'pooling_loss': encode_loss}
            return res

        else: # validation
            assert not self.training
            return loss, label_size#, lm_logits

    def predict_strategy(self, logits, encoded_info):
        assert not self.training
        strat_id = encoded_info.get('strat_id', None)
        # logits = logits[:, 0, -8:]
    
        # if strat_id is not None:
        #     pred = strat_id
        # else:
        #     if SAMPLE: #random strat
        #         filtered_logits = top_k_top_p_filtering(logits / TEMPERATURE, top_p=0.9)
        #         pred = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(-1)
        #     else: #
        #         pred = torch.argmax(logits, dim=-1)
        # logtis = dqn.choose_action2(encoded_info['last_sentence'], encoded_info['attention_mask_ls'], 
        #     encoded_info['strat_hist'], encoded_info['sentiment_hist'], 
        #     encoded_info['utterance_num'], encoded_info['emotion'], encoded_info['problem'])
        pred = torch.argmax(logits, dim=-1).squeeze(-1)
        pred_top1 = torch.topk(logits, k=1, dim=-1)[1].squeeze(-1)
        pred_top3 = torch.topk(logits, k=3, dim=-1)[1].squeeze(-1)
    
        encoded_info.update({
            'pred_strat_id': pred,
            'pred_strat_id_top1': pred_top1,
            'pred_strat_id_top3': pred_top3,
            'pred_strat_id_dist': F.softmax(logits, dim=-1)
        })

        return pred
    
    @torch.no_grad()
    def generate(
        self,
        
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        return_dict=None,
        **kwargs
    ):
        assert not self.training
        assert self.toker is not None
        #print(f"!!!!!!{kwargs.keys()}")
        encoded_info = kwargs
        assert decoder_input_ids.size(1) == 1
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            **kwargs,
        )

        strat_outputs = self.model.encoder(
            input_ids=kwargs['strat_def'],
            attention_mask=kwargs['strat_mask'],
            return_dict=return_dict,
            **kwargs,
        )
        
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            strat_def=strat_outputs[0],
            strat_mask=kwargs['strat_mask'],
            return_dict=return_dict,
            strat_id=kwargs['strat_id'],
            preds=kwargs['preds']
        )
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias #?
        self.predict_strategy(encoded_info['strat_logits'], encoded_info)
        
        decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.toker) - 8], dim=-1)
        
        assert 'max_length' in kwargs
        kwargs['max_length'] = kwargs['max_length'] + decoder_input_ids.size(1)
        kwargs['use_cache'] = True
        
        if len(self.toker) > self.toker.vocab_size:
            bad_words_ids = [[i] for i in range(self.toker.vocab_size, len(self.toker))]
            kwargs['bad_words_ids'] = bad_words_ids
        
        # *****????
        generations = super().generate(
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            **kwargs
            # strat_def = kwargs['strat_def'],
            # strat_mask = kwargs['strat_mask'],
            # strat_id = kwargs['strat_id'],
            # preds = kwargs['preds']
        )
        return encoded_info, generations[:, decoder_input_ids.size(1):]


# coding=utf-8
# copied from bart


class Model2(BaseModel, BlenderbotSmallForConditionalGeneration):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        validation=False,
        **kwargs
    ):
        assert self.toker is not None
        
        encoded_info = kwargs
        assert (self.training or validation) == (labels is not None)
        if validation:
            labels[:, 0] = -100
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if not self.training and not validation: # inference
            use_cache = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        
        if validation:
            lm_logits = lm_logits[..., :self.toker.vocab_size].contiguous()

        masked_lm_loss = None
        if labels is not None:
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), reduction='none')
            loss = loss.view(labels.size(0), labels.size(1))
            #labels.ne(-100) = [True, True ...] if no elements equal to -100
            #-100 here is a value of mask,sum(labels.ne(-100)) will give u the size of unmasked tokens
            label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
            masked_lm_loss = torch.sum(loss) / torch.sum(label_size)
            ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))

        if not self.training and not validation: # inference
            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

        elif self.training: # training
            assert not validation
            res = {'all': masked_lm_loss, 'ppl': ppl_value, }
            return res

        else: # validation
            assert not self.training
            return loss, label_size

    def predict_strategy(self, logits, encoded_info):
        assert not self.training
        strat_id = encoded_info.get('strat_id', None)
        logits = logits[:, 0, -8:]
    
        if strat_id is not None:
            pred = strat_id
        else:
            if SAMPLE: #random strat
                filtered_logits = top_k_top_p_filtering(logits / TEMPERATURE, top_p=0.9)
                pred = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(-1)
            else: #
                pred = torch.argmax(logits, dim=-1)
        
        pred_top1 = torch.topk(logits, k=1, dim=-1)[1]
        pred_top3 = torch.topk(logits, k=3, dim=-1)[1]
    
        encoded_info.update({
            'pred_strat_id': pred,
            'pred_strat_id_top1': pred_top1,
            'pred_strat_id_top3': pred_top3,
            'pred_strat_id_dist': F.softmax(logits, dim=-1)
        })
    
    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        return_dict=None,
        **kwargs
    ):
        assert not self.training
        assert self.toker is not None
        
        encoded_info = kwargs
        assert decoder_input_ids.size(1) == 1
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias #?
        self.predict_strategy(lm_logits, encoded_info)
        
        decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.toker) - 8], dim=-1)
        
        assert 'max_length' in kwargs
        kwargs['max_length'] = kwargs['max_length'] + decoder_input_ids.size(1)
        kwargs['use_cache'] = True
        
        if len(self.toker) > self.toker.vocab_size:
            bad_words_ids = [[i] for i in range(self.toker.vocab_size, len(self.toker))]
            kwargs['bad_words_ids'] = bad_words_ids
        
        # *****????
        generations = super().generate(
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            **kwargs
        )
        return encoded_info, generations[:, decoder_input_ids.size(1):]