from os.path import join
import torch                                    # 导入torch
import torch.nn as nn                           # 导入torch.nn
import torch.nn.functional as F                 # 导入torch.nn.functional
import numpy as np                              # 导入numpy
from transformers import AutoModel, AutoTokenizer
from zmq import device
import copy


BATCH_SIZE = 32                                 # 样本数量
LR = 0.01                                       # 学习率
EPSILON = 0.9                                   # greedy policy
GAMMA = 0.9                                     # reward discount
N_ACTIONS = 8
# class Agent(nn.Module):
#     def __init__(self, text_in_size, out_size = 256, strat_in_size=5*8, sentiment_in_size=3*5, 
#                 sentiment_embed_size=64, text_embed_size=128, strat_embed_size=64):
#         super().__init__()                             
#         self.one_hot = nn.functional.one_hot()
#         self.text_linear = nn.Linear(text_in_size, text_embed_size)
#         self.strat_linear = nn.Linear(strat_in_size, strat_embed_size)
#         self.sentiment_linear = nn.Linear(sentiment_in_size, sentiment_embed_size)
#         self.mlp = nn.Linear(strat_embed_size + text_embed_size + sentiment_embed_size, out_size)
#         self.classifier = nn.Linear(out_size, strat_in_size)
#         self.activision = F.relu()

#     # def _init_weights(self, m):
#     #     if isinstance(m, nn.Linear):
#     #         trunc_normal_(m.weight, std=.02)
#     #         if isinstance(m, nn.Linear) and m.bias is not None:
    
#     def forward(self, context, strat_hist, sentiment_hist):
#         strat_hist = self.one_hot(strat_hist).reshape(strat_hist.shape[0], -1)
#         context, strat_hist = self.activision(self.text_linear(context)), self.activision(self.strat_linear(strat_hist))
#         sentiment_hist = self.activision(self.sentiment_linear(sentiment_hist))
#         x = torch.cat((context, strat_hist, sentiment_hist), axis=1)
#         x = self.activision(self.mlp(x))
#         out = self.classifier(x)

#         return out

# from transformers import BertForSequenceClassification
# class QNet(nn.Module):
#     def __init__(self, text_in_size, out_size = 1024, strat_in_size=5*9, sentiment_in_size=3*5, 
#             sentiment_embed_size=64, text_embed_size=1024, strat_embed_size=256):
#         super().__init__()                             
#         self.one_hot = nn.functional.one_hot
#         self.text_linear = nn.Linear(text_in_size, text_embed_size)
#         self.text_linear.weight.data.normal_(0, 0.1)
#         self.strat_linear = nn.Linear(strat_in_size, strat_embed_size)
#         self.strat_linear.weight.data.normal_(0, 0.1)
#         self.sentiment_linear = nn.Linear(sentiment_in_size, sentiment_embed_size)
#         self.sentiment_linear.weight.data.normal_(0, 0.1)
#         self.mlp = nn.Linear(strat_embed_size + text_embed_size + sentiment_embed_size, out_size)
#         self.mlp.weight.data.normal_(0, 0.1)
#         self.predict = nn.Linear(out_size, N_ACTIONS)
#         self.activision = F.relu

#     def forward(self, context, strat_hist, sentiment_hist):
#         strat_hist = self.one_hot(strat_hist,9).type(torch.float)#.reshape(strat_hist.shape[0], -1)
#         #print(f'asdfadfasdfas{strat_hist.shape}')
#         strat_hist = self.activision(self.strat_linear(strat_hist.reshape(strat_hist.shape[0], -1)))
#         strat_hist = strat_hist.reshape(strat_hist.shape[0], -1)
#         context = torch.mean(context, 1)
#         context = self.activision(self.text_linear(context))#.reshape(context.shape[0], -1)
#         sentiment_hist = sentiment_hist.reshape(sentiment_hist.shape[0], -1)
#         sentiment_hist = self.activision(self.sentiment_linear(sentiment_hist))
#         x = torch.cat((context, strat_hist, sentiment_hist), axis=1)
#         x = self.activision(self.mlp(x))
#         predict = self.predict(x)

#         return predict
from transformers import BertForSequenceClassification
class QNet(nn.Module):
    def __init__(self, text_in_size, out_size = 1024, strat_in_size=5*9, sentiment_in_size=3*5, 
            sentiment_embed_size=64, text_embed_size=1024, strat_embed_size=256, 
            utterance_embed_size=128, emotion_embed_size=64, prob_embed_size=64):
        super().__init__()                             
        self.one_hot = nn.functional.one_hot
        self.text_linear = nn.Linear(text_in_size, text_embed_size)
        self.text_linear.weight.data.normal_(0, 0.1)
        self.strat_linear = nn.Linear(strat_in_size, strat_embed_size)
        self.strat_linear.weight.data.normal_(0, 0.1)
        self.sentiment_linear = nn.Linear(sentiment_in_size, sentiment_embed_size)
        self.sentiment_linear.weight.data.normal_(0, 0.1)
        self.utterance_linear = nn.Linear(5, utterance_embed_size)
        self.utterance_linear.weight.data.normal_(0, 0.1)
        self.emotion_linear = nn.Linear(11, emotion_embed_size)
        self.emotion_linear.weight.data.normal_(0, 0.1)
        self.prob_linaer = nn.Linear(13, prob_embed_size)
        self.prob_linaer.weight.data.normal_(0, 0.1)
        
        self.mlp = nn.Linear(strat_embed_size + text_embed_size + utterance_embed_size + emotion_embed_size + prob_embed_size, out_size)
        #self.mlp = nn.Linear(text_embed_size + strat_embed_size + sentiment_embed_size, out_size)
        #self.mlp = nn.Linear(text_embed_size, out_size)
        self.mlp.weight.data.normal_(0, 0.1)
        self.predict = nn.Linear(out_size, N_ACTIONS)
        self.activision = F.relu
        self.drop = nn.Dropout(0.)

    def forward(self, context, strat_hist, sentiment_hist, utterance_num, emotion, problem, infer = False):
        strat_hist = self.one_hot(strat_hist,9).type(torch.float)#.reshape(strat_hist.shape[0], -1)
        #print(f'asdfadfasdfas{strat_hist.shape}')
        #utterance_num = torch.unsqueeze(utterance_num, 1)
        strat_hist = self.activision(self.strat_linear(strat_hist.reshape(strat_hist.shape[0], -1)))
        strat_hist = strat_hist.reshape(strat_hist.shape[0], -1)
        #print(utterance_num.shape, '!!!!!!!!!!DACD')
        utterance_num = self.activision(self.utterance_linear(utterance_num))
        emotion = self.activision(self.emotion_linear(emotion))
        problem = self.activision(self.prob_linaer(problem))
        context = torch.mean(context, 1)
        context = self.activision(self.text_linear(context))#.reshape(context.shape[0], -1)
        sentiment_hist = sentiment_hist.reshape(sentiment_hist.shape[0], -1)
        sentiment_hist = self.activision(self.sentiment_linear(sentiment_hist))
        x = torch.cat((context, strat_hist, utterance_num, emotion, problem), axis=1)
        #x = torch.cat((strat_hist, sentiment_hist), axis=1)
        #x = context
        if infer:
            x = self.activision(self.mlp(x))
        else:
            x = self.activision(self.drop(self.mlp(x)))
        predict = self.predict(x)

        return predict

class DQN(object):
    def __init__(self, model, toker, text_in_size=512, checkpt='./roberta-base', device='cuda', lr=LR):     
        self.model = model  
        self.loss_func = nn.MSELoss()
        self.tokenizer = toker#AutoTokenizer.from_pretrained(checkpt)
        #self.model = model#AutoModel.from_pretrained(checkpt).to(device)                                               # 定义DQN的一系列属性
        #self.eval_net, self.target = QNet(text_in_size).to(device), QNet(text_in_size).to(device)           
        self.eval_net = QNet(text_in_size).to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.parameters = self.eval_net.parameters()
        self.loss_func2 = nn.CrossEntropyLoss()
        self.embed = self.model.get_strat_encoder()
       # self.embed = self.model.get_encoder()
    def choose_action(self, context, attention_mask, strat_hist, sentiment_hist,utterance_num, emotion, problem):   
        return_dict = self.model.config.use_return_dict
        embed = self.embed(
        input_ids=context,
        attention_mask=attention_mask,
        return_dict=return_dict,
        )[0].detach()        
        #embed = context                                    
        #if np.random.uniform() < EPSILON:                                       # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
        actions_value = self.eval_net.forward(embed, strat_hist, sentiment_hist,utterance_num, 
                                                emotion, problem, infer=True)             
        #action = torch.max(actions_value, 1)[1].data.numpy()                # 输出每一行最大值的索引，并转化为numpy ndarray形式
        _, actions = actions_value.max(1)
        #action = action[0]                                                  # 输出action的第一个数
        #urn action  
        return actions   

    def choose_action2(self, context, attention_mask, strat_hist, sentiment_hist,utterance_num, emotion, problem):   
        return_dict = self.model.config.use_return_dict
        embed = self.embed(
        input_ids=context,
        attention_mask=attention_mask,
        return_dict=return_dict,
        )[0].detach()        
        #embed = context                                    
        #if np.random.uniform() < EPSILON:                                       # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
        actions_value = self.eval_net.forward(embed, strat_hist, sentiment_hist,utterance_num, 
                                                emotion, problem, infer=True)             
        #action = torch.max(actions_value, 1)[1].data.numpy()                # 输出每一行最大值的索引，并转化为numpy ndarray形式
        #_, actions = actions_value.max(1)
        #action = action[0]                                                  # 输出action的第一个数
        #urn action  
        return actions_value 


    def save(self, output_dir, global_step):
        torch.save(self.eval_net.state_dict(), join(output_dir, f'DQN_{global_step}.bin'))
        torch.save(self.embed.state_dict(), join(output_dir, f'DQN_embed_{global_step}.bin'))


    def load(self, checkpoint, checkpoint2, device2):
        self.eval_net.load_state_dict(torch.load(checkpoint, map_location=device2))
        self.embed.load_state_dict(torch.load(checkpoint2, map_location=device2))

    def learn(self, context, attention_mask, strat_hist, sentiment_hist, V, strat_ids, utterance_num, emotion, problem):    
        #inputs = self.tokenizer(context)#.to('cuda')
        #context = self.tokenizer(context)
        #embed = self.model.get_encoder()(context)
        return_dict = self.model.config.use_return_dict
        embed = self.embed(
        input_ids=context,
        attention_mask=attention_mask,
        return_dict=return_dict,
        )[0]#.detach()
        #print(f'shape of embed: {embed.shape}, shape of inputs: {context.shape}')
        #embed = self.model(context).last_hidden_state #detach?
        
        preds = self.eval_net(embed, strat_hist, sentiment_hist,utterance_num,emotion, problem) #16,8
        strat_act = strat_ids.clone()
        for i in range(strat_act.shape[0]):
            if np.random.uniform() < EPSILON:
                strat_act[i] = preds[i].argmax()
                #pass
                
            else:
                strat_act[i] = np.random.randint(0, N_ACTIONS)  

        #q_eval = preds.gather(1, strat_act.unsqueeze(0)) 
        #q_eval = preds.gather(1, strat_ids.unsqueeze(0)) 
        #q_next = self.target_net()
        # _, pred_idx = preds.max(1)
        # match = (pred_idx==strat_ids).data
        # reward = sparse_reward#torch.ones(targets.shape)
        # reward[~match] = penalty
        # direct_reward = 
        target = torch.zeros((preds.shape[0], N_ACTIONS)).to(self.device)
        #target2 = torch.zeros((preds.shape[0], N_ACTIONS)).to(self.device)
        for j in  range(preds.shape[0]):
            target[j, strat_ids[j]] = 10+V[j]#V[j]
            #target2[j , strat_ids[j]] = 1
        #V[strat_act != strat_ids] *= 0#V[strat_act == strat_ids] += 5 #direct reward, when the preds act equals to strat_ids, 
        # print('gggg', strat_ids)
        # print('hhhh', target)
        loss = self.loss_func(preds, target) # q_eval:1, 16, V:16
        #print('sdfadf', target2)
        #loss2 = self.loss_func2(preds, strat_ids)#target2.type(torch.LongTensor).to(self.device))

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        #loss2.backward()
        self.optimizer.step()
        #print(f'DQN LOSSS:{loss.detach().cpu().numpy()}')
        return loss.detach().cpu().numpy()



class QNet2(nn.Module):
    def __init__(self, text_in_size, out_size = 1024, strat_in_size=5*9, sentiment_in_size=3*5, 
            sentiment_embed_size=64, text_embed_size=1024, strat_embed_size=256):
        super().__init__()                             
        self.one_hot = nn.functional.one_hot
        self.text_linear = nn.Linear(text_in_size, text_embed_size)
        self.text_linear.weight.data.normal_(0, 0.1)
        self.strat_linear = nn.Linear(strat_in_size, strat_embed_size)
        self.strat_linear.weight.data.normal_(0, 0.1)
        self.sentiment_linear = nn.Linear(sentiment_in_size, sentiment_embed_size)
        self.sentiment_linear.weight.data.normal_(0, 0.1)
        #self.mlp = nn.Linear(strat_embed_size + sentiment_embed_size, out_size)
        self.mlp = nn.Linear(text_embed_size + strat_embed_size + sentiment_embed_size, out_size)
        #self.mlp = nn.Linear(text_embed_size, out_size)
        self.mlp.weight.data.normal_(0, 0.1)
        self.predict = nn.Linear(out_size, N_ACTIONS)
        self.activision = F.relu

    def forward(self, context, strat_hist, sentiment_hist):
        strat_hist = self.one_hot(strat_hist,9).type(torch.float)#.reshape(strat_hist.shape[0], -1)
        #print(f'asdfadfasdfas{strat_hist.shape}')
        strat_hist = self.activision(self.strat_linear(strat_hist.reshape(strat_hist.shape[0], -1)))
        strat_hist = strat_hist.reshape(strat_hist.shape[0], -1)
        context = torch.mean(context, 1)
        context = self.activision(self.text_linear(context))#.reshape(context.shape[0], -1)
        sentiment_hist = sentiment_hist.reshape(sentiment_hist.shape[0], -1)
        sentiment_hist = self.activision(self.sentiment_linear(sentiment_hist))
        x = torch.cat((context, strat_hist, sentiment_hist), axis=1)
        #x = torch.cat((strat_hist, sentiment_hist), axis=1)
        #x = context
        x = self.activision(self.mlp(x))
        predict = self.predict(x)

        return predict

class DQNRL(object):
    def __init__(self, model, toker, text_in_size=512, checkpt='./roberta-base', device='cuda'):     
        self.model = model  
        self.loss_func = nn.MSELoss()
        self.tokenizer = toker#AutoTokenizer.from_pretrained(checkpt)
        self.model = model#AutoModel.from_pretrained(checkpt).to(device)                                               # 定义DQN的一系列属性
        self.eval_net, self.target_net = QNet2(text_in_size).to(device), QNet2(text_in_size).to(device)           
        #self.eval_net = QNet(text_in_size).to(device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)

    def choose_action(self, context, attention_mask, strat_hist, sentiment_hist):   
        return_dict = self.model.config.use_return_dict
        embed = self.model.get_encoder()(
        input_ids=context,
        attention_mask=attention_mask,
        return_dict=return_dict,
        )[0]        
        #embed = context                                    
        #if np.random.uniform() < EPSILON:                                       # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
        actions_value = self.eval_net.forward(embed, strat_hist, sentiment_hist)                
        #action = torch.max(actions_value, 1)[1].data.numpy()                # 输出每一行最大值的索引，并转化为numpy ndarray形式
        _, actions = actions_value.max(1)
        #action = action[0]                                                  # 输出action的第一个数
        #urn action  
        return actions   

    def save(self, output_dir, global_step):
        torch.save(self.eval_net.state_dict(), join(output_dir, f'DQN_{global_step}.bin'))

    def learn(self, context, attention_mask, strat_hist, sentiment_hist, reward, strat_ids, 
              next_sentence, attention_mask_nx, next_strat_hist, next_sentiment_hist):    
        #inputs = self.tokenizer(context)#.to('cuda')
        #context = self.tokenizer(context)
        #embed = self.model.get_encoder()(context)
        return_dict = self.model.config.use_return_dict
        embed = self.model.get_encoder()(
        input_ids=context,
        attention_mask=attention_mask,
        return_dict=return_dict,
        )[0]
        #embed = self.model(context).last_hidden_state #detach?
        
        q_eval = self.eval_net(embed, strat_hist, sentiment_hist) #16,8
        strat_act = strat_ids.clone()
        for i in range(strat_act.shape[0]):
            if np.random.uniform() < EPSILON:
                strat_act[i] = strat_ids[i]#q_eval[i].argmax()
                reward[i] += 1
                #pass
                
            else:
                strat_act[i] = np.random.randint(0, N_ACTIONS)  
                reward[i] = 0

        q_eval = q_eval.gather(1, strat_act.unsqueeze(0)) 

        embed_next = self.model.get_encoder()(
        input_ids=next_sentence,
        attention_mask=attention_mask_nx,
        return_dict=return_dict,
        )[0]

        q_next = self.target_net(embed_next, next_strat_hist, next_sentiment_hist).detach()

        #reward shape: (batch_size,)
        q_target = reward + GAMMA * q_next.max(1)[0].view(1,-1)

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #print(f'DQN LOSSS:{loss.detach().cpu().numpy()}')
        return loss.detach().cpu().numpy()