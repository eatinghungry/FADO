from os.path import join
import torch                                    # 导入torch
import torch.nn as nn                           # 导入torch.nn
import torch.nn.functional as F                 # 导入torch.nn.functional
import numpy as np                              # 导入numpy
from transformers import AutoModel, AutoTokenizer
from zmq import device
from torch.distributions import Categorical

N_ACTIONS = 8
LR = 0.01      

class Policy(nn.Module):
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
        predict = F.sofmax(predict)
        return predict

    
class Policy_Gradient(object):
    def __init__(self, model, toker, text_in_size=512, checkpt='./roberta-base', device='cuda', lr=LR):     
        self.model = model  
        self.loss_func = nn.MSELoss()
        self.tokenizer = toker#AutoTokenizer.from_pretrained(checkpt)
        self.model = model#AutoModel.from_pretrained(checkpt).to(device)                                               # 定义DQN的一系列属性
        #self.eval_net, self.target_net = Policy(text_in_size).to(device), QNet2(text_in_size).to(device)           
        #self.eval_net = QNet(text_in_size).to(device)
        self.agent = Policy(text_in_size).to(device)
        self.embed = self.model.get_strat_encoder()
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=LR)
        

    def choose_action(self, context, attention_mask, strat_hist, sentiment_hist,utterance_num, emotion, problem):   
        return_dict = self.model.config.use_return_dict
        embed = self.embed(
        input_ids=context,
        attention_mask=attention_mask,
        return_dict=return_dict,
        )[0].detach()        
        #embed = context                                    
        #if np.random.uniform() < EPSILON:                                       # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
        actions_value = self.agent.forward(embed, strat_hist, sentiment_hist,utterance_num, 
                                                emotion, problem, infer=True)             
        #action = torch.max(actions_value, 1)[1].data.numpy()                # 输出每一行最大值的索引，并转化为numpy ndarray形式
        _, actions = actions_value.max(1)
        #action = action[0]                                                  # 输出action的第一个数
        #urn action  
        return actions,actions_value.detach()

    def choose_action2(self, context, attention_mask, strat_hist, sentiment_hist,utterance_num, emotion, problem):   
        return_dict = self.model.config.use_return_dict
        embed = self.embed(
        input_ids=context,
        attention_mask=attention_mask,
        return_dict=return_dict,
        )[0].detach()        
        #embed = context                                    
        #if np.random.uniform() < EPSILON:                                       # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
        actions_value = self.agent.forward(embed, strat_hist, sentiment_hist,utterance_num, 
                                                emotion, problem, infer=True)             
        #action = torch.max(actions_value, 1)[1].data.numpy()                # 输出每一行最大值的索引，并转化为numpy ndarray形式
        #_, actions = actions_value.max(1)
        #action = action[0]                                                  # 输出action的第一个数
        #urn action  
        return actions_value 

    def save(self, output_dir, global_step):
        torch.save(self.agent.state_dict(), join(output_dir, f'DQN_{global_step}.bin'))
        torch.save(self.embed.state_dict(), join(output_dir, f'DQN_embed_{global_step}.bin'))


    def load(self, checkpoint, checkpoint2, device2):
        self.agent.load_state_dict(torch.load(checkpoint, map_location=device2))
        self.embed.load_state_dict(torch.load(checkpoint2, map_location=device2))

    def learn(self, context, attention_mask, strat_hist, sentiment_hist, V, strat_ids, utterance_num, emotion, problem):
        return_dict = self.model.config.use_return_dict
        embed = self.embed(
        input_ids=context,
        attention_mask=attention_mask,
        return_dict=return_dict,
        )[0]

        preds = self.agent(embed, strat_hist, sentiment_hist,utterance_num,emotion, problem)
        predict_prob = preds
        distr = Categorical(predict_prob)
        actions = distr.sample()
        log_prob = distr.log_prob(actions)

        returns = actions == strat_ids#).float()
        returns = returns.float()
        loss = -log_prob * returns.expand_as(log_prob)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.detach().cpu().numpy(), preds.detach()
