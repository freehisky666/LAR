# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KT_backbone(nn.Module):
    def __init__(self, skill_dim, answer_dim, hidden_dim, output_dim):
        super(KT_backbone, self).__init__()
        self.skill_dim=skill_dim
        self.answer_dim=answer_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(self.skill_dim*2+self.answer_dim*4, self.hidden_dim, batch_first=True)
        #self.rnn = nn.LSTM((self.skill_dim+self.answer_dim)*2, 80, batch_first=True,bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()
        ##
        self.fc1 = nn.Linear(self.hidden_dim, self.skill_dim+self.answer_dim)
        self.fc2 = nn.Linear(self.skill_dim+self.answer_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.skill_dim+self.answer_dim+hidden_dim, self.hidden_dim)
        self.relu = nn.functional.relu
        ##
        self.skill_emb = nn.Embedding(self.output_dim+1, self.skill_dim)
        self.skill_emb.weight.data[-1]= 0
        
        self.answer_emb = nn.Embedding(2+1, self.answer_dim)
        self.answer_emb.weight.data[-1]= 0
        
        self.attention_dim = 60
        self.attention_dim_s = 60
        self.mlp = nn.Linear(self.skill_dim+self.answer_dim, self.attention_dim)
        self.mlp2 = nn.Linear(self.answer_dim, self.attention_dim_s)
        self.similarity = nn.Linear(self.attention_dim, 1, bias=False)
        self.similarity2 = nn.Linear(self.attention_dim_s, 1, bias=False)
        
    def _get_next_pred(self, res, skill):
        
        one_hot = torch.eye(self.output_dim, device=res.device)
        one_hot = torch.cat((one_hot, torch.zeros(1, self.output_dim).to(device)), dim=0)
        next_skill = skill[:, 1:]
        one_hot_skill = F.embedding(next_skill, one_hot)
        
        pred = (res * one_hot_skill).sum(dim=-1)
        return pred
    
    def attention_module(self, lstm_output):
        
        att_w = self.mlp(lstm_output)
        att_w = torch.tanh(att_w)
        att_w = self.similarity(att_w)
        
        alphas=nn.Softmax(dim=1)(att_w)
        
        attn_ouput=alphas*lstm_output
        attn_output_cum=torch.cumsum(attn_ouput, dim=1)
        attn_output_cum_1=attn_output_cum-attn_ouput

        final_output=torch.cat((attn_output_cum_1, lstm_output),2)
        
        return final_output
    def attention_module2(self, lstm_output):
        
        att_w = self.mlp2(lstm_output)
        att_w = torch.tanh(att_w)
        att_w = self.similarity2(att_w)
        
        alphas=nn.Softmax(dim=1)(att_w)
        
        attn_ouput=alphas*lstm_output
        attn_output_cum=torch.cumsum(attn_ouput, dim=1)
        attn_output_cum_1=attn_output_cum-attn_ouput

        final_output=torch.cat((attn_output_cum_1, lstm_output),2)
        
        return final_output


    def forward(self, skill, answer, perturbation=None):
    #def forward(self, skill, answer, perturbation=None):
        
        skill_embedding=self.skill_emb(skill)
        answer_embedding=self.answer_emb(answer)
        
        skill_answer=torch.cat((skill_embedding,answer_embedding), 2)
        answer_skill=torch.cat((answer_embedding,skill_embedding), 2)
        
        answer=answer.unsqueeze(2).expand_as(skill_answer)
        #skill_answer_embedding=torch.where(answer==1, skill_answer, answer_skill)
        skill_answer_embedding=torch.where(answer==0, skill_answer, answer_skill)

        skill_answer_embedding1=skill_answer_embedding
        
        if  perturbation is not None:
            skill_answer_embedding+=perturbation
        
        
        out1=self.attention_module2(answer_embedding)
        out2=self.attention_module(skill_answer_embedding)
        out3=torch.cat((out1,out2), 2)
        
        #
        #out2=torch.cat((skill_answer_embedding,out1), 2)
        out4,_ = self.rnn(out3)
        '''
        if dropout_lb is True:
            dropout1=nn.Dropout(0.3)
            out_d=dropout1(out4)
        else:
            dropout1=nn.Dropout(0.3)
            out_d=dropout1(out4)
        '''
        #out5=self.relu(out_d)
        #out3=torch.cat((skill_answer_embedding,out2), 2)
        #out4 = self.fc3(out3)
        #out=self.fc2(skill_answer_embedding)
        
        #out6 = self.fc1(out5)
        #out7,_ = self.rnn(out6)
        #out,_ = self.rnn(out)
        ##
        
        #out=self.attention_module(skill_answer_embedding)
        #out,_ = self.rnn(out)
        #out = self.fc1(out)
        #res = self.sig(self.fc(out))
        ##
        res = self.sig(self.fc(out4))

        res = res[:, :-1, :]
        pred_res = self._get_next_pred(res, skill)
        
        return pred_res, skill_answer_embedding1
