import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import GatedSessionGraphConv

class SRGNN(nn.Module):
  
  def __init__(self, hidden_size, n_items, aggregation):
    
    super(SRGNN, self).__init__()
    self.hidden_size = hidden_size
    self.n_items = n_items
    self.embedding = nn.Embedding(self.n_items, self.hidden_size)
    self.gated = GatedSessionGraphConv(self.hidden_size, aggr=aggregation)
    self.q = nn.Linear(self.hidden_size, 1)
    self.W_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
    self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)

  def reset_parameters(self):
    stdv = 1.0 / torch.sqrt(self.hidden_size)
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)
  
  def forward(self, data):
    
    x, edge_index, batch_map = data.x, data.edge_index, data.batch
    embedding = self.embedding(x).squeeze()
    v_i = self.gated(embedding, edge_index)
    
    sections = list(torch.bincount(batch_map).cpu())
    v_i_split = torch.split(v_i, sections)
    v_n, v_n_repeat = [], []
    for session in v_i_split:
      v_n.append(session[-1])
      v_n_repeat.append(session[-1].view(1, -1).repeat(session.shape[0], 1))
    v_n, v_n_repeat = torch.stack(v_n), torch.cat(v_n_repeat, dim=0)
    
    q1 = self.W_1(v_n_repeat)
    q2 = self.W_2(v_i)
    alpha = self.q(F.sigmoid(q1 + q2))
    s_g_split = torch.split(alpha * v_i, sections)
    
    s_g = []
    for session in s_g_split:
      s_g_session = torch.sum(session, dim=0)
      s_g.append(s_g_session)
    s_g = torch.stack(s_g)
    s_l = v_n
    s_h = self.W_3(torch.cat([s_l, s_g], dim=-1))
    z = torch.mm(self.embedding.weight, s_h.T).T
    
    return z