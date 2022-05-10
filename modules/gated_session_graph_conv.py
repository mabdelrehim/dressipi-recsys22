import imp
import torch
import torch_geometric as pyg

class GatedSessionGraphConv(pyg.nn.conv.MessagePassing):
  
  def __init__(self, out_channels, aggr: str = 'add', **kwargs):
    super().__init__(aggr=aggr, **kwargs)
    self.out_channels = out_channels
    self.gru = torch.nn.GRUCell(out_channels, out_channels, bias=False)
  
  def forward(self, x, edge_index):
    
    m = self.propagate(edge_index, x=x, size=None)
    x = self.gru(m, x)
    return x
  
  def message(self, x_j):
    return x_j
  
  def message_and_aggregate(self, adj_t, x):
    return torch.matmul(adj_t, x, reduce=self.aggr)