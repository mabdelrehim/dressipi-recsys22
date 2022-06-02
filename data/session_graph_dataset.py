import torch_geometric.data as pyg_data
import json
import os
import torch
import pandas as pd

from utils import ItemDictionary

class SessionGraphDataset(pyg_data.InMemoryDataset):
  def __init__(self, root, file_name, candidate_items_path, item_features_path, transform=None, pre_transform=None, test=False):
    self.file_name = file_name
    self.items_dict = ItemDictionary(candidate_items_path, item_features_path)
    self.test = test
    super().__init__(root, transform, pre_transform)
    self.data, self.slices = torch.load(self.processed_paths[0])
  
  @property
  def raw_file_names(self):
    return [f'{self.file_name}.json']
  
  @property
  def processed_file_names(self):
    return [f'{self.file_name}.pt']
  
  def download(self):
    pass
  
  def process(self):
    
    raw_data_file = os.path.join(self.raw_dir, self.raw_file_names[0])
    with open(raw_data_file) as json_file:
      sessions = json.load(json_file)
    
    data_list = []
    if not self.test:
      for session in sessions:
        session, y = session['items'], session['purchase']
        
        session = [0] + session
        for i in range(len(session)):
          session[i] = self.items_dict.get_idx(session[i])
        y = self.items_dict.get_idx(y)

        codes, uniques = pd.factorize(session)
        senders, receivers = codes[:-1], codes[1:]

        # Build Data instance
        edge_index = torch.tensor([senders, receivers], dtype=torch.long)
        x = torch.tensor(uniques, dtype=torch.long).unsqueeze(1)
        y = torch.tensor([y], dtype=torch.long)
        data_list.append(pyg_data.Data(x=x, edge_index=edge_index, y=y))

      data, slices = self.collate(data_list)
      torch.save((data, slices), self.processed_paths[0])
    else:
      for session in sessions:
        session = session['items']
        
        session = [0] + session
        for i in range(len(session)):
          session[i] = self.items_dict.get_idx(session[i])

        codes, uniques = pd.factorize(session)
        senders, receivers = codes[:-1], codes[1:]

        # Build Data instance
        edge_index = torch.tensor([senders, receivers], dtype=torch.long)
        x = torch.tensor(uniques, dtype=torch.long).unsqueeze(1)
        data_list.append(pyg_data.Data(x=x, edge_index=edge_index))

      data, slices = self.collate(data_list)
      torch.save((data, slices), self.processed_paths[0])