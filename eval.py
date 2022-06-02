import json
import torch
import wandb
import argparse

from utils import Config
from data import  SessionGraphDataset
from models import SRGNN
from tqdm.auto import tqdm
import torch_geometric.data as pyg_data
import pandas as pd

def get_args_parser():
  parser = argparse.ArgumentParser(add_help=False)

  parser.add_argument("--config", 
                          type=str, 
                          default="config/data/preprocess.yaml",
                          help="path to yaml configuration file")

  return parser


def predict_step(model, batch, device):
  batch.to(device)
  scores = model(batch)
  _, indices = torch.topk(scores, 100)
  return indices

def test(loader, model, device, sessions_ids, save_path):
  predictions = {}
  predictions['session_id'] = []
  predictions['item_id'] = []
  predictions['rank'] = []
  model.to(device)
  model.eval()
  idx_to_item = loader.dataset.items_dict.idx_to_item
  print(len(sessions_ids))
  print(len(loader))
  for session_id, session in zip(sessions_ids, tqdm(loader)):
      with torch.no_grad():
        pred_items = predict_step(model, session, device).view(-1).cpu().detach().numpy()
        for rank, pred_item in enumerate(pred_items):
          predictions['session_id'].append(session_id['id'])
          predictions['item_id'].append(idx_to_item[int(pred_item)])
          predictions['rank'].append(rank + 1)
  pred_df = pd.DataFrame.from_dict(predictions)
  pred_df.sort_values(['session_id'])
  pred_df.to_csv(save_path, index=False)
          


def main(cfg):
  seed = cfg.get('seed')
  torch.manual_seed(seed)

  device = cfg.get('device', 'cpu')
  if device == 'gpu' and torch.cuda.is_available():
    device = 'cuda'
  else:
    device = 'cpu'

  ############## dataset related arguments ##############

  dataset = cfg.get('dataset/name')
  dataset_implementation = cfg.get('dataset/implementation')
  num_items = cfg.get('dataset/num_items')

  ############## model related arguments ##############

  if cfg.get('model/name') == 'srgnn':
    hidden_dim = cfg.get('model/hidden_dim')
    checkpoint_path = cfg.get('model/checkpoint')
    checkpoint = torch.load(checkpoint_path)
    aggr = cfg.get('model/aggregation')
    model = SRGNN(hidden_dim, num_items, aggr)
    model.load_state_dict(checkpoint['model_state_dict'])

  if dataset_implementation == 'session_graph_dataset':
    data_root = cfg.get('dataset/path')
    test_split = cfg.get('dataset/test_split')
    candidate_items_path = cfg.get('dataset/candidate_items_path')
    item_features_path = cfg.get('dataset/item_features_path')
    test_dataset =  SessionGraphDataset(data_root, test_split, candidate_items_path, item_features_path, test=True)

  batch_size = cfg.get('batch_size')
  test_loader = pyg_data.DataLoader(test_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       drop_last=True)
  with open(cfg.get('dataset/sessions')) as sess_data:
    sessions = json.load(sess_data)
  save_path = cfg.get('save_path')
  test(test_loader, model, device, sessions, save_path)

if __name__ == "__main__":
  parser = argparse.ArgumentParser('Testing', parents=[get_args_parser()])
  args = parser.parse_args()
  cfg = Config(args.config)
  main(cfg)
