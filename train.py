import torch
import wandb
import argparse

from utils import Config
from data import  SessionGraphDataset
from models import SRGNN
from tqdm.auto import tqdm
import torch_geometric.data as pyg_data

def get_args_parser():
  parser = argparse.ArgumentParser(add_help=False)

  parser.add_argument("--config", 
                          type=str, 
                          default="config/data/preprocess.yaml",
                          help="path to yaml configuration file")

  return parser


class SRGNNTrainer():
  
  def __init__(self, 
                train_dataloader,
                valid_dataloader,
                model,
                optimizer,
                scheduler,
                criterion,
                device):
      self.train_dataloader = train_dataloader
      self.valid_dataloader = valid_dataloader
      self.model = model
      self.optimizer = optimizer
      self.scheduler = scheduler
      self.criterion = criterion
      self.device = device

      self.model.to(self.device)

  def train_step(self, batch):
    
    batch.to(self.device)
    self.optimizer.zero_grad()
    scores = self.model(batch)
    label = batch.y
    loss = self.criterion(scores, label)
    loss.backward()
    self.optimizer.step()
    return loss, scores


  def valid_step(self, batch):
    batch.to(self.device)
    scores = self.model(batch)
    label = batch.y
    loss = self.criterion(scores, label)
    return loss, scores


  def train_for_epoch(self):
    self.model.train()
    total_loss = 0
    for _, batch in enumerate(tqdm(self.train_dataloader)):
      loss, scores = self.train_step(batch)
      wandb.log({"loss": loss})
      total_loss += loss.item() * batch.num_graphs
    total_loss /= len(self.train_dataloader.dataset)
    return total_loss
      

  def valid(self):
    self.model.eval()
    total_loss = 0
    for _, batch in enumerate(tqdm(self.valid_dataloader)):
      with torch.no_grad():
        loss, scores = self.valid_step(batch)
        total_loss += loss.item() * batch.num_graphs
    total_loss /= len(self.valid_dataloader.dataset)
    return total_loss

  def train(self, num_epochs, save_dir):
    wandb.log({
      'training_samples': len(self.train_dataloader.dataset),
      'validation_samples': len(self.valid_dataloader.dataset)
    })

    best_valid_loss = float('inf')
    for i in range(num_epochs):
      train_loss = self.train_for_epoch()
      valid_loss = self.valid()
      if valid_loss < best_valid_loss:
        print("Saving Checkpoint")
        torch.save({
            'epoch': i,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': train_loss,
            }, f"{save_dir}/model.pt")
        best_valid_loss = valid_loss

      print(f"Epoch{i}/{num_epochs}: Train loss: {train_loss}, Valid loss: {valid_loss}, Best Valid Loss: {best_valid_loss}")
      wandb.log({
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'best_valid_loss': best_valid_loss
      })


def main(cfg):

  wandb.init(project="dressipi-recsys22", entity="mabdelrehim")
  seed = cfg.get('seed')
  torch.manual_seed(seed)
  wandb.log({
    'loaded_configs': cfg.data,
    'seed': seed
  })

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
    aggr = cfg.get('model/aggregation')
    model = SRGNN(hidden_dim, num_items, aggr)
    wandb.log({
      'model': str(model),
      'device': device
    })

  ############## training related arguments ##############

  if cfg.get('training/optimizer/name') == 'adam':
    lr = cfg.get('training/optimizer/lr')
    weight_decay = cfg.get('training/optimizer/weight_decay')
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=lr,
                                  weight_decay=weight_decay)

  if cfg.get('training/lr_scheduler/name') == 'step_lr':
    step_size = cfg.get('training/lr_scheduler/step_size')
    gamma = cfg.get('training/lr_scheduler/gamma')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                          step_size=step_size,
                                          gamma=gamma)

  if cfg.get('training/criterion/name') == 'cross_entropy_loss':
    criterion = torch.nn.CrossEntropyLoss()

  if dataset_implementation == 'session_graph_dataset':
    data_root = cfg.get('dataset/path')
    train_split = cfg.get('dataset/train_split')
    valid_split = cfg.get('dataset/valid_split')
    candidate_items_path = cfg.get('dataset/candidate_items_path')
    item_features_path = cfg.get('dataset/item_features_path')
    train_dataset = SessionGraphDataset(data_root, train_split, candidate_items_path, item_features_path)
    valid_dataset = SessionGraphDataset(data_root, valid_split, candidate_items_path, item_features_path)
    

  batch_size = cfg.get('training/batch_size')
  
  train_loader = pyg_data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       drop_last=True)
  val_loader = pyg_data.DataLoader(valid_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     drop_last=True)
  wandb.config = cfg.get_data()

  num_epochs = cfg.get('training/num_epochs')
  save_dir = cfg.get('model/save_dir')
  trainer = SRGNNTrainer(
    train_dataloader=train_loader,
    valid_dataloader=val_loader,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    device=device
  )
  trainer.train(num_epochs, save_dir)

if __name__ == "__main__":
  parser = argparse.ArgumentParser('Preprocess session data from csv format to json format.', parents=[get_args_parser()])
  args = parser.parse_args()
  cfg = Config(args.config)
  main(cfg)
