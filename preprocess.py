import pandas as pd
import argparse
import json
import os
import random
from sklearn.utils import shuffle
from tqdm.auto import tqdm
from utils import NpEncoder, Config
from sklearn.model_selection import train_test_split


def get_args_parser():
  parser = argparse.ArgumentParser(add_help=False)

  parser.add_argument("--config", 
                          type=str, 
                          default="config/data/preprocess.yaml",
                          help="path to yaml configuration file")
  

  return parser

def get_sessions(sessions, purchases=None):
  """
  Given a sessions data as dataframe, constructs a list of lists where each 
  sublist has 'item_id' for each item viewed in a session for each 'session_id'. 
  Items in each sublist are sorted according to 'date'. For training set,
  constructs a list of items where each element is the item purchased for
  the corresponding session

  parameters:
    sessions: pandas dataframe for sessions data with columns 
    [session_id, item_id, date]

    purchases(optional): pandas dataframe for session purchases with columns
    [session_id, item_id, date]. Default value is None

  Returns:
    List[dict]: session sequences 
      a list of dicts where each dict contains:
        {
          id: session_id
          items: items viewed in this session in cronological order
          purchase: purchased item for this session (this attribute is present only for training data)
        }
  
  """
  session_sequences = []
  sessions_grouped = sessions.groupby(['session_id'])
  for session, session_data in tqdm(sessions_grouped, desc='Preprocessing sessions data'):
    session_data = session_data.sort_values(by=['date'], inplace=False, ascending=True)
    session_sequences.append({
      'id': int(session),
      'items': list(session_data['item_id'])
    })
    if purchases is not None:
      session_sequences[-1]['purchase'] = purchases[purchases['session_id'] == session]['item_id'].values[0]
  return session_sequences

def get_item_features(items):
  """
  Given item features as a dataframe constructs a dict where each key is the item id and each value is
  a dict with category ids as keys and values for thes categories as values
  e.g. {
    1: {
      3: 6,
      2: 8
    }-
  }

  parameters:
    items: pandas dataframe for item features data with columns 
    [item_id, feature_category_id, feature_value_id]

  Returns:
    dict: category information for each item. see e.g. above
  
  """
  items_dict = {}
  item_grouped_features = items.groupby(['item_id'])
  for id, attributes in tqdm(item_grouped_features, desc='Preprocessing item features'):
    items_dict[int(id)] = {}
    for _, row in attributes.iterrows():
      items_dict[int(id)][int(row['feature_category_id'])] = int(row['feature_value_id'])
  return items_dict

def get_candidate_items(items):
  return list(items['item_id'])

def preprocess_dressipi(cfg):

  """
  download dataset and extract to [dataset_root] directory
  from: https://www.dressipi-recsys2022.com/ 
  """
  
  items_features_path = os.path.join(cfg.get('root'), 'item_features.csv')
  candidate_items_path = os.path.join(cfg.get('root'), 'candidate_items.csv')
  train_sessions_path = os.path.join(cfg.get('root'), 'train_sessions.csv')
  train_purchases_path = os.path.join(cfg.get('root'), 'train_purchases.csv')
  test_leaderboard_sessions_path = os.path.join(cfg.get('root'), 'test_leaderboard_sessions.csv')
  test_final_sessions_path = os.path.join(cfg.get('root'), 'test_final_sessions.csv')

  train_sessions = pd.read_csv(train_sessions_path)
  train_purchases = pd.read_csv(train_purchases_path)
  item_features = pd.read_csv(items_features_path)
  test_leaderboard_sessions = pd.read_csv(test_leaderboard_sessions_path)
  test_final_sessions = pd.read_csv(test_final_sessions_path)
  candidate_items = pd.read_csv(candidate_items_path)

  preprocessed_dir = os.path.join(cfg.get('root'), 'preprocessed')
  if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)
  
  print("Preprosessing training dataset")
  preprocessed_train_data = get_sessions(train_sessions, train_purchases)
  random.seed(cfg.get('seed'))
  random.shuffle(preprocessed_train_data)
  test_candidates = set(candidate_items['item_id'].to_list())
  test_valid_size = cfg.get('test_valid_size')
  test_valid_split = []
  train_split = []
  for i, session in enumerate(preprocessed_train_data):
    if session['purchase'] in test_candidates and i < test_valid_size:
      test_valid_split.append(session)
    else:
      train_split.append(session)
  test_local, valid_data = train_test_split(test_valid_split, test_size=0.285714286, random_state=cfg.get('seed'))
  json_data = json.dumps(train_split, indent=2, cls=NpEncoder)
  out_file_path = os.path.join(preprocessed_dir, 'train.json')
  with open(out_file_path, 'w') as out_file:
    out_file.write(json_data)
  json_data = json.dumps(valid_data, indent=2, cls=NpEncoder)
  out_file_path = os.path.join(preprocessed_dir, 'valid.json')
  with open(out_file_path, 'w') as out_file:
    out_file.write(json_data)
  json_data = json.dumps(test_local, indent=2, cls=NpEncoder)
  out_file_path = os.path.join(preprocessed_dir, 'test_local.json')
  with open(out_file_path, 'w') as out_file:
    out_file.write(json_data)

  print("Preprosessing test leaderboard dataset")
  preprocessed_test_leaderboard_data = get_sessions(test_leaderboard_sessions)
  json_data = json.dumps(preprocessed_test_leaderboard_data, indent=2, cls=NpEncoder)
  out_file_path = os.path.join(preprocessed_dir, 'test_leaderboard.json')
  with open(out_file_path, 'w') as out_file:
    out_file.write(json_data)

  print("Preprosessing test final dataset")
  preprocessed_test_final_data = get_sessions(test_final_sessions)
  json_data = json.dumps(preprocessed_test_final_data, indent=2, cls=NpEncoder)
  out_file_path = os.path.join(preprocessed_dir, 'test_final.json')
  with open(out_file_path, 'w') as out_file:
    out_file.write(json_data)

  print("Preprosessing item features")
  preprocessed_item_features = get_item_features(item_features)
  json_data = json.dumps(preprocessed_item_features, indent=2, cls=NpEncoder)
  out_file_path = os.path.join(preprocessed_dir, 'item_features.json')
  with open(out_file_path, 'w') as out_file:
    out_file.write(json_data)

  print("Preprosessing candidate items")
  preprocessed_candidate_items = get_candidate_items(candidate_items)
  json_data = json.dumps(preprocessed_candidate_items, indent=2, cls=NpEncoder)
  out_file_path = os.path.join(preprocessed_dir, 'candidate_items.json')
  with open(out_file_path, 'w') as out_file:
    out_file.write(json_data)

def main(cfg):
  if cfg.get('dataset') == "dressipi_recsys22":
    preprocess_dressipi(cfg)

if "__main__" in __name__:
  parser = argparse.ArgumentParser('Preprocess session data from csv format to json format.', parents=[get_args_parser()])
  args = parser.parse_args()
  cfg = Config(args.config)
  main(cfg)