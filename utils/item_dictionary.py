import json

class ItemDictionary():
  """
  A dictionary class to convert from item_id to index

  candidate items: item_ids to recommend from (4990 items in the dressipi dataset)
  item_features: the item_ids for remaining items in the dataset
  
  """
  def __init__(self, candidate_items_path, item_features_path):

    #with open(candidate_items_path) as json_file_candidate:
    #  candidate_items = json.load(json_file_candidate)
    with open(item_features_path) as json_file_item_feature:
      items = [0] # make an entry for fake item inserted at the beggining
      item_features = json.load(json_file_item_feature)
      items = items + [int(k) for k in item_features.keys()]
    
    # use itemfeatures only to create dict (results in a harder classification 
    # problem but the data labels will be balanced)
    
    #self.idx_to_item_candidate = {}
    #self.item_to_idx_candidate = {}
    #for i in range(len(candidate_items)):
    #  self.idx_to_item_candidate[i] = candidate_items[i]
    #  self.item_to_idx_candidate[candidate_items[i]] = i
    # items not in the candidate list are treated as others


    self.idx_to_item = {}
    self.item_to_idx = {}
    for i in range(len(items)):
      self.idx_to_item[i] = items[i]
      self.item_to_idx[items[i]] = i

  #def get_idx_candidate(self, item):
  #  return self.item_to_idx_candidate[item]

  #def get_item_candidate(self, idx):
  #  return self.idx_to_item_candidate[idx]

  def get_idx(self, item):
    return self.item_to_idx[item]

  def get_item(self, idx):
    return self.idx_to_item[idx]
