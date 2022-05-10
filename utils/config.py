import yaml

class Config(object):  
  """Simple dict wrapper that adds a thin API allowing for slash-based retrieval of
  nested elements, e.g. cfg.get_config("meta/dataset_name")
  """

  def __init__(self, config_path):
    with open(config_path) as cf_file:
      self.data = yaml.safe_load( cf_file.read() )
  
  def get(self, path=None, default=None):
    
    # we need to deep-copy self.data to avoid over-writing its data
    sub_dictionary = dict(self.data)
    
    if path is None:
      return sub_dictionary
    
    path_items = path.split("/")[:-1]
    data_item = path.split("/")[-1]
    
    try:
      recursive_dict = sub_dictionary
      for path_item in path_items:
        recursive_dict = recursive_dict.get(path_item)
      value = recursive_dict.get(data_item, default)
      return value
    except (TypeError, AttributeError):
      return default
  
  
  def _dict_flatten(self, in_dict, dict_out=None, parent_key=None, separator="."):
   if dict_out is None:
      dict_out = {}

   for k, v in in_dict.items():
      k = f"{parent_key}{separator}{k}" if parent_key else k
      if isinstance(v, dict):
         self._dict_flatten(in_dict=v, dict_out=dict_out, parent_key=k)
         continue

      dict_out[k] = v

   return dict_out


  def get_data(self, flatten=True):
    if flatten:
      return self._dict_flatten(self.data)
    else:
      return self.data