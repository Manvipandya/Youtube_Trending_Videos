import json
from file import *

# load file paths
_fname = all_json()
_category_dict = {}
for _file in _fname:
    with open(_file) as f:
        _data = json.load(f)
        for i in range(len(_data['items'])):
            _channel_id = _data['items'][i]['id']
            _channel_cat = _data['items'][i]['snippet']['title']
            if _channel_id not in _category_dict:
                _category_dict.update({_channel_id: _channel_cat})


def category_dict():
    return _category_dict


def category_name(id):
    assert isinstance(id, int)
    return _category_dict[str(id)]
