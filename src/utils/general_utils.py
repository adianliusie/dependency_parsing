import json

def flatten(x:list)->list:
    """ flattens list [[1,2,3],[4],[5,6]] to [1,2,3,4,5,6]"""
    return [tok for sent in x for tok in sent]

######## BASIC JSON SAVING AND LOADING FUNCTIONS #######################

def save_json(data:list, path:str):
    with open(path, "x") as outfile:
        json.dump(data, outfile, indent=2)

def load_json(path:str)->dict:
    with open(path) as jsonFile:
        data = json.load(jsonFile)
    return data
