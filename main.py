import json
from experiment import Experiment

filename = 'maze2'
with open('data/input/cfgs/'+filename+'.json') as f:
   load_file = json.load(f)
   cfg = load_file[filename][0]

exp = Experiment(filename,cfg,10)

exp.execute(False)


