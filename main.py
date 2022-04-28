import json
from experiment import Experiment

filename = 'rocks2'
with open('data/input/cfgs/'+filename+'.json') as f:
   load_file = json.load(f)
   cfg = load_file[filename][0]

for policy in ["qmdp", "mdp"]:
   cfg["policy"] = policy
   exp = Experiment(filename + "_" + policy, cfg, 10)
   exp.execute(False)
