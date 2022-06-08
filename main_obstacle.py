import json
from experiment import Experiment

filename = 'obstacle'
with open('data/input/cfgs/'+filename+'.json') as f:
   load_file = json.load(f)
   cfg = load_file[filename][0]

for cfg in load_file[filename]:
   for policy in ["qmdp", "mdp"]:
      cfg["policy"] = policy
      if not cfg.get("mdp_include"):
         cfg["mdp_include"] = False
      exp = Experiment(cfg["name"] + "_Test_" + policy, cfg, 100)
      exp.execute(False)
