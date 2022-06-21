import json
from experiment import Experiment

filename = 'mazes'
with open('data/input/cfgs/'+filename+'_robust'+'.json') as f:
   load_file = json.load(f)
   cfg = load_file[filename][0]

for cfg in load_file[filename]:
   for policy in ["qmdp"]:
      cfg["policy"] = policy
      if not cfg.get("mdp_include"):
         cfg["mdp_include"] = False
      exp = Experiment(cfg["name"] + "_Robust_" +f'{cfg["p_init"]["sl"]}', cfg, 100)
      exp.execute(False)
