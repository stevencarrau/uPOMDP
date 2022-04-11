from experiment import Experiment
import gridstorm

cfg = {'name':'maze2',
'formula_str':'Rmax=? [F "goal"]',
    'batch_size':128,
       'epochs':32, #
       'ctrx_gen':'crt_full', # Counterexample generation
       'rounds':10,
       'p_init':{'sl':0.2}, # First parameter initialization
       'p_evals':[0.1,0.2,0.3,0.4],
       'p_bounds':{'sl':[0.1,0.4]},
       'a_memory_dim':3,
       'bottleneck_dim':2, # Base for number of memory nodes
       'a_lr':0.001,
       'r_lr': 0.001,
       'length':10,
       'r_batch_size':128,
       'r_epochs':16,
       'a_batch_size':128,
       'a_epochs':16,
       'num_hx_qbns':8,
       'method': 'QBN',
       'memory_dim':4,
       'pso_n':500,
       'pso_fr':0.5,
       'blow_up':2,
'pso_ar':0.2,
'pso_rounds':5,
'pso_phi_g':0.5,
'pso_phi_r':0.5,
'batch_dim':4
}

exp = Experiment('test',cfg,10)

exp.execute(False)

