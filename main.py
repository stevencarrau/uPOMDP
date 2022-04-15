from experiment import Experiment
import stormpy

cfg = {'name':'maze2',
'formula_str':'Rmin=? [F "goal"]',
    'batch_size':128,
       'epochs':32, #
       'ctrx_gen':'crt_full', # Counterexample generation
       'rounds':10,
       'p_init':{'sl':0.2}, # First parameter initialization
       'p_evals': [{'sl': 0.1}, {'sl': 0.2}, {'sl': 0.3}, {'sl': 0.4}],
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
       'policy': 'qmdp', # mdp or qmdp
'pso_lr':0.1,
'pso_ar':0.2,
'pso_rounds':5,
'pso_phi_g':0.5,
'pso_phi_r':0.5,
'batch_dim':4
}


# prism_program = stormpy.parse_prism_program('data/input/envs/prism/maze2.prism')
# expression_manager = prism_program.expression_manager
# constants = prism_program.constants
# undefined_constants = []
# for c in constants:
#    if not c.defined:
#       undefined_constants.append(c.expression_variable)
#       # if c.name not in self.p_bounds:
#       #    raise ValueError(f'Parameter {c.name} appears in PRISM program, but no bounds were set.')
#
# options = stormpy.BuilderOptions([stormpy.parse_properties_without_context(cfg['formula_str'])[0].raw_formula])
# options.set_build_choice_labels()
# options.set_build_with_choice_origins()
# options.set_build_all_labels()
# options.set_build_all_reward_models()
# options.set_build_state_valuations()
#
# if prism_program.has_undefined_constants:
#    # model = stormpy.build_parametric_model(prism_program, properties = self.properties)
#    model = stormpy.build_sparse_parametric_model_with_options(prism_program, options)

# model = stormpy.pomdp.make_canonic(model)

exp = Experiment('test',cfg,10)

exp.execute(False)


