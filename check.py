import stormpy
from stormpy.utility import ShortestPathsGenerator
import numpy as np
from scipy.linalg import lu
from numpy.linalg import inv

import utils

class Checker:

    def __init__(self, instance, cfg):
        self.cfg = cfg
        self.instance = instance
        self.env = stormpy.Environment()
        self.env.solver_environment.set_linear_equation_solver_type(stormpy.EquationSolverType.native)
        self.env.solver_environment.native_solver_environment.method = stormpy.NativeLinearEquationSolverMethod.sound_value_iteration
        # self.env.solver_environment.native_solver_environment.precision = stormpy.Rational("0.9")
        # self.env.solver_environment.native_solver_environment.maximum_iterations = 30
        # self.env.solver_environment.set_force_sound(True)
        # self.env.solver_environment.native_solver_environment.set_force_sound(True)
        self.check_results = []

    def check_pdtmc(self, pdtmc):
        """
        Checks the parametric model of args for the specification.

        """

        if pdtmc.is_parametric:
            properties = pdtmc.properties[0].raw_formula
            region_checker = stormpy.pars.create_region_checker(
                self.env, pdtmc.model, properties, allow_model_simplification = False)
            check_lb = region_checker.get_bound_all_states(self.env, pdtmc.parameter_region, maximise = False)
            check_ub = region_checker.get_bound_all_states(self.env, pdtmc.parameter_region, maximise = True)

            lb_values = np.array(check_lb.get_values())
            ub_values = np.array(check_ub.get_values())

        else:
            check = stormpy.model_checking(pdtmc.model, pdtmc.properties[0])
            lb_values = np.array(check.get_values())
            ub_values = np.array(check.get_values())

        check_result = CheckResultWrapper(self.instance, self.cfg, pdtmc, lb_values, ub_values)
        self.check_results.append(check_result)
        return check_result


class CheckResultWrapper:
    def __init__(self, instance, cfg, pdtmc, lb_values, ub_values):

        self.instance = instance
        self.cfg = cfg
        self.pdtmc = pdtmc

        self._lb_values = lb_values
        self._ub_values = ub_values
        self._diffs = lb_values - ub_values

        self.lb_values = np.full((pdtmc.underlying_nS, pdtmc.underlying_nM), np.nan)
        self.ub_values = np.full((pdtmc.underlying_nS, pdtmc.underlying_nM), np.nan)
        for s in range(pdtmc.nS):
            state_label = pdtmc.state_labels[s]
            memory_label = pdtmc.memory_labels[s]
            if not isinstance(state_label, str):
                self.lb_values[state_label, memory_label] = lb_values[s]
                self.ub_values[state_label, memory_label] = ub_values[s]

        self.diffs = self.ub_values - self.lb_values

        if instance.objective == 'max':
            self.result_at_init = lb_values[0]
            self._values = self._lb_values
            # dirty fix to address WARNING GmmxLinearEquationSolver.
            # self.is_valid = self._values[0] <= self.instance.mdp.state_values[0]
        elif instance.objective == 'min':
            self.result_at_init = ub_values[0]
            self._values = self._ub_values
            # dirty fix to address WARNING GmmxLinearEquationSolver.
            # self.is_valid = self._values[0] >= self.instance.mdp.state_values[0]

        pomdp, label_to_reach = self.instance.pomdp, self.instance.label_to_reach
        (target_states, ) = np.where(pomdp.observation_labels[pomdp.O[pdtmc.state_labels]] == label_to_reach)
        self.max_distance = -1 # the higher the better.
        self.min_distance = 1 # the higher the better.
        self.bounded_reach_prob = -1
        # reach_formula_str = f"Pmax=?[F<=40 \"{label_to_reach}\"]" # causes memory leak.m
        # self.reasonability_property = stormpy.parse_properties_without_context(reach_formula_str)
        # result = stormpy.model_checking(pdtmc.model, self.reasonability_property[0])
        # values = [utils.value_to_float(x) for x in result.get_values()]
        # self.bounded_reach_prob = values[0] # bounded reachability probability.
        if not pdtmc.is_parametric: # perform reachability analysis.
            # reach_formula_str = f"R=?[I={4 * int(instance.mdp.state_values[0])}]"
            # check = stormpy.model_checking(pdtmc.model, self.reasonability_property[0].raw_formula)
            # values = check.get_values()
            # self.bounded_reach_prob = values[0] # bounded reachability probability.
            for s in range(pdtmc.nS):
                spg = ShortestPathsGenerator(pdtmc.model, s)
                distance = spg.get_distance(1)
                self.max_distance = max(distance, self.max_distance)
                self.min_distance = min(distance, self.min_distance)
        # self.critical_states, self.critical_memories = np.where(self.counter_examples)

    def worst_parameter_value(self, fsc, p_bounds):
        """
        Find the parameter value p* that induced the check result, i.e. is critical w.r.t. the specification.

        """

        p_range = [p_bounds[p][0] < p_bounds[p][1] for p in p_bounds]
        if not self.pdtmc.is_parametric or sum(p_range) == 0: # parametric or no range for p is given.
            return None, None

        if self.instance.name == 'simple':
            if fsc.action_distributions[0, 1, 0] > 0.50:
                return [0.45], self.result_at_init
            else:
                return [0.99], self.result_at_init

        self.instantiator = stormpy.pars.ModelInstantiator(self.pdtmc.model)
        ps = self.pdtmc.probability_parameters
        n = self.cfg['pso_n']
        particles = np.random.rand(n, len(ps))
        for idx, p in enumerate(p_bounds):
            particles[:, idx] = np.clip(particles[:, idx], p_bounds[p][0], p_bounds[p][1]) # clip to keep bounds of parameters.
        particle_fitnesses = self._evaluate(ps, particles)
        velocities = np.random.uniform(low = -0.99, high = 0.99, size = (n, len(ps)))

        if self.instance.objective == 'min':
            best_particle = particles[np.argmax(particle_fitnesses)] # adversarially chosen.
        else:
            raise NotImplementedError('PSO for maximization objectives is not implemented.')
        best_particle_positions = np.array(particles)

        lr = self.cfg['pso_lr'] # learning rate
        forget_rate = self.cfg['pso_fr'] # forget rate
        phi_g = self.cfg['pso_phi_g']
        phi_r = self.cfg['pso_phi_r']

        utils.inform('Starting PSO')
        for loop in range(self.cfg['pso_rounds']):
            new_particles = np.array(particles)
            new_velocities = np.array(velocities)
            for p, particle in enumerate(particles):
                r_p, r_g = np.random.uniform(size = len(ps)), np.random.uniform(size = len(ps))
                new_velocities[p] = forget_rate * velocities[p] + \
                    (phi_r * r_p * (best_particle_positions[p] - particle) + \
                        phi_g * r_g * (best_particle - particle))

            new_particles = particles + lr * new_velocities # was new_particles = particle + lr * new_velocities

            for idx, p in enumerate(p_bounds):
                new_particles[:, idx] = np.clip(new_particles[:, idx], p_bounds[p][0], p_bounds[p][1]) # clip to keep probabilities.
            new_particle_fitnesses = self._evaluate(ps, new_particles)

            if self.instance.objective == 'max':
                raise NotImplementedError('PSO for maximization objectives is not implemented.')
            else:
                best_particle = particles[np.argmax(particle_fitnesses)]
                (improvements, ) = np.where(new_particle_fitnesses > particle_fitnesses)
                best_particle_fitness = np.max(new_particle_fitnesses)

            best_particle_positions[improvements] = np.array(new_particles[improvements])

            particles = np.array(new_particles)
            velocities = np.array(new_velocities)

            if np.isclose(best_particle_fitness, self._values[0], atol = 0.02 * self._values[0]): # we have found an instantiation that yields lower bound.
                break

        if self.instance.objective == 'min':
            best_particle = particles[np.argmax(particle_fitnesses)]
            best_particle_fitness = np.max(particle_fitnesses)
        else:
            raise NotImplementedError('PSO for maximization objectives is not implemented.')
        return best_particle, best_particle_fitness

    def _evaluate(self, ps, particles):
        particle_fitnesses = np.full((len(particles)), np.nan)
        for p, particle in enumerate(particles):
            # pycarl.clear_pools()
            points = {p : stormpy.RationalRF(value) for p, value in zip(ps, particle)}
            model = self.instantiator.instantiate(points)
            check = stormpy.model_checking(model, self.instance.properties[0])
            values = np.array(check.get_values())
            particle_fitnesses[p] = values[0]
        return particle_fitnesses

    def evaluate(self, p_evals):
        """ Evaluates the value of the PDTMC for parameter values in args. """
        if not p_evals:
            return
        particles = []
        ps = self.pdtmc.probability_parameters
        self.instantiator = stormpy.pars.ModelInstantiator(self.pdtmc.model)
        values = []
        for p_eval in p_evals:
            particles = []
            for p in ps:
                particles.append(p_eval[p.name])
            value = self._evaluate(ps, [particles])
            values.append(value)
        return np.array(values)
