from skopt import gp_minimize
import numpy as np
from skopt.utils import use_named_args


def bayesian_search(Trainer,
                    search_space,
                    metric = None, 
                    n_calls = 20):
    #Maximize metric
    @use_named_args(dimensions=search_space)
    def evaluate_model(dimensions): 
        trainer = Trainer.model.initialize()

        performnce = Trainer.train()
        
        performance = [x['validation'] for x in performance]
        accuracy = np.mean([performance[fold][-1]['accuracy'] for fold in range(len(performance))])
        accuracy = np.mean(accuracy)
        return 1 - metric
    result = gp_minimize(evaluate_model, 
                        search_space,
                        n_calls = n_calls,
                        n_initial_points = 5)

    return result