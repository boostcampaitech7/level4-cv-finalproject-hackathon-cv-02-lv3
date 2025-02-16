import random
import warnings
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.event import Events, DEFAULT_EVENTS

class CustomBayesianOptimization(BayesianOptimization):
    """
    iter을 돌때마다 acquisition function을 랜덤으로 선택하기 위해 BayesianOptimization maximize 함수 커스텀 진행
    """

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acquisition_function=None,
                 acq=None,
                 kappa=None,
                 kappa_decay=None,
                 kappa_decay_delay=None,
                 xi=None,
                 **gp_params):

        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points)

        old_params_used = any([param is not None for param in [acq, kappa, kappa_decay, kappa_decay_delay, xi]])
        if old_params_used or gp_params:
            warnings.warn('\nPassing acquisition function parameters or gaussian process parameters to maximize'
                                     '\nis no longer supported, and will cause an error in future releases. Instead,'
                                     '\nplease use the "set_gp_params" method to set the gp params, and pass an instance'
                                     '\n of bayes_opt.util.UtilityFunction using the acquisition_function argument\n',
                          DeprecationWarning, stacklevel=2)
        
        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            acq_list = ["ucb", "ei", "poi"]
            selected_acq = random.choice(acq_list)
            if acquisition_function is None:
                util = UtilityFunction(kind=selected_acq,
                                    kappa=2.576,
                                    xi=0.1,                                 #0.0
                                    kappa_decay=1,
                                    kappa_decay_delay=0)
            else:
                util = acquisition_function
            
            try:
                x_probe = next(self._queue)
            except StopIteration:
                util.update_params()
                x_probe = self.suggest(util)
                iteration += 1
            self.probe(x_probe, lazy=False)

            if self._bounds_transformer and iteration > 0:
                # The bounds transformer should only modify the bounds after
                # the init_points points (only for the true iterations)
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

        self.dispatch(Events.OPTIMIZATION_END)