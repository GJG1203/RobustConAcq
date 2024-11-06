import time

from .algorithm_core import AlgorithmCAInteractive
from ..problem_instance import ProblemInstance
from ..answering_queries import Oracle, UserOracle, MisclassifyingUserOracle
from ..ca_environment.active_ca import ActiveCAEnv
from ..utils import get_kappa
from .. import Metrics


class RobustAcq(AlgorithmCAInteractive):
    """
    RobustAcq is a modified version of the QuAcq algorithm with additional robustness features.
    """

    def __init__(self, ca_env: ActiveCAEnv = None, stop_thresh=10, retrain_thresh=20):
        """
        Initialize the RobustAcq algorithm with optional thresholds.
        :param ca_env: An instance of CASystem, default is None.
        :param threshold1: Stopping threshold for convergence.
        :param threshold2: Size threshold for retraining classifier.
        """
        super().__init__(ca_env)
        self.stop_thresh = stop_thresh
        self.retrain_thresh = retrain_thresh
        self.stopping_threshold = 0
        self.Br = set()

    def retrain_classifier(self):
        """Retrain the classifier if |L| > retrain_thresh."""
        # Implement retraining logic here.
        pass

    def increase_stopping_threshold(self):
        """Increase the stopping threshold"""
        self.stopping_threshold += 1

    def learn(self, instance: ProblemInstance, oracle: Oracle = MisclassifyingUserOracle(), verbose=0, metrics: Metrics = None):
        self.env.init_state(instance, oracle, verbose, metrics)

        if len(self.env.instance.bias) == 0:
            self.env.instance.construct_bias()
            
        # Initialize the learned network L and sets B, Br
        L = self.env.instance.cl
        B = self.env.instance.bias
        self.Br = set()

        while True:
            if self.stopping_threshold > self.stop_thresh:
                return L  # Convergence condition

            if len(L) > self.retrain_thresh:
                self.retrain_classifier()  # Retrain classifier condition

            q1 = self.env.run_query_generation(L, B)
            if q1 is None:
                q2 = self.env.run_query_generation(L, self.Br)
                if q2 is None:
                    continue

                if self.env.ask_membership_query(q2):
                    self.increase_stopping_threshold()
                else:
                    scope = self.env.run_find_scope(q2)
                    c = self.env.run_findc(scope)
                    if c:
                        L.add(c)

            else:
                if self.env.ask_membership_query(q1):
                    # remove from B and add to Br
                    B.difference_update(self.get_kappa(B, q1))
                    self.Br.update(self.get_kappa(B, q1))
                else:
                    scope = self.env.run_find_scope(q1)
                    c = self.env.run_findc(scope)
                    if c:
                        L.add(c)