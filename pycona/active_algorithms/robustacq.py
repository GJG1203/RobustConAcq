import time

from .algorithm_core import AlgorithmCAInteractive
from ..problem_instance import ProblemInstance
from ..answering_queries import Oracle, UserOracle, MisclassifyingUserOracle
from ..ca_environment.active_ca import ActiveCAEnv
from ..ca_environment.active_ca_proba import ProbaActiveCAEnv
from ..utils import get_kappa
from .. import Metrics


class RobustAcq(AlgorithmCAInteractive):
    """
    RobustAcq is a modified version of the QuAcq algorithm with additional robustness features.
    """

    def __init__(self, ca_env: ProbaActiveCAEnv = None, stop_thresh=10, retrain_thresh=20, confidence_thresh=0.8):
        """
        Initialize the RobustAcq algorithm with optional thresholds.
        :param ca_env: An instance of CASystem, default is None.
        :param threshold1: Stopping threshold for convergence.
        :param threshold2: Size threshold for retraining classifier.
        """
        super().__init__(ca_env if ca_env is not None else ProbaActiveCAEnv())
        self.stop_thresh = stop_thresh
        self.retrain_thresh = retrain_thresh
        self.stopping_threshold = 0
        self.confidence_thresh = confidence_thresh
        self.Br = set()

    def retrain_classifier(self):
        """Retrain the classifier and reclassify constraints in Br."""
        # Use ProbaActiveCAEnv's training functionality to retrain on current constraints
        self.env._train_classifier()

        # Reclassify constraints in Br based on new predictions
        self.env._predict_bias_proba()
        self._reclassify_constraints()
        
    def _reclassify_constraints(self):
        """Move constraints back from Br to B if the classifier is confident they were misclassified."""
        constraints_to_move = set()
        for constraint in self.Br:
            # Get the feature representation for the constraint
            features = self.env.feature_representation.featurize_constraint(constraint)
            # Get classifier prediction probability
            prob = self.env.classifier.predict_proba([features])[0][0]  # Assuming class 0 is 'doesn't belong in Br'
            
            # If classifier is confident it's misclassified, mark it for moving
            if prob >= self.confidence_thresh:
                constraints_to_move.add(constraint)
        
        # Move the marked constraints from Br back to B
        self.remove_constraints_Br(constraints_to_move)
        self.env.instance.bias.extend(constraints_to_move)  # Add back to B
        
    def remove_constraints_Br(self, cons):
        return [item for item in self.Br if item not in cons]
    
    def remove_constraints_bias(self, cons):
        return [item for item in self.env.instance.bias if item not in cons]
        
    def increase_stopping_threshold(self):
        """Increase the stopping threshold"""
        self.stopping_threshold += 1

    def learn(self, instance: ProblemInstance, oracle: Oracle = MisclassifyingUserOracle(), verbose=0, metrics: Metrics = None):
        self.env.init_state(instance, oracle, verbose, metrics)

        if len(self.env.instance.bias) == 0:
            self.env.instance.construct_bias()
            for c in self.env.instance.bias:
                print(c)
            
        # Initialize Br
        self.Br = []

        while True:
            if self.stopping_threshold > self.stop_thresh:
                return self.env.instance.cl  # Convergence condition

            if len(self.env.instance.cl) > self.retrain_thresh:
                self.retrain_classifier()  # Retrain classifier condition

            q1 = self.env.run_robust_query_generation(self.env.instance.bias)
            if q1 is None:
                q2 = self.env.run_robust_query_generation(self.Br)
                if q2 is None:
                    continue

                if self.env.ask_membership_query(q2):
                    self.increase_stopping_threshold()
                else:
                    scope = self.env.run_find_scope(q2)
                    c = self.env.run_findc(scope)
                    if c:
                        self.env.add_to_cl(c)

            else:
                if self.env.ask_membership_query(q1):
                    # remove from B and add to Br
                    kappa = get_kappa(self.env.instance.bias, q1)
                    self.env.remove_from_bias(kappa)
                    self.Br.extend(kappa)
                else:
                    scope = self.env.run_find_scope(q1)
                    c = self.env.run_findc(scope)
                    if c:
                        self.env.add_to_cl(c)