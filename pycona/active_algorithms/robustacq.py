import time

from ..find_scope.findscope import FindScope
from .algorithm_core import AlgorithmCAInteractive
from ..problem_instance import ProblemInstance
from ..answering_queries import Oracle, UserOracle, MisclassifyingOracle
from ..ca_environment.active_ca import ActiveCAEnv
from ..ca_environment.active_ca_proba import ProbaActiveCAEnv
from ..utils import get_kappa
from .. import Metrics


class RobustAcq(AlgorithmCAInteractive):
    """
    RobustAcq is a modified version of the QuAcq algorithm with additional robustness features.
    """

    def __init__(self, ca_env: ProbaActiveCAEnv = None, stop_thresh=5, retrain_thresh=20, confidence_thresh=0.8):
        """
        Initialize the RobustAcq algorithm with optional thresholds.
        :param ca_env: An instance of CASystem, default is None.
        :param threshold1: Stopping threshold for convergence.
        :param threshold2: Size threshold for retraining classifier.
        """
        super().__init__(ca_env if ca_env is not None else ProbaActiveCAEnv(find_scope=FindScope()))
        self.stop_thresh = stop_thresh
        self.retrain_thresh = retrain_thresh
        self.stopping_threshold = 0
        self.confidence_thresh = confidence_thresh

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
        for constraint in self.env.Br:
            # Get the feature representation for the constraint
            features = self.env.feature_representation.featurize_constraint(constraint)
            # Get classifier prediction probability
            prob = self.env.classifier.predict_proba([features])[0][0]  # Assuming class 0 is 'doesn't belong in Br'
            
            # Print debug info only for constraints in oracle
            if constraint in set(self.env.oracle.constraints):
                print(constraint)
                print(prob)
                
            # If classifier is confident it's misclassified, mark it for moving
            if prob < (1 - self.confidence_thresh):
                constraints_to_move.add(constraint)
        
        # Move the marked constraints from Br back to B
        self.env.Br = self.remove_constraints_Br(constraints_to_move)
        print("removed " + str(len(constraints_to_move)) + " constraints from Br")
        self.env.instance.bias.extend(constraints_to_move)  # Add back to B
        print("added " + str(len(constraints_to_move)) + " constraints to bias")
        
    def remove_constraints_Br(self, cons):
        return (self.env.Br - cons)
        
    def increase_stopping_threshold(self):
        """Increase the stopping threshold"""
        self.stopping_threshold += 1
        #print("increase stopthresh")
        

    def learn(self, instance: ProblemInstance, oracle: Oracle = UserOracle(), verbose=0, metrics: Metrics = None):

        self.env.init_state(instance, oracle, verbose, metrics)
        
        if len(self.env.instance.bias) == 0:
            self.env.instance.construct_bias()
            self.env._bias_proba = {c: 0.01 for c in self.env.instance.bias}

        while True:
            if self.env.verbose > 2:
                print("Size of CL: ", len(self.env.instance.cl))
                print("Size of B: ", len(self.env.instance.bias))
                print("Size of Br: ", len(self.env.Br))
                print("Size of dataset: ", len(self.env.datasetX))
                print("Positive instances: ", sum(self.env.datasetY))
                print("Negative instances: ", len(self.env.datasetY) - sum(self.env.datasetY))
                print("flipped: ", self.env.oracle.flipped)
                
                print("Number of Queries: ", self.env.metrics.membership_queries_count)
        
            if self.stopping_threshold > self.stop_thresh:
                return self.env.instance # Convergence

            if len(self.env.instance.cl) > self.retrain_thresh:
                #print("retrain classifier")
                self.retrain_classifier()

            q1 = self.env.run_robust_query_generation(self.env.instance.bias)
            if len(q1) == 0:
                q2 = self.env.run_robust_query_generation(self.env.Br)
                # if len(q2) == 0:
                #     print("len q2 is 0")
                #     continue

                # user can only make mistakes here, not in findC or findScope
                if self.env.noisy_ask_membership_query(q2):
                    self.increase_stopping_threshold()
                else:
                    scope = self.env.run_find_scope(q2)
                    c = self.env.run_findc(scope)
                    if c:
                        self.env.add_to_cl(c)

            else:
                # user can only make mistakes here, not in findC or findScope
                if self.env.noisy_ask_membership_query(q1):
                    kappa = get_kappa(self.env.instance.bias, q1)
                    # remove from B and add to Br
                    self.env.remove_from_bias(kappa)
                else:
                    scope = self.env.run_find_scope(q1)
                    c = self.env.run_findc(scope)
                    if c:
                        self.env.add_to_cl(c)