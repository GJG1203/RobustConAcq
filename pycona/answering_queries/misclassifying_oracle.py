import random
from .oracle import Oracle
from ..utils import get_con_subset, check_value
from cpmpy.transformations.normalize import toplevel_list

class MisclassifyingOracle(Oracle):
    """
    The Oracle is a human user, who directly answers the given queries but can make mistakes
    """
    
    def __init__(self, constraints, misclassification_rate=0.1):
        """
        Initialize the MisclassifyingOracle instance with a given misclassification rate and the target set of constraints.
        
        :param misclassification_rate: The probability of misclassifying a query (e.g., 0.1 for 10% errors).
        :param constraints: The set of constraints C_T used for answering queries.
        """
        super().__init__()
        self.misclassification_rate = misclassification_rate
        self.constraints = constraints
        self.flipped = 0

    @property
    def constraints(self):
        """
        get the constraints of the ConstraintOracle
        :return: self._constraints
        """
        return self._constraints

    @constraints.setter
    def constraints(self, constraints):
        """
        setter for the constraints property
        """
        constraints = toplevel_list(constraints)
        self._constraints = constraints

        
    def _maybe_misclassify(self, answer, Y=[]):
        """
        With a certain probability, misclassifies the answer.

        :param answer: The original answer (True/False) from the user.
        :return: The (possibly misclassified) answer.
        """
        
        # generate random chance
        rng = random.random()
        #print("random: " + str(rng))
        
        suboracle = get_con_subset(self.constraints, Y)

        # if answer is 'Yes' or 'True', the user can make a mistake so the answer could be flipped
        if not answer and rng < self.misclassification_rate**len(suboracle):
            print("flipped answer")
            self.flipped += 1
            return not answer  # Flip the answer
        return answer
    
    def noisy_answer_membership_query(self, Y=None):
        """
        Answer a membership query, with a chance of misclassification.

        Determines whether the given assignment on Y is a solution or not using the constraints of the problem.

        :param Y: The input values to be checked for membership.
        :return: A boolean indicating a positive or negative answer.
        """

        # user can only make mistakes when answering yes
        #print('noisy MQ')
        user_answer = self.answer_membership_query(Y)
        if not user_answer:
            return self._maybe_misclassify(user_answer, Y)
        else:
            return user_answer
        
    def answer_membership_query(self, Y=None):
        """
        Answer a membership query, with a chance of misclassification.

        Determines whether the given assignment on Y is a solution or not using the constraints of the problem.

        :param Y: The input values to be checked for membership.
        :return: A boolean indicating a positive or negative answer.
        """
        # Need the oracle to answer based only on the constraints with a scope that is a subset of Y
        suboracle = get_con_subset(self.constraints, Y)

        # Check if at least one constraint is violated or not
        return all([check_value(c) for c in suboracle])

    def answer_recommendation_query(self, c=None):
        """
        Answer a recommendation queryby checking if the recommended constraint is part of the constraints, 
        with a probability of misclassification.

        :param c: The recommended constraint to be checked.
        :return: A boolean indicating if the recommended constraint is part of the constraints.
        """
        # Check if the recommended constraint is in the set of constraints
        user_answer = c in self.constraints
        return self._maybe_misclassify(user_answer)

    def answer_generalization_query(self, C=None):
        """
        Answer a generalization query, with a probability of misclassification.

        :param C: The generalization of constraints to be checked.
        :return: A boolean indicating if the generalization of constraints is correct.
        """
        user_answer = all(constraint in set(self.constraints) for constraint in C)
        return self._maybe_misclassify(user_answer)