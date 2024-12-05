import random
from .oracle import Oracle

class MisclassifyingUserOracle(Oracle):
    """
    The Oracle is a human user, who directly answers the given queries but can make mistakes
    """
    
    def __init__(self, misclassification_rate=0.1):
        """
        Initialize the UserOracle instance with a given misclassification rate.
        
        :param misclassification_rate: The probability of misclassifying a query (e.g., 0.1 for 10% errors).
        """
        super().__init__()
        self.misclassification_rate = misclassification_rate
        
    def _get_user_response(self):
        """
        Prompts the user for a yes/no response and returns True for 'yes'/'y' and False for 'no'/'n'.
        """
        response = input().strip().lower()
        while response not in ['yes', 'y', 'no', 'n']:
            response = input("Please answer with yes, y, no, or n: ").strip().lower()
        return response == 'yes' or response == 'y'
        
    def _maybe_misclassify(self, answer):
        """
        With a certain probability, misclassifies the answer.

        :param answer: The original answer (True/False) from the user.
        :return: The (possibly misclassified) answer.
        """
        
        # generate random chance
        rng = random.random()
        print("random: " + str(rng))
        
        # if answer is 'Yes' or 'True', the user can make a mistake so the answer could be flipped
        if answer and rng < self.misclassification_rate:
            print("flipped answer")
            return not answer  # Flip the answer
        return answer
        
    def answer_membership_query(self, Y=None):
        """
        Answer a membership query, with a probability of misclassification.

        :param Y: The values to be checked against the constraints.
        :return: A boolean indicating if the values satisfy the constraints.
        """
        print("answer MQ")
        user_answer = self._get_user_response()
        return self._maybe_misclassify(user_answer)

    def answer_recommendation_query(self, c=None):
        """
        Answer a recommendation query, with a probability of misclassification.

        :param c: The recommended constraint to be checked.
        :return: A boolean indicating if the recommended constraint is part of the constraints.
        """
        print("answer RQ")
        user_answer = self._get_user_response()
        return self._maybe_misclassify(user_answer)

    def answer_generalization_query(self, C=None):
        """
        Answer a generalization query, with a probability of misclassification.

        :param C: The generalization of constraints to be checked.
        :return: A boolean indicating if the generalization of constraints is correct.
        """
        print("answer GQ")
        user_answer = self._get_user_response()
        return self._maybe_misclassify(user_answer)