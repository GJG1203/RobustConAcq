import time

import networkx as nx
from cpmpy.transformations.get_variables import get_variables

from .algorithm_core import AlgorithmCAInteractive
from .utils import is_clique, can_be_clique
from ..ca_environment.active_ca import ActiveCAEnv
from ..utils import get_relation, get_scope, get_kappa
from ..problem_instance import ProblemInstance
from ..answering_queries import Oracle, UserOracle
from .. import Metrics


class GQuAcq(AlgorithmCAInteractive):

    """
    QuAcq variation algorithm, using mine&Ask to detect types of variables and ask genralization queries. From:
    "Detecting Types of Variables for Generalization in Constraint Acquisition", ICTAI 2015.
    """

    def __init__(self, ca_env: ActiveCAEnv = None, qg_max=10):
        """
        Initialize the PQuAcq algorithm with an optional constraint acquisition system.

        :param ca_env: An instance of CASystem, default is None.
        : param GQmax: maximum number of generalization queries
        """
        super().__init__(ca_env)
        self._negativeQ = []
        self._qg_max = qg_max

    def learn(self, instance: ProblemInstance, oracle: Oracle = UserOracle(), verbose=0, metrics: Metrics = None):
        """
        Learn constraints using the QuAcq algorithm by generating queries and analyzing the results.

        :param instance: the problem instance to acquire the constraints for
        :param oracle: An instance of Oracle, default is to use the user as the oracle.
        :param verbose: Verbosity level, default is 0.
        :param metrics: statistics logger during learning
        :return: the learned instance
        """
        self.env.init_state(instance, oracle, verbose, metrics)

        if len(self.env.instance.bias) == 0:
            self.env.instance.construct_bias()

        while True:
            if self.env.verbose > 0:
                print("Size of CL: ", len(self.env.instance.cl))
                print("Size of B: ", len(self.env.instance.bias))
                print("Number of Queries: ", self.env.metrics.membership_queries_count)

            gen_start = time.time()
            Y = self.env.run_query_generation()
            gen_end = time.time()

            if len(Y) == 0:
                # if no query can be generated it means we have (prematurely) converged to the target network -----
                self.env.metrics.finalize_statistics()
                if self.env.verbose >= 1:
                    print(f"\nLearned {self.env.metrics.cl} constraints in "
                          f"{self.env.metrics.membership_queries_count} queries.")
                return self.env.instance

            self.env.metrics.increase_generation_time(gen_end - gen_start)
            self.env.metrics.increase_generated_queries()
            self.env.metrics.increase_top_queries()
            kappaB = get_kappa(self.env.instance.bias, Y)

            answer = self.env.ask_membership_query(Y)
            if answer:
                # it is a solution, so all candidates violated must go
                # B <- B \setminus K_B(e)
                self.env.remove_from_bias(kappaB)

            else:  # user says UNSAT

                scope = self.env.run_find_scope(Y)
                c = self.env.run_findc(scope)
                self.env.add_to_cl(c)
                self.mineAsk(get_relation(c, self.env.instance.language))


    def mineAsk(self, r):
        """
        Mine&Ask function presented in
        "Detecting Types of Variables for Generalization in Constraint Acquisition", ICTAI 2015.

        :param r: The index of a relation in gamma.
        :return: List of learned constraints.
        """
        gq_counter = 0

        C = [c for c in self.env.instance.cl if get_relation(c, self.env.instance.language) == r]

        # Project Y to those in C that have relation r
        Y = [v.name for v in get_variables(C)]
        E = [tuple([v.name for v in get_scope(c)]) for c in C]  # all scopes

        # Create the graph
        G = nx.Graph()
        G.add_nodes_from(Y)
        G.add_edges_from(E)

        T = [comp for comp in nx.components.connected_components(G) if not is_clique(G.subgraph(comp))]

        while len(T) > 0 and gq_counter < self._qg_max:
            Y = T.pop()
            gen_flag = False
            B = [c for c in self.env.instance.bias if
                 get_relation(c, self.env.instance.language) == r and frozenset(get_scope(c)).issubset(Y)]
            D = [tuple([v.name for v in get_scope(c)]) for c in B]  # missing edges that can be completed (exist in B)

            # if already a subset of it was negative, or cannot be completed to a clique, continue to next
            if not any(Y2.issubset(Y) for Y2 in self._negativeQ) and can_be_clique(G.subgraph(Y), D):
                # if potentially generalizing leads to unsat, continue to next
                new_CL = self.env.instance.cl.copy()
                new_CL += B
                if new_CL.solve() and self.env.ask_generalization_query(r, B):
                    gen_flag = True
                    self.env.add_to_cl(B)
                else:
                    gq_counter += 1
                    self._negativeQ.append(Y)

            if not gen_flag:
                communities = nx.community.greedy_modularity_communities(G.subgraph(Y))
                [T.append(com) for com in communities if 2 < len(com) < len(Y)]