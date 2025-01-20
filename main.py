from pycona.answering_queries import MisclassifyingOracle
from pycona import *
from pycona.benchmarks import construct_sudoku

instance, oracle = construct_sudoku(2,2,4)

oracle = MisclassifyingOracle(oracle.constraints)
ca = RobustAcq()

learned_instance = ca.learn(instance, oracle=oracle, verbose=3)


print("\n\ncl: ", len(learned_instance.cl))
print("oracle: ", len(set(oracle.constraints)))
print("missing: ", len(list(set(oracle.constraints) - set(learned_instance.cl))))
