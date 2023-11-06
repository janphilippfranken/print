
from print.utility.base import BaseUtility

"Utility Function As Code"
def meta_utiliy(feedback_str: str):
    """
    Evaluates the algorithm in feedback_str to provide informative feedback for the improver for improving the algorithm in algorithm_str, 
    according to some downstream utility function. This meta-utility function can only be called n times.
    """
    n_tests = 3
    expected_utility = 0
    for _ in range(n_tests):
        improved_algorithm_str = improve_algorithm(algorithm_str, task_utility, feedback_str, language_model)
        expected_utility += task_utility.func(improved_algorithm_str) / n_tests
    return expected_utility, improved_algorithm_str

"Utility Function As String"
meta_utility_str = """def meta_utiliy(feedback_str: str):
    ""
    Evaluates the algorithm in feedback_str to provide informative feedback for the improver for improving the algorithm in algorithm_str, 
    according to some downstream utility function. This meta-utility function can only be called n times.
    ""
    n_tests = 3
    expected_utility = 0
    for _ in range(n_tests):
        try: 
            exec(feedback_str, globals())
        except:
            continue
        feedback_algorithm_str = feedback_algorithm(algorithm_str, task_utility, language_model)
        improved_algorithm_str = improve_algorithm(algorithm_str, task_utility, feedback_algorithm_str, language_model)
        expected_utility += task_utility.func(improved_algorithm_str) / n_tests
    return expected_utility"""

meta_utility_class = Utility(ustr=meta_utility_str, ufunc=meta_utiliy)