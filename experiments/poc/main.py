import os
import numpy as np
import pandas as pd 

import hydra
from omegaconf import DictConfig

from tqdm import tqdm

# taks
from print.task.tasks import TASKS

# llm class
from print.language_agents.llm import LLMAgent
from print.chat_models.azure import AsyncAzureChatLLM

# helpers and plots
from helpers import extract_code, save_data

# improve algorithms
from improver import improve_algorithm
from print_improver import insert_prints, generate_print_returns, print_improve_algorithm


# get llm api class
def get_llm(
    args: DictConfig,         
    is_azure: bool,
) -> AsyncAzureChatLLM:
    if is_azure:
        llm = AsyncAzureChatLLM(**args.model.azure_api)
        return llm
    else:
        print(f"ERROR: {args.model.api} is not a valid API (yet)")

# run 
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig) -> None:

    # data directory
    DATA_DIR = f'{hydra.utils.get_original_cwd()}/runs/{args.sim.sim_dir}'
    
    # get llm api and language model
    args.model.azure_api.api_key = os.getenv("OPENAI_API_KEY")
    is_azure = args.model.api_type
    llm = get_llm(args, is_azure)
    improve_language_model = LLMAgent(llm=llm, **args.model.llm)
    print_improve_language_model= LLMAgent(llm=llm, **args.model.llm)

    # get task and initial utility
    task = TASKS[args.task.task_id]
    task.utility.func, task.utility.str = task.utility.get_utility()
    initial_solution = task.get_solution()
    initial_utility = task.utility.func(initial_solution)[0]

    # store solutions and performances
    # improver
    improver_data = {
        'model': ['improver'] * (args.sim.n_runs + 1),
        'improvements': [0],
        'cost': [0], 
        'solutions': [initial_solution], 
        'utility': [initial_utility],
        'utility_error': [0],
        'cost_error': [0],
        'modified_solutions': [None] * (args.sim.n_runs + 1),
    }
    
    # print improver
    print_improver_data = {
        'model': ['print_improver'] * (args.sim.n_runs + 1),
        'improvements': [0],
        'cost': [0], 
        'modified_solutions': [initial_solution],
        'solutions': [initial_solution], 
        'utility': [initial_utility],
        'utility_error': [0],
        'cost_error': [0],
        'modified_solutions': [None] * (args.sim.n_runs + 1),
    }

    # main inner loop using the scaffold program to find improved solution to the task and evaluating these solutions
    for sim in tqdm(range(args.sim.n_runs)):

        if args.sim.verbose:
            improved_solution = improve_algorithm(improver_data['solutions'][-1], task.utility, improve_language_model)
            improver_data['cost'].append(improve_language_model.total_inference_cost)
            improver_data['solutions'].append(improved_solution)
            improver_data['utility'].append(task.utility.func(improved_solution)[0])
            improver_data['improvements'].append(sim + 1)
            improver_data['utility_error'].append(0)
            improver_data['cost_error'].append(0)
            # improver_data['utility'].append(task.utility.func(improved_solution)[0])


            print_improver_data['cost'].append(np.random.uniform(0, 1))
            print_improver_data['solutions'].append(initial_solution)
            print_improver_data['utility'].append(np.random.uniform(0, 1))
            print_improver_data['improvements'].append(sim + 1)
            print_improver_data['utility_error'].append(0.1)
            print_improver_data['cost_error'].append(0.1)

        



        # else: 
        #     # improve solution using generic stop-like scaffold
        #     improver_cost.append(improve_language_model.total_inference_cost)
        #     improved_solution = improve_algorithm(improver_solutions[-1], task.utility, improve_language_model)
        #     improver_solutions.append(improved_solution)
        #     improver_utility.append(task.utility.func(improved_solution)[0])
        #     # improve solution using print improver
        #     print_improver_cost.append(print_improve_language_model.total_inference_cost)
        #     modified_solutions = insert_prints(print_improver_solutions[-1], print_improve_language_model)
        #     print_returns = generate_print_returns(modified_solutions, task.utility) # evaluate modified code to get print returns
        #     print_improved_solution, modified_solution = print_improve_algorithm(modified_solutions, print_returns, task.utility, print_improve_language_model) # llm call 2: improve solution using print returns
        #     best_modified_solutions.append(modified_solution)
        #     print_improver_solutions.append(print_improved_solution)
        #     print_improver_utility.append(task.utility.func(print_improved_solution)[0])

        # store results as csvs
        print('total cost', improve_language_model.total_inference_cost)
        # 

    # save and plot data 
    save_data(improver_data=improver_data, 
              print_improver_data=print_improver_data, 
              data_dir=DATA_DIR, 
              sim_id=args.sim.sim_id)

        

if __name__ == '__main__':
    main()