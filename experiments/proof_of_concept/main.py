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
from helpers import extract_code, save_data, update_dict

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
    improve_language_model = LLMAgent(llm=llm, **args.model.llm_improve)
    print_improve_language_model= LLMAgent(llm=llm, **args.model.llm_print)

    # get task and initial utility
    task = TASKS[args.task.task_id]
    task.utility.func, task.utility.str = task.utility.get_utility()
    initial_solution = task.get_solution()
    initial_utility = task.utility.func(initial_solution)[0]

    # store solutions and performances
    # improver
    improver_data = {
        'model': ['improver'] * (args.sim.n_runs + 1),
        'improvements': np.arange(args.sim.n_runs + 1),
        'cost': [0], 
        'solutions': [initial_solution], 
        'utility': [task.utility.func(initial_solution)[0]],
        'average_utility': [initial_utility],
    }
    
    # print improver
    print_improver_data = {
        'model': ['print_improver'] * (args.sim.n_runs + 1),
        'improvements': np.arange(args.sim.n_runs + 1),
        'cost': [0], 
        'solutions': [initial_solution], 
        'utility': [task.utility.func(initial_solution)[0]],
        'modified_solutions': [initial_solution],
        'print_returns': ['None'],
        'average_utility': [initial_utility],
    }

    # main inner loop using the scaffold program to find improved solution to the task and evaluating these solutions
    for sim in tqdm(range(args.sim.n_runs)):

        if args.sim.verbose:
            breakpoint()
        else:
            # generate improved solution using improver 
            improved_solution, average_improved_utility = improve_algorithm(improver_data['solutions'][-1], 
                                                  task.utility, 
                                                  improve_language_model)
            # append results 
            if task.utility.func(improved_solution)[0] > improver_data['utility'][-1]:
                improver_data['cost'].append(improve_language_model.total_inference_cost)
                improver_data['solutions'].append(improved_solution)
                improver_data['utility'].append(task.utility.func(improved_solution)[0])
                improver_data['average_utility'].append(average_improved_utility)
            else:
                improver_data['cost'].append(improve_language_model.total_inference_cost)
                improver_data['solutions'].append(improver_data['solutions'][-1])
                improver_data['utility'].append(improver_data['utility'][-1])
                improver_data['average_utility'].append(improver_data['average_utility'][-1])

            # generate improved solution using print improver
            modified_solutions = insert_prints(print_improver_data['solutions'][-1], print_improve_language_model)
            print_returns = generate_print_returns(modified_solutions, task.utility) # evaluate modified code to get print returns
            breakpoint()
            print_improved_solution, modified_solution_idx, average_print_improved_utility, = print_improve_algorithm(print_improver_data['solutions'][-1], 
                                                                                     modified_solutions,
                                                                                     print_returns, 
                                                                                     task.utility, 
                                                                                     print_improve_language_model) # llm call 2: improve solution using print returns
            # append results
            if task.utility.func(print_improved_solution)[0] > print_improver_data['utility'][-1]:
                print_improver_data['cost'].append(print_improve_language_model.total_inference_cost)
                print_improver_data['solutions'].append(print_improved_solution)
                print_improver_data['utility'].append(task.utility.func(print_improved_solution)[0])
                print_improver_data['modified_solutions'].append(modified_solutions[modified_solution_idx])
                print_improver_data['print_returns'].append(print_returns[modified_solution_idx])
                print_improver_data['average_utility'].append(average_print_improved_utility)
            else:
                print_improver_data['cost'].append(print_improve_language_model.total_inference_cost)
                print_improver_data['solutions'].append(print_improver_data['solutions'][-1])
                print_improver_data['utility'].append(print_improver_data['utility'][-1])
                print_improver_data['modified_solutions'].append(print_improver_data['modified_solutions'][-1])
                print_improver_data['print_returns'].append(print_improver_data['print_returns'][-1])
                print_improver_data['average_utility'].append(print_improver_data['average_utility'][-1])

    # save and plot data 
    save_data(improver_data=improver_data, 
              print_improver_data=print_improver_data,
              data_dir=DATA_DIR, 
              sim_id=args.sim.sim_id)

if __name__ == '__main__':
    main()