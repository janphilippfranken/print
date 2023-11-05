import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import os

from experiments.apps.data import get_apps_dataset
 
from helpers import extract_code

# llm class
from printlm.chat_models.azure import AzureChatLLM
from printlm.agents.code import CodeAgent
from printlm.agents.feedback import FeedbackAgent
from printlm.agents.repair import RepairAgent

# prompts 
from printlm.prompts.code.prompts import CODE_PROMPTS
from printlm.prompts.repair.prompts import REPAIR_PROMPTS
from printlm.prompts.feedback.prompts import FEEDBACK_PROMPTS
from printlm.prompts.task.prompts import TASK_PROMPTS

# get llm api class
def get_llms(
    args: DictConfig,         
    is_azure: bool,
) -> AzureChatLLM:
    if is_azure:
        code_llm = AzureChatLLM(**args.model.azure_api)
        feedback_llm = AzureChatLLM(**args.model.azure_api)
        repair_llm = AzureChatLLM(**args.model.azure_api)
        return code_llm, feedback_llm, repair_llm
    else:
        print(f"ERROR: {args.model.api} is not a valid API (yet)")
        
# run 
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig) -> None:
    
    # get llm(s)
    args.model.azure_api.api_key = os.getenv("OPENAI_API_KEY")
    is_azure = args.api_name
    code_llm, feedback_llm, repair_llm = get_llms(args, is_azure)

    # instantiate agents
    code_agent = CodeAgent(llm=code_llm, model_id="code", **args.model.code)
    feedback_agent = FeedbackAgent(llm=feedback_llm, model_id="feedback", **args.model.feedback)
    repair_agent = RepairAgent(llm=repair_llm, model_id="repair", **args.model.repair)

    # get dataset 
    dataset = get_apps_dataset(difficulty=args.data.difficulty)

    # run agents
    for datum in tqdm(dataset):
        if datum["problem_id"] == 4004:

            # task prompt 
            task_prompt = TASK_PROMPTS["task_prompt_1"]
            task_prompt.content = datum["question"]
            
            # code model response 
            code_model_response = code_agent.run(
                code_prompt=CODE_PROMPTS["code_prompt_1"],
                task_prompt=task_prompt,
            )
    
            # feedback model response
            feedback_model_response = feedback_agent.run(
                feedback_prompt=FEEDBACK_PROMPTS["feedback_prompt_1"],
                task_prompt=task_prompt,
                initial_code=code_model_response["response_str"],
            )
            
            # repair model response
            repair_model_response = repair_agent.run(
                repair_prompt=REPAIR_PROMPTS["repair_prompt_1"],
                task_prompt=task_prompt,
                initial_code=code_model_response["response_str"],
                feedback=feedback_model_response["response_str"],
            )


        
if __name__ == '__main__':
    main()