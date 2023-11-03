from langchain.chat_models import AzureChatOpenAI

BASE_URL = "https://philipp.openai.azure.com/"
API_KEY = ""
DEPLOYMENT_NAME = "gpt-4"

llm = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version="2023-05-15",
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type="azure",
)