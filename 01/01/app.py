from rich import print 

from agents import Agent, Runner , AsyncOpenAI, OpenAIChatCompletionsModel , RunConfig

from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

external_client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url = "https://generativelanguage.googleapis.com/v1beta",
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client = external_client,

)
config = RunConfig(
    model = model,
    model_provider = model,
    tracing_disabled = True,
)

writer = Agent(
    name = "Translator Agent",
    instructions = 
    """You are a translator agent.
    you will be given a text in Sindhi and you will translate it to English.
    You will only return the translated text without any additional information or formatting.
    """,


)
user_input = input("\nEnter the text you want to translate to English: ")

response = Runner.run_sync(
    writer,
    input = user_input,

    run_config = config,
)

print("\n[bold yellow]Translated Text in English:[bold yellow]",response.final_output)