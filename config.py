from enum import Enum
from os.path import join
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = "./"
PROMPT_ROOT = join(PROJECT_ROOT, "prompt_templates")

def load_env():
    load_dotenv(join(PROJECT_ROOT, ".env"))


class PromptTemplate(Enum):
    UPWORK_PROFILE = "upwork_profile.txt"
    SCREENING_QUESTIONS = "screening_questions.txt"
    GENERATE = "generate.txt"
    JOB_COVER_LETTER = "job_cover_letter.txt"
    PROPOSAL = "proposal.txt"
    CONVERSATION_RESPONSE = "conversation_response.txt"
    
def get_prompt_template(prompt_template: PromptTemplate):
    with open(join(PROMPT_ROOT, prompt_template.value), "rt") as f:
        return f.read()
