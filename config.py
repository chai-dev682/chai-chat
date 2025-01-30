from enum import Enum
from os.path import join
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = "./"
PROMPT_ROOT = join(PROJECT_ROOT, "prompt_templates")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# pinecone
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

def load_env():
    load_dotenv(join(PROJECT_ROOT, ".env"))

class ModelType(str, Enum):
    embedding = "text-embedding-3-large"

class PromptTemplate(Enum):
    UPWORK_PROFILE = "upwork_profile.txt"
    UPWORK_SCREENING_QUESTIONS = "upwork_screening_questions.txt"
    JOB_SCREENING_QUESTIONS = "job_screening_questions.txt"
    GENERATE = "generate.txt"
    JOB_COVER_LETTER = "job_cover_letter.txt"
    PROPOSAL = "proposal.txt"
    CONVERSATION_RESPONSE = "conversation_response.txt"
    SAVED_REPLY = "saved_reply.txt"
    
def get_prompt_template(prompt_template: PromptTemplate):
    with open(join(PROMPT_ROOT, prompt_template.value), "rt") as f:
        return f.read()
