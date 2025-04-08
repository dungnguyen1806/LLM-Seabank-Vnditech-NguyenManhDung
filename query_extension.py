import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.oauth2 import service_account
from utils.jsonProcess import load_conversation_history, save_conversation_history

import os
from dotenv import load_dotenv
import json

load_dotenv()

DEFAULT_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_PATH")
credentials = service_account.Credentials.from_service_account_file(DEFAULT_CREDENTIALS_PATH)
DEFAULT_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
DEFAULT_LOCATION = os.getenv("VERTEX_AI_LOCATION")

vertexai.init(project=DEFAULT_PROJECT, location=DEFAULT_LOCATION, credentials=credentials)   

DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME")
DEFAULT_MODEL_CONFIG = {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 500,
} 

