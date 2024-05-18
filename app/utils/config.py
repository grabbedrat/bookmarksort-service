import logging
import os
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up the OpenAI API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Replace with your actual API key