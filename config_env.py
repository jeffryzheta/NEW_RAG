import os

# Set environment variables (for testing purposes)
os.environ["AZURE_OPENAI_LLM_ENDPOINT"] = "https://openai-edasquad1.openai.azure.com/openai/deployments/gpt-4o-mini-4/chat/completions?api-version=2023-03-15-preview"
os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"] = "https://openai-edasquad1.openai.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15"
os.environ["AZURE_OPENAI_VERSION"] = "2023-03-15-preview"
os.environ["AZURE_OPENAI_KEY"] = 'd7fd42addeff4f4a91f9beea8996f4cc'

# Access environment variables
llm_endpoint = os.environ["AZURE_OPENAI_LLM_ENDPOINT"]
embedding_endpoint = os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"]
AZURE_OPENAI_VERSION = os.environ['AZURE_OPENAI_VERSION']
api_key = os.environ['AZURE_OPENAI_KEY']