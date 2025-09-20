from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
	query_model: str = "bedrock/us.amazon.nova-pro-v1:0"
	embedding_model: str = "bedrock/amazon.titan-embed-text-v2:0"
	# embedding_model: str = "google/embeddinggemma-300m"
	documents_directory: str = "./documents"
	data_directory: str = "./datasets"
