from abc import ABC, abstractmethod
import os
from transformers import AutoTokenizer, AutoModel
import torch
import openai
from openai import RateLimitError
import time
from transformers import AutoTokenizer, AutoModel

class AbstractEmbeddingModel(ABC):
    filename = "modelBGE.pth"
    output_size = 1024
    input_multiple = 4
    @abstractmethod
    def get_embedding(self, text):
        pass

class BAIEmbeddingModel(AbstractEmbeddingModel):
    filename = "modelBGE.pth"
    output_size = 1024
    input_multiple = 4


    def __init__(self, model_name='BGE',classifier_module='custom_ai.ai', classifier_class='CustomClassifier', local_files_only=False):
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        self.model = AutoModel.from_pretrained("BAAI/bge-m3")

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        # Get the embeddings
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state
        # Get the vector for the [CLS] token (first token)
        vector = embeddings[0, 0, :].numpy()
        return vector
    def get_embedding_torch(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state
        embeddings = torch.mean(embeddings, dim=1)
        return embeddings.detach().numpy().tolist()


class OpenAIEmbeddingModel(AbstractEmbeddingModel):
    filename = "model2x.pth"
    output_size = 1536
    input_multiple = 2

    API_KEY = os.getenv("AZURE_HEALTHSCANNER_UK_EMBEDDING_API_KEY")
    SMALL_API_KEY = os.getenv("AZURE_HEALTHSCANNER_UK_SMALL_EMBEDDING_API_KEY")


    client = openai.AzureOpenAI(
        azure_endpoint="https://healthscanneruk.openai.azure.com",
        api_version="2023-05-15",
        api_key=API_KEY,
    )

    client2 = openai.AzureOpenAI(
        azure_endpoint="https://rashi-m24ednq1-japaneast.cognitiveservices.azure.com",
        api_version="2023-05-15",
        api_key=SMALL_API_KEY,
    )
    
    def get_embedding(self, text, retry = 1, model='text-embedding-large'):
        try:
            if (model == 'text-embedding-large'):
                response = self.client.embeddings.create(input=text, model=model)
            else:
                response = self.client2.embeddings.create(input=text, model=model)

            embedding_vectors = []

            for data in response.data:
                embedding_vectors.append(data.embedding)

            return embedding_vectors
        except RateLimitError:
            if (retry < 3):
                time.sleep(5)
                return self.get_embedding(text, retry=retry + 1, model=model)