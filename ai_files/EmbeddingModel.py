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
        # if not local_files_only:
        #     self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        #     self.model = AutoModel.from_pretrained("BAAI/bge-m3")
        #     # self.tokenizer.save_pretrained("./model_data/bgetoken")
        #     # self.model.save_pretrained("./model_data/bgemodel")
        # else:
        #     self.tokenizer = AutoTokenizer.from_pretrained("./model_data/bgetoken", local_files_only=local_files_only)
        #     self.model = AutoModel.from_pretrained("./model_data/bgemodel", local_files_only=local_files_only)


    def get_embedding(self, text):

        # Initialize the tokenizer and the model

        # Tokenize the text and convert to input IDs
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

    # def download_model_data(self):
    #     container_name = "transformermodeldata"
    #     folders = ["bgemodel", "bgetoken"]

    #     for folder in folders:
    #         blob_list = blob_service_client.get_container_client(container_name).list_blobs(name_starts_with=folder)
    #         for blob in blob_list:
    #             blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
    #             download_file_path = os.path.join("./model_data", blob.name)
    #             os.makedirs(os.path.dirname(download_file_path), exist_ok=True)
    #             with open(download_file_path, "wb") as download_file:
    #                 download_file.write(blob_client.download_blob().readall())



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

    # def __init__(self, classifier_module='custom_ai.ai', classifier_class='CustomClassifier'):
        # module = importlib.import_module(classifier_module)
        # self.classifier = getattr(module, classifier_class)
        # openai.api_key = os.getenv("AZURE_HEALTHSCANNER_UK_API_KEY")
        # self.model = openai.Completion.create(engine=model_name)
    
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