from abc import ABC, abstractmethod
import os
from transformers import AutoTokenizer, AutoModel
import torch
import openai
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
        if not local_files_only:
            self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
            self.model = AutoModel.from_pretrained("BAAI/bge-m3")
            self.tokenizer.save_pretrained("./model_data/bgetoken")
            self.model.save_pretrained("./model_data/bgemodel")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("./model_data/bgetoken", local_files_only=local_files_only)
            self.model = AutoModel.from_pretrained("./model_data/bgemodel", local_files_only=local_files_only)


        # module = importlib.import_module(classifier_module)
        # self.classifier = getattr(module, classifier_class)


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
        return embeddings.detach().numpy()

class OpenAIEmbeddingModel(AbstractEmbeddingModel):
    filename = "model2x.pth"
    output_size = 1536
    input_multiple = 2

    API_KEY = os.getenv("AZURE_HEALTHSCANNER_UK_EMBEDDING_API_KEY")


    client = openai.AzureOpenAI(
        azure_endpoint="https://healthscanneruk.openai.azure.com",
        api_version="2023-05-15",
        api_key=API_KEY,

    )

    def __init__(self, model_name='text-embedding-large',classifier_module='custom_ai.ai', classifier_class='CustomClassifier'):
        # module = importlib.import_module(classifier_module)
        # self.classifier = getattr(module, classifier_class)
        # openai.api_key = os.getenv("AZURE_HEALTHSCANNER_UK_API_KEY")
        # self.model = openai.Completion.create(engine=model_name)
        self.model = model_name
    
    def get_embedding(self, text):
        response = self.client.embeddings.create(
            input=text, model=self.model
        )

        embedding_vectors = []

        for data in response.data:
            embedding_vectors.append(data.embedding)

        return embedding_vectors