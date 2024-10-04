import azure.functions as func
import pandas as pd
import logging as log
import torch, os, json
from ai_files.AITrainingClass import TrainAIModel
from utils.embeddingFunctions import embdeddingFunc
from utils.AzureStorage import deleteAndSaveExcelDataToAzure
from azure.storage.blob import BlobServiceClient

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="hello")
def hello(req: func.HttpRequest) -> func.HttpResponse:
    log.info('Python HTTP trigger function processed a request.')

    return func.HttpResponse(
            "Http azure function is working",
            status_code=200
    )

@app.route(route="train_AIModel")
def train(req: func.HttpRequest) -> func.HttpResponse:
    log.info("Training function")

    try:
        res = req.get_json()

        path = str(res["path"]).replace("\\\\", "\\")

        normalizedPath = os.path.normpath(path)

        head, tail = os.path.split(normalizedPath)

        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
                
        container = blob_service_client.get_container_client("excelfiles")

        blobs = container.list_blob_names()

        for blob in blobs:
            h, t = os.path.split(os.path.normpath(blob))

            if (h == head):
                blob_client = container.get_blob_client(blob)
                blob_data = blob_client.download_blob().readall()

                # Decode binary data to string
                blob_str = blob_data.decode('utf-8')
                
                # Convert string to Python list
                blob_list = json.loads(blob_str)

                df = pd.DataFrame(blob_list)

                columns = df.columns

                selectedColumnIndex = int(res["columnNum"]) - 1

                encoded_df = embdeddingFunc(df, embedder=res["embedder"], columns=columns, selectedColumnIndex=selectedColumnIndex)

                newPath = os.path.join(h, "training_data.csv")

                log.info("change func 2")

                isBlobExist = container.get_blob_client(newPath).exists()

                deleteAndSaveExcelDataToAzure(blob_service_client, newPath, encoded_df, isBlobExist)

                container.delete_blob(blob, delete_snapshots="include")

        return func.HttpResponse(json.dumps(res), status_code=200)

    except Exception as e:
        log.info("json error: " + str(e))
        return func.HttpResponse("json error: " + str(e), status_code=200)

async def trainingFunc(df, current_user, embedder, path, label, db):
    try:
        # selectedColumnIndex = int(columnNum) - 1
        selectedColumnName = df.columns[0]
       
        AIModel = TrainAIModel(selectedColumnName, df, encodedClasses=None, mode="train")

        await AIModel.train_model(current_user.email)

        return {"training": "complete"}

    except Exception as e:
        log.info(e)
        return {"training": "failed"}


async def saving_model_and_data_in_tables(AIModel: TrainAIModel, filePath: str, label: str, embedder:str, current_user, db):
    # saving models data
    AIModel_data_filePath = os.path.join(filePath, 'model_data.pt')
    torch.save(AIModel.model.state_dict(), AIModel_data_filePath)

    return {
        "model_data": AIModel.model.state_dict(),
        "model_optimizer_data": AIModel.optimizer.state_dict(),
        "data": {
            "encodedClasses": {int(key): colName for key, colName in AIModel.classesWithEncoding.items()},
            "embedder": embedder,
            "targetColumn": AIModel.targetColumnName,
            "AIMODELTYPE": AIModel.modelType
        }
    }