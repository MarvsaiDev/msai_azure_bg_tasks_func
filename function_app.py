import azure.functions as func
import pandas as pd
import logging as log
import torch, os, json
from ai_files.AITrainingClass import TrainAIModel
from utils.embeddingFunctions import embdeddingFunc
from utils.AzureStorage import deleteAndSaveExcelDataToAzure
from azure.storage.blob import BlobServiceClient
from utils.RabbitMQ import publishMsgOnRabbitMQ

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="hello")
def hello(req: func.HttpRequest) -> func.HttpResponse:
    log.info('Python HTTP trigger function processed a request.')

    return func.HttpResponse(
            "Http azure function is working",
            status_code=200
    )

@app.route(route="train_AIModel")
async def train(req: func.HttpRequest) -> func.HttpResponse:
    log.info("Training function")

    try:
        res = req.get_json()

        path = str(res["path"]).replace("\\\\", "\\")

        normalizedPath = os.path.normpath(path)

        head, tail = os.path.split(normalizedPath)

        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))

        ####################################################################
        # creating container if not exists
        await publishMsgOnRabbitMQ({"container": "creating"}, res["email"])

        containerName, temp = os.path.split(head)

        containerName = extract_lowercase_and_numbers(containerName.lower()) + extract_lowercase_and_numbers(str(res["email"]))

        get_or_create_container(blob_service_client, containerName)

        await publishMsgOnRabbitMQ({"container": "created"}, res["email"])

        ####################################################################

        container = blob_service_client.get_container_client(containerName)

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

                await publishMsgOnRabbitMQ({"embedding on blob": str(blob)}, res["email"])


                encoded_df = embdeddingFunc(df, embedder=res["embedder"], columns=columns, selectedColumnIndex=selectedColumnIndex)

                await publishMsgOnRabbitMQ({"embedding done on": str(blob)}, res["email"])


                newPath = os.path.join(h, "training_data.csv")

                log.info("change func 2")

                isBlobExist = container.get_blob_client(newPath).exists()

                await publishMsgOnRabbitMQ({"saving embedded blob: ": str(blob)}, res["email"])

                deleteAndSaveExcelDataToAzure(blob_service_client, newPath, encoded_df, isBlobExist, containerName)

                await publishMsgOnRabbitMQ({"deleted blob: ": str(blob)}, res["email"])

                container.delete_blob(blob, delete_snapshots="include")

        await publishMsgOnRabbitMQ({"final": json.dumps(res)}, res["email"])
        return func.HttpResponse(json.dumps(res), status_code=200)

    except Exception as e:
        log.info("json error: " + str(e))
        await publishMsgOnRabbitMQ({"error": str(e)}, res["email"])
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

def extract_lowercase_and_numbers(input_string):
    result = ''.join(char for char in input_string if char.islower() or char.isdigit())
    return result

def get_or_create_container(blob_service_client, container_name):
    try:
        LimitedContainerName = container_name[: 62]
        blob_service_client.create_container(LimitedContainerName)
    except Exception as e:
        print(f"the exception for creating container {e}")


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