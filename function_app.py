import azure.functions as func
import pandas as pd
import logging as log
import os, json
import asyncio
from azure.storage.blob import BlobServiceClient
from utils.RabbitMQ import publishMsgOnRabbitMQ
from utils.utilityFunctions import EmbeddingFile, extract_lowercase_and_numbers, get_or_create_container, trainingFunc

# Configure logging
handler = log.StreamHandler()
handler.setLevel(log.INFO)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
log.getLogger().addHandler(handler)
log.getLogger().setLevel(log.INFO)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="hello")
async def hello(req: func.HttpRequest) -> func.HttpResponse:
    log.info('Python HTTP trigger function processed a request.')

    return func.HttpResponse(
        "Http azure function is working",
        status_code=200
    )

@app.route(route="train_AIModel")
async def train(req: func.HttpRequest) -> func.HttpResponse:
    log.info("Training function: 41")
    try:
        res = req.get_json()

        response = func.HttpResponse("Tasks started", status_code=200)

        asyncio.create_task(trainRouteFunc(res))

        return response
        
    except Exception as e:
        log.error("error: ")
        log.error(e)
        raise e

    


async def trainRouteFunc(res):
    try:
        ###################################################################
        # path validation 
        path = str(res["path"]).replace("\\\\", "\\")

        normalizedPath = os.path.normpath(path)

        head, tail = os.path.split(normalizedPath)

        await publishMsgOnRabbitMQ({"head from which i am checking the container or directory: ": str(head)}, res["email"])

        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))

        ####################################################################
        # creating container if not exists, create container with name
        await publishMsgOnRabbitMQ({"container": "creating"}, res["email"])

        containerName, temp = os.path.split(head)

        containerName = extract_lowercase_and_numbers(containerName.lower()) + extract_lowercase_and_numbers(str(res["email"]))

        if (len(containerName) > 62):
            containerName = containerName[: 62]

        get_or_create_container(blob_service_client, containerName)

        await publishMsgOnRabbitMQ({"container": "created", "container_name": containerName}, res["email"])

        ####################################################################
        # get container client
        container = blob_service_client.get_container_client(containerName)

        # Embedding the file (which is saved in chunks) and saving file on path with training_data.csv file
        pathsOfTrainingFiles, headers = await EmbeddingFile(blob_service_client, containerName, container, head, res)

        await publishMsgOnRabbitMQ({"task": "training", "condition": "preparing"}, res["email"])

        if (len(pathsOfTrainingFiles) < 1):
            await publishMsgOnRabbitMQ({"error": "Embedding function returned empty"}, res["email"])
            return 

        ####################################################################

        df = pd.DataFrame()

        for path in pathsOfTrainingFiles:
            # getting the blob
            blob_client = container.get_blob_client(path)
            # get the blob data
            blob_data = blob_client.download_blob().readall()

            # Decode binary data to string from blob
            blob_str = blob_data.decode('utf-8')
            
            # Convert string to Python list
            blob_list = json.loads(blob_str)

            # creating and merging data frame from list
            df = pd.concat([df, pd.DataFrame(blob_list)], ignore_index=True)

            # deleting embedding file blob after getting data
            blob_client.delete_blob(delete_snapshots="include")

        await trainingFunc(df, res["email"], container, pathsOfTrainingFiles[0], res["embedder"], res["label"], res["user_id"], res["epochsNumbers"], headers)

    except Exception as e:
        log.error("json error: ")
        log.error(e)
        await publishMsgOnRabbitMQ({"error": str(e)}, res["email"])
        raise e