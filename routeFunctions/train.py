import json
import os
from azure.storage.blob import BlobServiceClient
import pandas as pd
import logging as log
from utils.RabbitMQ import publishMsgOnRabbitMQ
from utils.utilityFunctions import EmbeddingFile, extract_lowercase_and_numbers, get_or_create_container, trainingFunc


async def trainFunc(res):
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

        await get_or_create_container(blob_service_client, containerName)

        await publishMsgOnRabbitMQ({"container": "created", "container_name": containerName}, res["email"])

        # ####################################################################
        # get container client
        container = blob_service_client.get_container_client(containerName)

        # Embedding the file (which is saved in chunks) and saving file on path with training_data.csv file
        pathsOfTrainingFiles, headers = await EmbeddingFile(blob_service_client, containerName, container, head, res)

        await publishMsgOnRabbitMQ({"task": "training", "condition": "preparing"}, res["email"])

        if (len(pathsOfTrainingFiles) < 1):
            await publishMsgOnRabbitMQ({"error": "Embedding function returned empty"}, res["email"])
            return 

        # ####################################################################

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

        # Delete the first row
        targetColumnName = df.iloc[0, 0]
        df.drop([0], inplace=True)

        await trainingFunc(df, res["email"], container, pathsOfTrainingFiles[0], res["embedder"], res["label"], res["user_id"], res["epochsNumbers"], headers, targetColumnName)

    except Exception as e:
        log.error("json error: ")
        log.error(e)
        await publishMsgOnRabbitMQ({"error": str(e)}, res["email"])
        raise e