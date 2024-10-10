from io import BytesIO
from sqlalchemy import text
import torch, os, json
import pandas as pd
from ai_files.AITrainingClass import TrainAIModel
import logging as log

from db.repository.AIModels import add_AIModelData_and_label, add_new_AIModelData
from db.repository.labels_users import add_new_AIModel_label
from db.connection import SessionLocal
from utils.AzureStorage import SaveExcelDataToAzure
from utils.RabbitMQ import publishMsgOnRabbitMQ
from utils.embeddingFunctions import embdeddingFunc

async def EmbeddingFile(blob_service_client, containerName, container, head, res):
    # getting all blobs paths in user's container
    blobs = container.list_blob_names()
    # variable to track is embedding is started or not
    isEmbeddingNotStarted = True
    # path for embedding file
    pathsOfTrainingData = []
    count = 1
    # this variable will be responsible for holder the data header row columns names
    headers = []
    # iterating each blob
    for blob in blobs:
        # splitting the blob path
        h, t = os.path.split(os.path.normpath(blob))

        # checking if blob is same as the current blob
        if (h == head):
            # getting the blob
            blob_client = container.get_blob_client(blob)
            # get the blob data
            blob_data = blob_client.download_blob().readall()

            # Decode binary data to string from blob
            blob_str = blob_data.decode('utf-8')
            
            # Convert string to Python list
            blob_list = json.loads(blob_str)

            # create data frame from list
            df = pd.DataFrame(blob_list)
            # getting columns
            columns = df.columns
            # getting index for the targetColumn
            selectedColumnIndex = int(res["columnNum"]) - 1
            # checking if the embedding is not started already
            if (isEmbeddingNotStarted):
                # if it is not started already, tell queue that embedding started
                await publishMsgOnRabbitMQ({"task": "embedding", "condition": "start"}, res["email"])
                # setting that, embedding started
                isEmbeddingNotStarted = False

            # message for telling the embedding is started on a specific blob
            await publishMsgOnRabbitMQ({"embedding on blob": str(blob)}, res["email"])

            # embedding the blob data and get list of embedded data
            encoded_df, headers = embdeddingFunc(df, headers, embedder=res["embedder"], columns=columns, selectedColumnIndex=selectedColumnIndex)

            # message for telling the embedding is done on a specific blob
            await publishMsgOnRabbitMQ({"embedding done on": str(blob)}, res["email"])

            # the path for the training_data in which the embedding data is stored
            newPath = os.path.join(h, f"training_data{count}.json")

            # message for telling saving a specific blob
            await publishMsgOnRabbitMQ({"saving embedded blob: ": str(blob)}, res["email"])

            # and saving training data file on azure storage
            SaveExcelDataToAzure(blob_service_client, newPath, encoded_df, containerName)

            pathsOfTrainingData.append(newPath)

            count += 1

            # deleting the current blob from azure storage, because it is not required anymore
            container.delete_blob(blob, delete_snapshots="include")

            # message for telling that a specific blob is deleted
            await publishMsgOnRabbitMQ({"deleted blob: ": str(blob)}, res["email"])

    # if embedding is not started, then it means there is an error with embedding
    if (isEmbeddingNotStarted):
        await publishMsgOnRabbitMQ({"task": "embedding", "condition": "failed"}, res["email"])
    else:
        await publishMsgOnRabbitMQ({"task": "embedding", "condition": "completed"}, res["email"])

    return pathsOfTrainingData

async def trainingFunc(df, email, container, embeddingFilePath, embedder, label, id):
    try:
        # selectedColumnIndex = int(columnNum) - 1
        selectedColumnName = df.columns[0]

        log.info(df.columns)
        log.info(df.columns[0])
       
        AIModel = TrainAIModel(selectedColumnName, df, encodedClasses=None, mode="train")

        await AIModel.train_model(email)

        await saving_model_data(AIModel, embedder, container, embeddingFilePath, email, label, id)

        await publishMsgOnRabbitMQ({"task": "complete"}, email)

    except Exception as e:
        log.info(e)
        await publishMsgOnRabbitMQ({"task": "training", "condition": "failed"}, email)

def extract_lowercase_and_numbers(input_string):
    result = ''.join(char for char in input_string if char.islower() or char.isdigit())
    return result

def get_or_create_container(blob_service_client, container_name):
    try:
        blob_service_client.create_container(container_name)
    except Exception as e:
        print(f"the exception for creating container {e}")


async def saving_model_data(AIModel: TrainAIModel, embedder:str, container, embeddingFilePath, email, label, id):
    await publishMsgOnRabbitMQ({"task": "saving", "condition": "continue"}, email)

    head, tail = os.path.split(embeddingFilePath)
    
    # Saving model data directly to Azure Blob Storage
    model_data = BytesIO()
    torch.save(AIModel.model.state_dict(), model_data)
    model_data.seek(0)
    await upload_blob(container, model_data, os.path.join(head, "model_data.pt"))

    # Saving model optimizer data directly to Azure Blob Storage
    optimizer_data = BytesIO()
    torch.save(AIModel.optimizer.state_dict(), optimizer_data)
    optimizer_data.seek(0)
    await upload_blob(container, optimizer_data, os.path.join(head, "model_opti_data.pt"))

    
    # Saving classes name with encoding directly to Azure Blob Storage
    classes_data = {
        "encodedClasses": {int(key): colName for key, colName in AIModel.classesWithEncoding.items()},
        "embedder": embedder,
        "targetColumn": AIModel.targetColumnName,
        "AIMODELTYPE": AIModel.modelType
    }
    classes_json = BytesIO(json.dumps(classes_data).encode('utf-8'))
    await upload_blob(container, classes_json, os.path.join(head, "model_json.json"))

    await saveModelInformationInDB(label, head, id, email)



async def upload_blob(container, data, path):
    log.info(path)
    blob = container.get_blob_client(path)
    blob.upload_blob(data)


async def saveModelInformationInDB(label, filePath, id, email):
    try:
        log.info("database: ")
        db = SessionLocal()

        result = db.execute(text("SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE' AND table_schema = 'marvsaiHS_Staging';"))

        log.info(result)
        log.info(result.all())

        for row in result:
            log.info(row)
            log.info(row['table_name'])
    
        log.info(db)
        newLabel = add_new_AIModel_label(label, id, email, db)
        log.info(newLabel)
        newAIModelData  = add_new_AIModelData(filePath, email, db)
        log.info(newAIModelData)
        add_AIModelData_and_label(newLabel.id, newAIModelData.id, email, db)
        log.info("new data in db added")
    except Exception as e:
        await publishMsgOnRabbitMQ({"error db": str(e)}, email)
    finally:
        db.close()