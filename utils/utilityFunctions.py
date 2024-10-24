from io import BytesIO
import torch, os, json
import pandas as pd
from ai_files.AITrainingClass import TrainAIModel
import logging as log

from db.repository.AIModels import addAIModel, updateAIModel
from db.connection import SessionLocal
from utils.AzureStorage import SaveExcelDataToAzure
from utils.RabbitMQ import publishMsgOnRabbitMQ
from utils.embeddingFunctions import embdeddingFunc

async def EmbeddingFile(blob_service_client, containerName, container, head, res, resume = False):
    # this variable is used for showing the percentage of embedding
    rowCount = 0
    # getting all blobs paths in user's container
    blobs = container.list_blob_names()

    if resume:
        embedder, encodedClasses = await getEmbedderFromModelFile(blobs=blobs, container=container)
    else:
        embedder = res["embedder"]

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

        if resume and (t.find("resumeFile_") == -1):
            continue
        
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

            # stripping the space from start and end of each element
            df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

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
            encoded_df, headers, rowCount = await embdeddingFunc(df, headers, embedder=embedder, columns=columns, selectedColumnIndex=selectedColumnIndex, rowCount=rowCount, totalRowCount=res["totalRowCount"], email=res["email"])

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

    if resume:
        return pathsOfTrainingData, headers.tolist(), encodedClasses
    else:
        return pathsOfTrainingData, headers.tolist()


async def trainingFunc(df, email, container, embeddingFilePath, embedder, label, id, epochsNumbers, headers, targetColumnName):
    try:
        log.info(df.columns)
        log.info(df.columns[0])

        path, tail = os.path.split(embeddingFilePath)
       
        AIModel = TrainAIModel(targetColumnName, df, encodedClasses=None, mode="train")

        log.info("model initialized")

        await AIModel.train_model(email, epochsNumbers)

        await saving_model_data(AIModel, embedder, container, path, email, label, id, headers)

        await publishMsgOnRabbitMQ({"task": "complete"}, email)

    except Exception as e:
        log.info(e)
        log.error(e)
        await publishMsgOnRabbitMQ({"task": "training", "condition": "failed"}, email)

async def trainingResumeFunc(df, email, container, embeddingFilePath, embedder, label, id, epochsNumbers, headers, targetColumnName, encodedClasses):
    try:
        log.info(df.columns)
        log.info(df.columns[0])

        log.error(embeddingFilePath)

        path, tail = os.path.split(embeddingFilePath)

        AIModel = TrainAIModel(targetColumnName, df, encodedClasses=None, mode="train")

        await AIModel.resume_train_model(email, epochsNumbers, path, container, encodedClasses)

        await saving_model_data(AIModel, embedder, container, path, email, label, id, headers, resume=True)

        await publishMsgOnRabbitMQ({"task": "complete"}, email)

    except Exception as e:
        log.info(e)
        log.error(e)
        await publishMsgOnRabbitMQ({"task": "training", "condition": "failed"}, email)

def extract_lowercase_and_numbers(input_string):
    result = ''.join(char for char in input_string if char.islower() or char.isdigit())
    return result

def get_or_create_container(blob_service_client, container_name):
    try:
        blob_service_client.create_container(container_name)
    except Exception as e:
        print(f"the exception for creating container {e}")


async def saving_model_data(AIModel: TrainAIModel, embedder:str, container, path, email, label, id, headers = [], resume = False):
    try:
        await publishMsgOnRabbitMQ({"task": "saving", "condition": "continue"}, email)

        log.info(AIModel.model)
        
        # Saving model data directly to Azure Blob Storage
        model_data = BytesIO()
        torch.save(AIModel.model.state_dict(), model_data)
        model_data.seek(0)
        await upload_blob(container, model_data, os.path.join(path, "model_data.pt"))

        # Saving model optimizer data directly to Azure Blob Storage
        optimizer_data = BytesIO()
        torch.save(AIModel.optimizer.state_dict(), optimizer_data)
        optimizer_data.seek(0)
        await upload_blob(container, optimizer_data, os.path.join(path, "model_opti_data.pt"))

        # Saving confusion matrix to CSV directly to Azure Blob Storage
        confusionMatrixFile = pd.DataFrame(AIModel.cm)
        csv_data = BytesIO()
        confusionMatrixFile.to_csv(csv_data, index=False)
        csv_data.seek(0)
        await upload_blob(container, csv_data, os.path.join(path, "confusion_matrix.csv"))

        
        # Saving classes name with encoding directly to Azure Blob Storage
        classes_data = {
            "encodedClasses": {int(key): colName for key, colName in AIModel.classesWithEncoding.items()},
            "embedder": embedder,
            "targetColumn": AIModel.targetColumnName,
            "AIMODELTYPE": AIModel.modelType,
            "fileColumns": headers
        }
        classes_json = BytesIO(json.dumps(classes_data).encode('utf-8'))
        await upload_blob(container, classes_json, os.path.join(path, "model_json.json"))

        await saveModelInformationInDB(label, path, id, email, AIModel.accuray, AIModel.loss, resume=resume)
    except Exception as e:
        await publishMsgOnRabbitMQ({"error": str(e)}, email)
        raise e

async def getEmbedderFromModelFile(blobs, container):
    for blob in blobs:
        if blob.find("/model_json.json") != -1:
            # getting the blob
            blob_client = container.get_blob_client(blob)
            # get the blob data
            blob_data = blob_client.download_blob().readall()

            # Decode binary data to string from blob
            blob_str = blob_data.decode('utf-8')
            
            # Convert string to Python list
            blob_list = json.loads(blob_str)

            return blob_list["embedder"], blob_list["encodedClasses"]
    return ""


async def upload_blob(container, data, path):
    log.info(path)
    blob = container.get_blob_client(path)
    blob.upload_blob(data, overwrite=True)


async def get_target_column_encodede_classes_from_model_file(blobs, container):
    for blob in blobs:
        if blob.find("/model_json.json") != -1:
            # getting the blob
            blob_client = container.get_blob_client(blob)
            # get the blob data
            blob_data = blob_client.download_blob().readall()

            # Decode binary data to string from blob
            blob_str = blob_data.decode('utf-8')
            
            # Convert string to Python list
            blob_list = json.loads(blob_str)

            return blob_list["encodedClasses"]
    return ""


async def saveModelInformationInDB(label, filePath, id, email, acc, loss, resume = False):
    try:
        log.info("database: ")
        db = SessionLocal()

        if resume:
            updateAIModel(label, db, acc, loss)
        else:     
            addAIModel(path = filePath, email = email, label = label, user_id = id, db = db, acc = acc, loss = loss)

        log.info("new data in db added")
    except Exception as e:
        await publishMsgOnRabbitMQ({"error db": str(e)}, email)
        raise e
    finally:
        db.close()