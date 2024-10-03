from azure.storage.blob import BlobServiceClient
import os, json, logging as log
from io import BytesIO

# Create the BlobServiceClient object
blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))

def deleteAndSaveExcelDataToAzure(path: str, data, isBlobExist: True):
    try:
        # Convert JSON data to bytes
        binary_data = json.dumps(data).encode('utf-8')

        if (isBlobExist):
            blob_client = blob_service_client.get_blob_client(container='excelfiles', blob=path)

            blob_client.append_block(BytesIO(binary_data))
        else:
            blob_client = blob_service_client.get_blob_client(container='excelfiles', blob=path)

            blob_client.upload_blob(BytesIO(binary_data))

    except Exception as e:
        log.info("exception on excel azure save: ", e)