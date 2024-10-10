from ai_files.EmbeddingModel import BAIEmbeddingModel, OpenAIEmbeddingModel
import logging as log



embedding_model = BAIEmbeddingModel()
OPENAI_embedding_model = OpenAIEmbeddingModel()

def check_All_Rows_Combined_Text_Length(rows, limit=7000):
    fileTexts = ''

    for row in rows:
        fileTexts += row

    if (len(fileTexts) > limit):
        return True
    
    return False

def embeddingTexts(arr, embedder="OpenAI"):
    data = []

    if (embedder == "transformer"):
        # This line is using Transformer embedding function to embedd the text
        embedding_Texts = embedding_model.get_embedding_torch(arr)
    else:
        # This line is using OPENAI embedding function to embedd the text
        embedding_Texts = OPENAI_embedding_model.get_embedding(arr)

    for idx, embedding_Text in enumerate(embedding_Texts):
        data.append({})

        for idx, embedding in enumerate(embedding_Text):
            data[-1]['emb_'+str(idx)] = embedding

    return data

def embdeddingFunc(df, h, embedder="OpenAI", columns = [], selectedColumnIndex = 0):
    log.info("embedding start")
    data = []

    targetColumns = []
    rowTexts = []
    targetColumnName = ""
    headers = h
    for idx, row in df.iterrows():
        if (idx == 0):
            if (len(headers) < 1):
                headers = row.array
                targetColumnName = str(row.get(selectedColumnIndex))
            else:
                targetColumnName = headers[selectedColumnIndex]          

        fileText = ''

        for i, col in enumerate(columns):
            if (i != selectedColumnIndex):
                fileText += " " + str(headers[i]) + ": "+ str(row.get(col)) + " |"
            else:
                selectedColumnName = col

        # this is bug
        # fileText = targetColumnName + ": " + str(row.get(selectedColumnName)) + " |" + fileText

        if(idx == 1):
            log.info("file text value: ")
            log.info(fileText)

        targetColumns.append(str(row.get(selectedColumnName)))
        rowTexts.append(fileText)

        isLengthGreaterThenLimit = check_All_Rows_Combined_Text_Length(rowTexts, 7000)

        if (isLengthGreaterThenLimit or idx == df.shape[0] - 1):
            embeddedRows = embeddingTexts(rowTexts, embedder=embedder)

            for idx1, row in enumerate(embeddedRows):
                data.append({targetColumnName: targetColumns[idx1]})

                for col in row:
                    data[-1][col] = row[col]
            
            targetColumns = []
            rowTexts = []

    return data, headers

def embeddingFuncForInference(df, embedder="OpenAI", columns = []):
    data = []

    rowTexts = []
    labels = []
    for idx, row in df.iterrows():
        if (idx == 0):
            for r in row:
                labels.append(r)
            continue


        fileText = ''

        for i, col in enumerate(columns):
            fileText += " " + str(labels[i]) + ": "+ str(row.get(col)) + " |"
        
        fileText = fileText

        rowTexts.append(fileText)

        isLengthGreaterThenLimit = check_All_Rows_Combined_Text_Length(rowTexts, 7000)

        if (isLengthGreaterThenLimit or idx == df.shape[0] - 1):
            embeddedRows = embeddingTexts(rowTexts, embedder=embedder)

            for row in embeddedRows:
                data.append({})

                for col in row:
                    data[-1][col] = row[col]
            
            rowTexts = []

    return data