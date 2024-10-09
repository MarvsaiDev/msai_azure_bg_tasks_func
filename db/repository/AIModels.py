import logging as log

from sqlalchemy.orm import Session


from db.models.AIModels import AIModelsData, AIModelsData_labels


def add_new_AIModelData(path: str, email: str, db: Session):
    log.info("Adding new AI Model path DB")
    try:
        AIModelData = AIModelsData(
            path = path,
            created_by = email
        )
        db.add(AIModelData)

        db.commit()
        db.refresh(AIModelData)

        return AIModelData
    except Exception as e:
        print(e)
        raise Exception
    
def add_AIModelData_and_label(labelID: int, AIModelID: int, email: str, db: Session):
    log.info("Adding new AIModel with label DB")
    try:
        AIModelData_Label = AIModelsData_labels(
            AIModel_label_id = labelID,
            AIModelsData_id = AIModelID,
            created_by = email
        )
        db.add(AIModelData_Label)

        db.commit()
        db.refresh(AIModelData_Label)

        return AIModelData_Label
    except Exception as e:
        print(e)
        raise Exception