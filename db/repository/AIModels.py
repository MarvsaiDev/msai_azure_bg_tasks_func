import logging as log

from sqlalchemy.orm import Session

from db.models.aimodels import AIModels, UsersAIModels


def addAIModel(path: str, email: str, label: str, user_id: int, db: Session, acc: any, loss: any):
    log.info("Adding New AI Model")
    try:
        aimodel = AIModels(path = path, created_by = email, label = label, accuracy = acc, loss = loss)
        db.add(aimodel)

        db.commit()

        user_aimodel = UsersAIModels(user_id = user_id, aimodel_id = aimodel.id, created_by = email)
        db.add(user_aimodel)

        db.commit()
    except Exception as e:
        db.rollback()
        log.error(e)
        raise e