import logging as log

from sqlalchemy.orm import Session

from db.models.AIModel_labels import AIModel_labels, User_AIModel_Labels



def add_new_AIModel_label(label: str, id: int, email: str, db: Session):
    log.info("Adding new label")
    try:
        newLabel = AIModel_labels(
            label_name = label,
            created_by = email
        )
        db.add(newLabel)

        db.commit()

        newUserLabel = User_AIModel_Labels(
            user_id = id,
            label_id = newLabel.id,
            created_by = email
        )

        db.add(newUserLabel)

        db.commit()
        db.refresh(newUserLabel)
        db.refresh(newLabel)

        return newLabel
    except Exception as e:
        print(e)
        raise Exception

