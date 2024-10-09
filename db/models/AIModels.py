from datetime import datetime

from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from db.base_class import Base


class AIModelsData(Base):
    __tablename__ = "AIModelsData"

    id = Column(Integer, primary_key=True, index=True)
    path = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime)
    deleted_at = Column(DateTime)
    created_by = Column(String(200), nullable=False)
    updated_by = Column(String(200))
    deleted_by = Column(String(200))

class AIModelsData_labels(Base):
    __tablename__ = "AIModelsData_labels"

    id = Column(Integer, primary_key=True, index=True)
    AIModel_label_id = Column(Integer, nullable=False)
    AIModelsData_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime)
    deleted_at = Column(DateTime)
    created_by = Column(String(200), nullable=False)
    updated_by = Column(String(200))
    deleted_by = Column(String(200))
    status = Column(String(200), default="active")

