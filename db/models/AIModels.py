from datetime import datetime

from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from db.base_class import Base


class AIModels(Base):
    __tablename__ = "aimodels"

    id = Column(Integer, primary_key=True, index=True)
    label = Column(String(200), nullable=False)
    path = Column(String, nullable=False)
    accuracy = Column(String, nullable=False)
    loss = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime)
    deleted_at = Column(DateTime)
    created_by = Column(String(200), nullable=False)
    updated_by = Column(String(200))
    deleted_by = Column(String(200))

class UsersAIModels(Base):
    __tablename__ = "users_aimodels"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    aimodel_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime)
    deleted_at = Column(DateTime)
    created_by = Column(String(200), nullable=False)
    updated_by = Column(String(200))
    deleted_by = Column(String(200))