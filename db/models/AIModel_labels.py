from datetime import datetime

from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from db.base_class import Base


class AIModel_labels(Base):
    __tablename__ = "AIModel_labels"

    id = Column(Integer, primary_key=True, index=True)
    label_name = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime)
    deleted_at = Column(DateTime)
    created_by = Column(String(200), nullable=False)
    updated_by = Column(String(200))
    deleted_by = Column(String(200))
    status = Column(String(200), default="active")


class TrainFile(Base):
    __tablename__ = "train_files"

    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String, nullable=True)
    target_column = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime)
    deleted_at = Column(DateTime)
    created_by = Column(String(200), nullable=False)
    updated_by = Column(String(200))
    deleted_by = Column(String(200))
    status = Column(String(200), default="active")


class TrainFile_AIModel_Labels(Base):
    __tablename__ = "trainfile_AIModel_labels"

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(Integer, nullable=False)
    label_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime)
    deleted_at = Column(DateTime)
    created_by = Column(String(200), nullable=False)
    updated_by = Column(String(200))
    deleted_by = Column(String(200))
    status = Column(String(200), default="active")


class User_AIModel_Labels(Base):
    __tablename__ = "user_AIModel_labels"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    label_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime)
    deleted_at = Column(DateTime)
    created_by = Column(String(200), nullable=False)
    updated_by = Column(String(200))
    deleted_by = Column(String(200))
    status = Column(String(200), default="active")
