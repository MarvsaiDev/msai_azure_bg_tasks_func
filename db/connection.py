import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = os.getenv("SQLALCHEMY_DATABASE_URL")

engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# from contextlib import contextmanager◘
# from typing import GeneratorÄ
# class DB_Class:
#     def __enter__(self):
#         self.db = SessionLocal()
#         return self.db

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         if self.db:
#             self.db.close()


# def get_db():
#     return DB_Class()