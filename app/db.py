"""Task 5 - database trail (SQLAlchemy: SQLite locally, Postgres in Compose)."""
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, select
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    filename = Column(String(200))
    label = Column(String(20))
    confidence = Column(Float)
    model_name = Column(String(100))
    model_version = Column(String(20))

class Store:
    def __init__(self, url):
        self.engine = create_engine(url, pool_pre_ping=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine)

    def add(self, **kw):
        with self.Session() as s:
            s.add(Prediction(**kw)); s.commit()

    def recent(self, n=100):
        with self.Session() as s:
            return list(s.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(n)))
