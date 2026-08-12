import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


class Annotation(BaseModel):
    query_id: str
    doc_id: str
    relevance: int
    timestamp: str


class AnnotationManager:
    def __init__(self, annotations_path: Path):
        self.annotations_path = annotations_path
        self.annotations: list[Annotation] = []
        self._load_existing()

    def _load_existing(self):
        if self.annotations_path.exists():
            with open(self.annotations_path) as f:
                data = json.load(f)
            self.annotations = [Annotation(**item) for item in data]

    def add_annotation(self, query_id: str, doc_id: str, relevance: int):
        annotation = Annotation(
            query_id=query_id,
            doc_id=doc_id,
            relevance=relevance,
            timestamp=datetime.utcnow().isoformat(),
        )
        self.annotations.append(annotation)
        self._save()

    def _save(self):
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.annotations_path, "w") as f:
            json.dump([a.model_dump() for a in self.annotations], f, indent=2)

    def is_annotated(self, query_id: str, doc_id: str) -> bool:
        return any(a.query_id == query_id and a.doc_id == doc_id for a in self.annotations)

    def get_annotation(self, query_id: str, doc_id: str) -> Annotation | None:
        for annotation in self.annotations:
            if annotation.query_id == query_id and annotation.doc_id == doc_id:
                return annotation
        return None

    def get_progress(self, total_items: int) -> dict:
        completed = len(self.annotations)
        return {
            "completed": completed,
            "total": total_items,
            "percentage": (completed / total_items * 100) if total_items > 0 else 0,
        }

    def undo_last(self) -> Annotation | None:
        if not self.annotations:
            return None
        last_annotation = self.annotations.pop()
        self._save()
        return last_annotation
