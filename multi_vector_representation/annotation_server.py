from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.evaluation.pooling import ResultPooler, QueryPool
from src.evaluation.annotations import AnnotationManager

app = FastAPI(title="Relevance Annotation UI")

DATA_DIR = Path("data")
POOLS_PATH = DATA_DIR / "query_pools.json"
ANNOTATIONS_PATH = DATA_DIR / "annotations.json"

templates = Jinja2Templates(directory="templates")

pools: list[QueryPool] = []
annotation_manager: AnnotationManager = None


@app.on_event("startup")
async def startup():
    global pools, annotation_manager

    pools = ResultPooler.load_pools(POOLS_PATH)
    annotation_manager = AnnotationManager(ANNOTATIONS_PATH)

    print(f"Loaded {len(pools)} query pools")
    print(f"Loaded {len(annotation_manager.annotations)} existing annotations")


class AnnotationRequest(BaseModel):
    query_id: str
    doc_id: str
    relevance: int


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("annotate.html", {"request": request})


@app.get("/api/next")
async def get_next():
    for pool in pools:
        for doc in pool.pooled_docs:
            if not annotation_manager.is_annotated(pool.query_id, doc.doc_id):
                total_items = sum(len(p.pooled_docs) for p in pools)
                progress = annotation_manager.get_progress(total_items)

                is_ground_truth = doc.doc_id in pool.ground_truth_doc_ids

                return JSONResponse(
                    {
                        "query_id": pool.query_id,
                        "query_text": pool.query_text,
                        "doc": {
                            "doc_id": doc.doc_id,
                            "title": doc.title,
                            "content_type": doc.content_type,
                            "preview": doc.preview,
                        },
                        "is_ground_truth": is_ground_truth,
                        "progress": progress,
                    }
                )

    total_items = sum(len(p.pooled_docs) for p in pools)
    return JSONResponse(
        {"complete": True, "progress": annotation_manager.get_progress(total_items)}
    )


@app.post("/api/annotate")
async def annotate(request: AnnotationRequest):
    annotation_manager.add_annotation(request.query_id, request.doc_id, request.relevance)

    total_items = sum(len(p.pooled_docs) for p in pools)
    progress = annotation_manager.get_progress(total_items)

    return JSONResponse({"success": True, "progress": progress})


@app.get("/api/progress")
async def get_progress():
    total_items = sum(len(p.pooled_docs) for p in pools)
    return JSONResponse(annotation_manager.get_progress(total_items))


@app.post("/api/undo")
async def undo_last():
    undone = annotation_manager.undo_last()

    if not undone:
        return JSONResponse({"success": False, "message": "No annotations to undo"})

    for pool in pools:
        for doc in pool.pooled_docs:
            if pool.query_id == undone.query_id and doc.doc_id == undone.doc_id:
                is_ground_truth = doc.doc_id in pool.ground_truth_doc_ids
                total_items = sum(len(p.pooled_docs) for p in pools)
                progress = annotation_manager.get_progress(total_items)

                return JSONResponse(
                    {
                        "success": True,
                        "query_id": pool.query_id,
                        "query_text": pool.query_text,
                        "doc": {
                            "doc_id": doc.doc_id,
                            "title": doc.title,
                            "content_type": doc.content_type,
                            "preview": doc.preview,
                        },
                        "is_ground_truth": is_ground_truth,
                        "progress": progress,
                    }
                )

    return JSONResponse({"success": False, "message": "Could not find document"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
