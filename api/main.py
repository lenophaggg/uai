from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import secrets

from search_engine.query_processor import process_query, reload_models

app = FastAPI(title="University Search API", version="1.0.1")

jwt_scheme = HTTPBearer(auto_error=False)
basic_scheme = HTTPBasic(auto_error=False)

def verify_jwt(credentials: HTTPAuthorizationCredentials | None = Depends(jwt_scheme)):
    if credentials is None:
        return None
    return "user@university"

def verify_basic(creds: HTTPBasicCredentials | None = Depends(basic_scheme)):
    if creds is None:
        return
    ADMIN_USER, ADMIN_PASS = "admin", "s3cr3t"
    good = (
        secrets.compare_digest(creds.username, ADMIN_USER)
        and secrets.compare_digest(creds.password, ADMIN_PASS)
    )
    if not good:
        raise HTTPException(status_code=401, detail="Bad creds")

class QueryRequest(BaseModel):
    q: str
    k: int | None = 10

@app.post("/api/v1/query", tags=["public"])
async def query_endpoint(req: QueryRequest, _user=Depends(verify_jwt)):
    try:
        return process_query(req.q, k=req.k or 10)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            token = msg.get("jwt")
            try:
                verify_jwt(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token) if token else None)
            except HTTPException as e:
                await ws.send_json({"error": e.detail})
                continue
            try:
                res = process_query(msg.get("q", ""), k=int(msg.get("k", 10)))
                await ws.send_json(res)
            except Exception as e:
                await ws.send_json({"error": str(e)})
    except WebSocketDisconnect:
        pass

@app.get("/api/v1/health", tags=["public"])
async def health():
    return {"status": "ok"}

@app.post("/api/v1/admin/reload-model", tags=["admin"], dependencies=[Depends(verify_basic)])
async def reload_model_ep():
    reload_models()
    return {"status": "reloaded"}
