from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag import router as rag_router


app = FastAPI()

# CORS setting
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include RAG router
app.include_router(rag_router)

# root
@app.get("/")
async def root():
    return {"message": "RAG API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
# swagger: http://localhost:8000/docs#/