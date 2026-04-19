from fastapi import FastAPI
from app.api.endpoints import router
from app.config import settings
import uvicorn

app = FastAPI(
    title="Galamsey Segmentation Service",
    description="AI Service for detecting illegal mining footprints in satellite imagery",
    version="1.0.0"
)

# Include API routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "Welcome to Galamsey Segmentation Service API",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", 
        host=settings.API_HOST, 
        port=settings.API_PORT, 
        reload=True
    )
