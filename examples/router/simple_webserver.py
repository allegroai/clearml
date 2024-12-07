"""
A simple webserver, used as a tool to showcase the capabilities of
ClearML HTTP router. See `http_router.py` for more details.
"""


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()

actions = {
    "1": {"name": "Action 1", "description": "This is model action 1"},
    "2": {"name": "Action 2", "description": "This is model action 2"},
    "3": {"name": "Action 3", "description": "This is model action 3"},
}


class Item(BaseModel):
    name: str
    description: str


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application!"}


@app.get("/serve/{action}", response_model=Item)
def read_item(action: str):
    if action in actions:
        return actions[action]
    else:
        raise HTTPException(status_code=404, detail="Item not found")


if __name__ == "__main__":
    uvicorn.run(
        "simple_webserver:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
