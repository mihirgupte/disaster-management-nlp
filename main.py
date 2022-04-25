from fastapi import FastAPI
from pipeline import predict
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

app = FastAPI()

class UserInput(BaseModel):
    userTweet: str

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <html>
        <head>
            <title>Disaster management API</title>
        </head>
        <body>
            <h1>Disaster management API</h1>
            <p>Send a POST request with the text to get back the predicted score</p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/predict")
async def get_prediction(user_input: UserInput):
    return predict(user_input.userTweet)