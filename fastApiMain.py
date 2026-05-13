from fastapi import FastAPI, UploadFile
from funasr import AutoModel
import uvicorn
import shutil

app = FastAPI()

model = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    device="mps"
)


@app.post("/asr")
async def asr(file: UploadFile):
    file_path = file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = model.generate(
        input=file_path
    )

    return result


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
