from fastapi import FastAPI, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from model_manager import ModelManager
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import tempfile

app = FastAPI()

# 服务启动时预热模型
model = ModelManager.get_model()


def transcribe_audio(temp_path: str) -> str:
    model = ModelManager.get_model()
    res = model.generate(
        input=temp_path,
        batch_size_s=300
    )
    return rich_transcription_postprocess(res[0]["text"])


@app.post("/asr")
async def asr(file: UploadFile = File(...)):
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(await file.read())
        temp_path = f.name

    # 把阻塞型推理放到线程池里执行，避免卡住事件循环
    text = await run_in_threadpool(transcribe_audio, temp_path)

    return {
        "text": text
    }
