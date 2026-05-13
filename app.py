from fastapi import FastAPI, UploadFile, File
from model_manager import ModelManager
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import tempfile

app = FastAPI()

# 服务启动时预热模型
model = ModelManager.get_model()


@app.post("/asr")
async def asr(file: UploadFile = File(...)):
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(await file.read())
        temp_path = f.name

    # 获取单例模型
    model = ModelManager.get_model()

    # 推理
    res = model.generate(
        input=temp_path,
        batch_size_s=300
    )

    # 清洗特殊token
    text = rich_transcription_postprocess(
        res[0]["text"]
    )

    return {
        "text": text
    }
