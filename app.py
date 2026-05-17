import json
import tempfile

import numpy as np
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from model_manager import ModelManager
from funasr.utils.postprocess_utils import rich_transcription_postprocess

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


def pcm16le_to_float32(audio_bytes: bytes) -> np.ndarray:
    # 浏览器或前端实时录音时，常见做法是把 PCM 原始字节流直接发给服务端。
    # 这里假设客户端发送的是：
    # 1. 单声道
    # 2. 16k 采样率
    # 3. 16-bit PCM little-endian
    #
    # FunASR 流式模型更常见的 Python 输入是 numpy 音频采样点，
    # 所以这里先把 bytes 转成 numpy.ndarray，再做归一化。
    pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
    return pcm16.astype(np.float32) / 32768.0


def stream_transcribe_audio(
    audio_chunk: np.ndarray,
    cache: dict,
    is_final: bool,
    chunk_size: list[int],
    encoder_chunk_look_back: int,
    decoder_chunk_look_back: int,
) -> str:
    # 这个函数故意写成“同步函数”。
    #
    # 原因和前面的 HTTP /asr 一样：
    # model.generate(...) 是阻塞型推理，不应该直接放在 async WebSocket 协程里执行，
    # 否则一个连接在推理时，会卡住事件循环，影响其他连接。
    # 所以后面会用 run_in_threadpool(...) 来调用它。
    model = ModelManager.get_streaming_model()
    res = model.generate(
        input=audio_chunk,
        cache=cache,
        is_final=is_final,
        chunk_size=chunk_size,
        encoder_chunk_look_back=encoder_chunk_look_back,
        decoder_chunk_look_back=decoder_chunk_look_back,
    )

    if not res:
        return ""

    text = res[0].get("text", "")
    if not text:
        return ""

    return rich_transcription_postprocess(text)


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


@app.websocket("/ws/asr")
async def websocket_asr(websocket: WebSocket):
    # WebSocket 和普通 HTTP 最大的区别是：
    # HTTP 通常是“一问一答”。
    # WebSocket 是“先建立连接，然后在这条连接上持续收发很多次消息”。
    #
    # 这个接口的学习协议如下：
    #
    # 第一步：客户端建立连接
    # ws://127.0.0.1:8000/ws/asr
    #
    # 第二步：先发送一条 JSON 文本消息，告诉服务端流式识别参数
    # 例如：
    # {
    #   "event": "start",
    #   "sample_rate": 16000,
    #   "chunk_size": [0, 10, 5],
    #   "encoder_chunk_look_back": 4,
    #   "decoder_chunk_look_back": 1
    # }
    #
    # 第三步：持续发送“二进制音频消息”
    # 每条二进制消息都表示一段 PCM 音频字节流。
    #
    # 第四步：音频发送完后，再发一条 JSON 文本消息结束本次识别
    # {
    #   "event": "end"
    # }
    #
    # 服务端会不断返回 JSON 结果：
    # 1. partial: 中间识别结果
    # 2. final: 最终识别结果
    await websocket.accept()

    # 给第一次接触 WebSocket 的你一个“握手成功”的提示消息。
    # 这样你一连上就知道服务端已经准备好了。
    await websocket.send_json(
        {
            "event": "ready",
            "message": "WebSocket 已连接。先发 start 配置，再发二进制 PCM 音频，最后发 end。"
        }
    )

    sample_rate = 16000
    chunk_size = [0, 10, 5]
    encoder_chunk_look_back = 4
    decoder_chunk_look_back = 1

    # 每个 WebSocket 连接都必须维护自己独立的 cache。
    # 不能多个用户共用一个 cache，
    # 否则不同人的音频上下文会串在一起，识别结果一定混乱。
    cache = {}

    # 这是当前连接的“累计转写结果”。
    # 流式模型每次可能返回一小段增量文本，这里把它们拼起来，方便观察实时效果。
    transcript_parts = []

    # 为什么需要 buffer？
    # 因为浏览器或客户端发来的二进制包大小不一定正好等于模型需要的 chunk。
    # 所以服务端要先把零散字节攒起来，够一个 chunk 了再送给模型。
    pcm_buffer = bytearray()

    # 这里把“一个 chunk 需要多少采样点”算出来。
    # 官方示例中：chunk_stride = chunk_size[1] * 960
    # 当 chunk_size[1] = 10 时，就是 9600 个采样点，也就是约 600ms。
    def get_chunk_bytes() -> int:
        chunk_samples = chunk_size[1] * 960
        return chunk_samples * 2

    async def run_one_chunk(chunk_bytes: bytes, is_final: bool) -> None:
        if not chunk_bytes:
            return

        audio_chunk = pcm16le_to_float32(chunk_bytes)
        text = await run_in_threadpool(
            stream_transcribe_audio,
            audio_chunk,
            cache,
            is_final,
            chunk_size,
            encoder_chunk_look_back,
            decoder_chunk_look_back,
        )

        if text:
            transcript_parts.append(text)

        await websocket.send_json(
            {
                "event": "final" if is_final else "partial",
                "text": text,
                "full_text": "".join(transcript_parts),
            }
        )

    try:
        while True:
            # receive() 会返回一个“原始消息对象”。
            # 这样我们就能自己区分：
            # 1. 这是文本消息（JSON 配置/结束指令）
            # 2. 这是二进制消息（音频数据）
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if message.get("text") is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {
                            "event": "error",
                            "message": "文本消息必须是合法 JSON。"
                        }
                    )
                    continue

                event = payload.get("event")

                if event == "start":
                    # 这里允许客户端覆盖默认参数，方便你实验不同的实时配置。
                    sample_rate = int(payload.get("sample_rate", 16000))
                    chunk_size = payload.get("chunk_size", [0, 10, 5])
                    encoder_chunk_look_back = int(
                        payload.get("encoder_chunk_look_back", 4)
                    )
                    decoder_chunk_look_back = int(
                        payload.get("decoder_chunk_look_back", 1)
                    )

                    cache.clear()
                    transcript_parts.clear()
                    pcm_buffer.clear()

                    await websocket.send_json(
                        {
                            "event": "start_ack",
                            "message": "流式识别参数已生效，可以开始发送二进制音频。",
                            "sample_rate": sample_rate,
                            "chunk_size": chunk_size,
                            "chunk_duration_ms": chunk_size[1] * 60,
                        }
                    )
                    continue

                if event == "end":
                    # 这里有一个很重要的“小技巧”：
                    # 平时实时处理中，我们会故意把“最后一个完整 chunk”留在 buffer 里，
                    # 不立刻推理。这样当客户端发送 end 时，
                    # 我们就能把最后一块音频带着 is_final=True 一起送给模型，
                    # 避免最后几个字丢失。
                    while len(pcm_buffer) > get_chunk_bytes():
                        current = bytes(pcm_buffer[:get_chunk_bytes()])
                        del pcm_buffer[:get_chunk_bytes()]
                        await run_one_chunk(current, is_final=False)

                    await run_one_chunk(bytes(pcm_buffer), is_final=True)
                    pcm_buffer.clear()
                    continue

                await websocket.send_json(
                    {
                        "event": "error",
                        "message": "暂不支持的事件类型，请使用 start 或 end。"
                    }
                )
                continue

            if message.get("bytes") is not None:
                pcm_buffer.extend(message["bytes"])

                # 这里不要写成 >=，而是写成 >。
                # 目的就是始终把“最后一个完整 chunk”先留在 buffer 里。
                #
                # 举例：
                # 如果客户端刚好发来了 3 个整 chunk，
                # 那么我们只先处理前 2 个，最后 1 个保留。
                # 等客户端发 end 时，再把最后 1 个用 is_final=True 推理。
                while len(pcm_buffer) > get_chunk_bytes():
                    current = bytes(pcm_buffer[:get_chunk_bytes()])
                    del pcm_buffer[:get_chunk_bytes()]
                    await run_one_chunk(current, is_final=False)

                continue

            await websocket.send_json(
                {
                    "event": "error",
                    "message": "无法识别的 WebSocket 消息类型。"
                }
            )
    except WebSocketDisconnect:
        # 客户端主动断开连接时，FastAPI 会抛出这个异常。
        # 这里直接吞掉即可，因为这属于正常关闭场景。
        pass
