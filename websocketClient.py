import asyncio
import json
import wave
import websockets

WS_URL = "ws://127.0.0.1:8000/ws/asr"
CHUNK_MS = 200


async def receive_loop(ws):
    """
    专门负责持续接收 ASR 结果
    """
    try:
        while True:
            result = await ws.recv()
            print("ASR结果:", result)
    except websockets.ConnectionClosed:
        print("连接关闭")


async def stream_wav(wav_path: str):
    async with websockets.connect(WS_URL) as ws:

        # 启动“后台接收任务”
        recv_task = asyncio.create_task(receive_loop(ws))

        # 发送 start
        await ws.send(json.dumps({
            "event": "start",
            "sample_rate": 8000,   # 你的wav是8k
            "chunk_size": [0, 10, 5],
            "encoder_chunk_look_back": 4,
            "decoder_chunk_look_back": 1
        }))

        # 打开 wav
        wf = wave.open(wav_path, "rb")

        print("采样率:", wf.getframerate())
        print("声道数:", wf.getnchannels())
        print("位深:", wf.getsampwidth() * 8)

        sample_rate = wf.getframerate()

        # 200ms 对应的采样点数
        chunk_samples = int(sample_rate * CHUNK_MS / 1000)

        while True:
            # 读取 PCM
            pcm_data = wf.readframes(chunk_samples)

            if not pcm_data:
                break

            print("发送音频字节:", len(pcm_data))

            # 发送二进制音频
            await ws.send(pcm_data)

            # 模拟真实实时语音
            await asyncio.sleep(CHUNK_MS / 1000)

        # 通知结束
        await ws.send(json.dumps({
            "event": "end"
        }))

        print("音频发送结束")

        # 等待几秒接收最终结果
        await asyncio.sleep(5)

        recv_task.cancel()


if __name__ == "__main__":
    wav_path = "/Users/wuqi/Downloads/Chrome Download/标准语音测试包/中文/OSR_cn_000_0075_8k.wav"

    asyncio.run(stream_wav(wav_path))