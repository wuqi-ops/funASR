from funasr import AutoModel
import time


def main():
    print("开始加载模型...")

    start = time.time()

    # 加载模型
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        device="mps"
    )

    print(f"模型加载完成，耗时: {time.time() - start:.2f} 秒")

    print("开始语音识别...")

    start = time.time()

    # 识别音频
    result = model.generate(
        input="OSR_us_000_0032_8k.wav",
        batch_size_s=300,
        hotword='',
    )

    print(f"识别完成，耗时: {time.time() - start:.2f} 秒")

    print("\n识别结果：")
    print(result)


if __name__ == '__main__':
    main()
