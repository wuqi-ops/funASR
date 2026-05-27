import numpy as np
import soundfile as sf


def merge_to_stereo(
        left_wav_path,
        right_wav_path,
        output_path="stereo_output.wav"
):
    """
    合并两个单声道 wav 为双声道 wav

    左声道:
        通常放客服

    右声道:
        通常放客户
    """

    # 读取音频
    left_audio, left_sr = sf.read(left_wav_path)
    right_audio, right_sr = sf.read(right_wav_path)

    # 检查采样率
    if left_sr != right_sr:
        raise ValueError(
            f"采样率不一致: "
            f"left={left_sr}, right={right_sr}"
        )

    # 检查是否单声道
    if len(left_audio.shape) != 1:
        raise ValueError("左音频不是单声道")

    if len(right_audio.shape) != 1:
        raise ValueError("右音频不是单声道")

    # 对齐长度
    max_len = max(len(left_audio), len(right_audio))

    left_audio = np.pad(
        left_audio,
        (0, max_len - len(left_audio))
    )

    right_audio = np.pad(
        right_audio,
        (0, max_len - len(right_audio))
    )

    # 合并为双声道
    stereo_audio = np.stack(
        [left_audio, right_audio],
        axis=1
    )

    # 导出 wav
    sf.write(
        output_path,
        stereo_audio,
        left_sr
    )

    print("双声道文件已生成:")
    print(output_path)

    print("shape:", stereo_audio.shape)
    print("sample rate:", left_sr)


if __name__ == "__main__":

    merge_to_stereo(
        left_wav_path="/Users/wuqi/Downloads/Chrome Download/标准语音测试包/中文/OSR_cn_000_0075_8k.wav",
        right_wav_path="/Users/wuqi/Downloads/Chrome Download/标准语音测试包/中文/OSR_cn_000_0073_8k.wav",
        output_path="callcenter_stereo.wav"
    )