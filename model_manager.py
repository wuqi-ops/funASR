from funasr import AutoModel


# 定义一个“模型管理类”
# 作用：
# 专门负责加载和获取 ASR 模型
class ModelManager:
    # 类变量（非常重要）
    # 整个类只有一份
    # 初始值为 None，表示“模型还没加载”
    _model = None

    # @classmethod 表示：
    # 这是“类方法”
    # 不需要 new 对象也能直接调用
    #
    # 可以这样调用：
    # ModelManager.get_model()
    #
    # cls 代表当前类本身
    @classmethod
    def get_model(cls):
        # 单例判断
        #
        # 如果模型还没加载
        # 才进行加载
        #
        # 第一次：
        # cls._model == None
        #
        # 后续：
        # cls._model 已经有值了
        # 不会再次加载
        if cls._model is None:
            print("开始加载ASR模型...")

            # 加载 FunASR 模型
            cls._model = AutoModel(

                # 主 ASR 模型
                # 用于语音识别
                model="iic/SenseVoiceSmall",

                # VAD（Voice Activity Detection）
                # 用于检测哪里有人声、哪里是静音
                vad_model="fsmn-vad",

                # 标点模型
                # 自动给识别结果加标点
                punc_model="ct-punc",

                # 允许执行模型仓库里的自定义代码
                # 很多模型必须开启
                trust_remote_code=True,

                # 指定运行设备
                #
                # mps:
                # Apple M1/M2 GPU加速
                #
                # 以后Linux GPU:
                # device="cuda"
                #
                # CPU:
                # device="cpu"
                device="mps"
            )

            print("ASR模型加载完成")

        # 返回模型对象
        #
        # 第一次：
        # 返回刚加载的模型
        #
        # 后续：
        # 直接返回已经存在的模型
        return cls._model
