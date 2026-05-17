"""
MiMo-V2.5-TTS 语音合成提供商（小米官方 API）
使用小米 MiMo 开放平台的 /chat/completions 接口进行语音合成。

API 文档: https://platform.xiaomimimo.com/docs/zh-CN/usage-guide/speech-synthesis-v2.5
API Key: https://platform.xiaomimimo.com/console/api-keys

支持的模型:
  - mimo-v2.5-tts          : 预置音色模式
  - mimo-v2.5-tts-voicedesign : 文本设计音色
  - mimo-v2.5-tts-voiceclone  : 音色复刻

预置音色列表:
  中文女声: 冰糖、茉莉
  中文男声: 苏打、白桦
  英文女声: Mia、Chloe
  英文男声: Milo、Dean
  基础音色: mimo_default

调用地址:
  - 通用: https://api.xiaomimimo.com/v1
  - Token Plan 国内: https://token-plan-cn.xiaomimimo.com/v1
"""

import os
import time
import base64

from openai import AsyncOpenAI
from openai import APIError as OpenAIAPIError
from openai import APIConnectionError as OpenAIConnectionError
from openai import AuthenticationError as OpenAIAuthError
from openai import RateLimitError as OpenAIRateLimitError

from config.logger import setup_logging
from core.utils.util import check_model_key
from core.providers.tts.base import TTSProviderBase

TAG = __name__
logger = setup_logging()


class TTSProvider(TTSProviderBase):
    """MiMo-V2.5-TTS 语音合成提供商（小米官方 API）

    通过 OpenAI 兼容的 /chat/completions 接口调用 MiMo-V2.5-TTS 服务。
    目标文本放在 assistant 消息中，风格控制放在 user 消息中。

    配置参数:
        api_key           : API 密钥（必填）
        base_url          : API 基础 URL（默认: https://token-plan-cn.xiaomimimo.com/v1）
        model             : TTS 模型名称（默认: mimo-v2.5-tts）
        voice             : 预置音色 ID（必填，预置模式下）
        response_format   : 音频输出格式（默认: wav，可选: wav/pcm16/mp3）
        style_instruction : 自然语言风格指令（可选，放在 user message 中）
        streaming         : 是否使用流式调用（默认: false）
        output_dir        : 输出目录（默认: tmp/）
    """

    # 支持的音频格式
    SUPPORTED_FORMATS = ("wav", "pcm16", "mp3")

    # 预置音色列表
    PRESET_VOICES = {
        "mimo_default": "MiMo-默认（因部署集群而异）",
        "冰糖": "中文女声",
        "茉莉": "中文女声",
        "苏打": "中文男声",
        "白桦": "中文男声",
        "Mia": "英文女声",
        "Chloe": "英文女声",
        "Milo": "英文男声",
        "Dean": "英文男声",
    }

    # 支持的模型
    SUPPORTED_MODELS = (
        "mimo-v2.5-tts",
        "mimo-v2.5-tts-voicedesign",
        "mimo-v2.5-tts-voiceclone",
    )

    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)

        # API 认证配置
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://token-plan-cn.xiaomimimo.com/v1")

        # 模型配置
        self.model = config.get("model", "mimo-v2.5-tts")

        # 音色配置
        self.voice = config.get("voice", "")

        # 音频输出格式
        self.response_format = config.get("response_format", "wav")

        # 风格控制（自然语言指令，放在 user message 中）
        self.style_instruction = config.get("style_instruction", "")

        # 行为配置
        self.streaming = bool(config.get("streaming", False))

        # 音色克隆的音频样本路径（仅 voiceclone 模式下使用）
        self.clone_audio_file = config.get("clone_audio_file", "")

        # 是否开启文本智能润色（仅 voicedesign 模式下使用）
        self.optimize_text_preview = bool(config.get("optimize_text_preview", False))

        # 请求计数器
        self._request_count = 0

        # 参数校验
        self._validate_config()

        # 同步 audio_file_type，让 base.py 能正确识别音频格式
        self.audio_file_type = self.response_format

        # 初始化 OpenAI 兼容客户端
        # 小米 API 使用 "api-key" header 而非 "Authorization: Bearer"
        self._client = AsyncOpenAI(
            api_key="not-used",
            base_url=self.base_url,
            timeout=60.0,
            max_retries=2,
            default_headers={"api-key": self.api_key},
        )

        # 检查 API Key
        model_key_msg = check_model_key("TTS", self.api_key)
        if model_key_msg:
            logger.bind(tag=TAG).error(model_key_msg)
        else:
            logger.bind(tag=TAG).info(
                f"MiMo-V2.5-TTS (小米官方) 初始化完成 | "
                f"model={self.model} | voice={self.voice} | "
                f"format={self.response_format} | "
                f"streaming={self.streaming} | base_url={self.base_url}"
            )

    def _validate_config(self):
        """校验并修正配置参数"""
        # 校验模型
        if self.model not in self.SUPPORTED_MODELS:
            logger.bind(tag=TAG).warning(
                f"不支持的模型 '{self.model}'，已回退为 'mimo-v2.5-tts'。"
                f"支持的模型: {self.SUPPORTED_MODELS}"
            )
            self.model = "mimo-v2.5-tts"

        # 校验音频格式
        fmt = self.response_format.lower().strip(".")
        if fmt not in self.SUPPORTED_FORMATS:
            logger.bind(tag=TAG).warning(
                f"不支持的音频格式 '{self.response_format}'，"
                f"已回退为默认格式 'wav'。支持的格式: {self.SUPPORTED_FORMATS}"
            )
            self.response_format = "wav"
        else:
            self.response_format = fmt

        # 流式模式下必须使用 pcm16
        if self.streaming and self.response_format != "pcm16":
            logger.bind(tag=TAG).info(
                f"流式模式要求音频格式为 pcm16，已从 '{self.response_format}' 调整为 'pcm16'"
            )
            self.response_format = "pcm16"

        # 预置音色模式下必须指定 voice
        if self.model == "mimo-v2.5-tts" and not self.voice:
            logger.bind(tag=TAG).warning(
                f"预置音色模式(model=mimo-v2.5-tts)下，voice 参数为必填项。"
                f"可用预置音色: {list(self.PRESET_VOICES.keys())}"
            )

        # 确保输出目录存在
        if self.output_file:
            os.makedirs(self.output_file, exist_ok=True)

    def _sanitize_text(self, text: str) -> str:
        """清理文本中的非法字符"""
        if not text:
            return ""
        # 过滤控制字符（保留换行和空格）
        sanitized = "".join(
            ch for ch in text if ch == "\n" or ch == "\r" or ch == "\t" or ("\u0020" <= ch <= "\U0010FFFF")
        )
        return sanitized.strip()

    def _build_messages(self, text: str) -> list:
        """构建 chat completions 的消息列表

        规则:
        - assistant message: 必须包含要合成的文本
        - user message: 可选的风格控制指令

        Args:
            text: 要合成的目标文本

        Returns:
            messages 列表
        """
        messages = []

        # user message: 自然语言风格指令（可选）
        # voicedesign 模式下为必填
        if self.style_instruction:
            messages.append({
                "role": "user",
                "content": self.style_instruction,
            })

        # assistant message: 要合成的文本（必须）
        messages.append({
            "role": "assistant",
            "content": text,
        })

        return messages

    def _build_audio_param(self) -> dict:
        """构建 audio 参数

        Returns:
            audio 参数字典
        """
        audio = {
            "format": self.response_format,
        }

        if self.model == "mimo-v2.5-tts":
            # 预置音色模式
            audio["voice"] = self.voice
        elif self.model == "mimo-v2.5-tts-voicedesign":
            # 文本设计音色模式
            if self.optimize_text_preview:
                audio["optimize_text_preview"] = True
        elif self.model == "mimo-v2.5-tts-voiceclone":
            # 音色复刻模式
            if self.clone_audio_file and os.path.exists(self.clone_audio_file):
                import base64
                mime_type = "audio/mpeg" if self.clone_audio_file.endswith(".mp3") else "audio/wav"
                with open(self.clone_audio_file, "rb") as f:
                    audio_data = base64.b64encode(f.read()).decode("utf-8")
                audio["voice"] = f"data:{mime_type};base64,{audio_data}"
            else:
                logger.bind(tag=TAG).warning(
                    f"音色复刻模式需要配置有效的 clone_audio_file，当前文件不存在: {self.clone_audio_file}"
                )

        return audio

    def _build_request_params(self, text: str) -> dict:
        """构建完整的 API 请求参数

        Args:
            text: 要合成的文本

        Returns:
            请求参数字典
        """
        params = {
            "model": self.model,
            "messages": self._build_messages(text),
            "audio": self._build_audio_param(),
            "stream": self.streaming,
        }
        return params

    async def text_to_speak(self, text, output_file):
        """将文本转换为语音

        使用小米 MiMo-V2.5-TTS 的 /chat/completions 接口。
        支持非流式和流式两种模式（流式目前为兼容模式，实际一次性返回）。

        Args:
            text: 要转换的文本（已由上层做 markdown 清洗和替换词处理）
            output_file: 输出文件路径。
                - 为 None 时返回音频 bytes
                - 不为 None 时写入文件

        Returns:
            output_file 为 None 时返回音频 bytes；否则不返回值

        Raises:
            ValueError: 参数校验失败
            Exception: API 调用失败
        """
        if not text:
            logger.bind(tag=TAG).warning("输入文本为空，跳过语音合成")
            return b"" if output_file is None else None

        if not self.api_key:
            raise ValueError(
                f"{TAG}: api_key 未配置，请前往 https://platform.xiaomimimo.com/console/api-keys 获取"
            )

        # 预置音色模式下必须指定 voice
        if self.model == "mimo-v2.5-tts" and not self.voice:
            raise ValueError(
                f"{TAG}: 预置音色模式下 voice 参数为必填项。"
                f"可用音色: {', '.join(self.PRESET_VOICES.keys())}"
            )

        # 清理文本
        clean_text = self._sanitize_text(text)
        if not clean_text:
            logger.bind(tag=TAG).warning("清理后文本为空，跳过语音合成")
            return b"" if output_file is None else None

        # 构建请求参数
        params = self._build_request_params(clean_text)
        self._request_count += 1
        req_id = self._request_count

        logger.bind(tag=TAG).info(
            f"[请求#{req_id}] 开始语音合成 | text_len={len(clean_text)} | "
            f"model={self.model} | voice={self.voice} | format={self.response_format} | "
            f"streaming={self.streaming}"
        )

        start_time = time.time()

        try:
            if self.streaming:
                audio_bytes = await self._speak_streaming(params, req_id, start_time)
            else:
                audio_bytes = await self._speak_non_streaming(params, req_id, start_time)

            elapsed_ms = (time.time() - start_time) * 1000

            if audio_bytes:
                logger.bind(tag=TAG).info(
                    f"[请求#{req_id}] 语音合成完成 | 耗时={elapsed_ms:.1f}ms | "
                    f"音频大小={len(audio_bytes)} bytes"
                )

                if output_file:
                    with open(output_file, "wb") as f:
                        f.write(audio_bytes)
                    logger.bind(tag=TAG).debug(
                        f"[请求#{req_id}] 音频已写入文件: {output_file}"
                    )
                else:
                    return audio_bytes
            else:
                logger.bind(tag=TAG).warning(
                    f"[请求#{req_id}] 语音合成返回空数据"
                )
                return b"" if output_file is None else None

        except OpenAIAuthError as e:
            logger.bind(tag=TAG).error(
                f"[请求#{req_id}] API 认证失败，请检查 api_key 是否正确: {e}"
            )
            raise Exception(f"{TAG} 认证失败: {e}")

        except OpenAIConnectionError as e:
            logger.bind(tag=TAG).error(
                f"[请求#{req_id}] API 连接失败，请检查网络或 base_url: {e}"
            )
            raise Exception(f"{TAG} 连接失败: {e}")

        except OpenAIRateLimitError as e:
            logger.bind(tag=TAG).error(
                f"[请求#{req_id}] API 请求频率超限: {e}"
            )
            raise Exception(f"{TAG} 请求频率超限: {e}")

        except OpenAIAPIError as e:
            logger.bind(tag=TAG).error(
                f"[请求#{req_id}] API 调用失败: status={e.status_code} "
                f"message={getattr(e, 'message', str(e))}"
            )
            raise Exception(f"{TAG} API错误 [{e.status_code}]: {e}")

        except Exception as e:
            logger.bind(tag=TAG).error(
                f"[请求#{req_id}] 语音合成异常: {e}", exc_info=True
            )
            raise Exception(f"{TAG} error: {e}")

    async def _speak_non_streaming(self, params: dict, req_id: int, start_time: float) -> bytes:
        """非流式调用：一次性获取完整音频

        Args:
            params: API 请求参数
            req_id: 请求编号
            start_time: 开始时间戳

        Returns:
            音频 bytes
        """
        response = await self._client.chat.completions.create(**params)

        # 从响应中提取音频数据
        message = response.choices[0].message
        audio_data = getattr(message, "audio", None)

        if audio_data and hasattr(audio_data, "data"):
            audio_bytes = base64.b64decode(audio_data.data)
            ttfb_ms = (time.time() - start_time) * 1000
            logger.bind(tag=TAG).debug(
                f"[请求#{req_id}] 非流式响应 | 音频大小={len(audio_bytes)} bytes | "
                f"TTFB={ttfb_ms:.1f}ms"
            )
            return audio_bytes
        else:
            logger.bind(tag=TAG).warning(
                f"[请求#{req_id}] 响应中未找到音频数据，message={message}"
            )
            return b""

    async def _speak_streaming(self, params: dict, req_id: int, start_time: float) -> bytes:
        """流式调用：逐块接收音频数据

        注意: MiMo-V2.5-TTS 的低延迟流式输出功能暂未上线，
        流式调用接口目前降级为兼容模式，仅在所有推理完成后
        以流式格式返回一次结果。

        Args:
            params: API 请求参数
            req_id: 请求编号
            start_time: 开始时间戳

        Returns:
            合并后的音频 bytes
        """
        audio_chunks = []
        first_chunk_time = None

        stream = await self._client.chat.completions.create(**params)

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            audio = getattr(delta, "audio", None)

            if audio is not None and isinstance(audio, dict) and "data" in audio:
                pcm_bytes = base64.b64decode(audio["data"])
                audio_chunks.append(pcm_bytes)

                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    ttfb_ms = (first_chunk_time - start_time) * 1000
                    logger.bind(tag=TAG).debug(
                        f"[请求#{req_id}] 流式首包 | TTFB={ttfb_ms:.1f}ms | "
                        f"chunk_size={len(pcm_bytes)} bytes"
                    )

        total_bytes = sum(len(c) for c in audio_chunks)
        logger.bind(tag=TAG).debug(
            f"[请求#{req_id}] 流式传输完成 | 总数据={total_bytes} bytes | "
            f"chunks={len(audio_chunks)}"
        )

        return b"".join(audio_chunks)