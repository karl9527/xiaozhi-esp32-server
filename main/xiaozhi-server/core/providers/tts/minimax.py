import json
import requests
from core.utils.util import check_model_key
from core.providers.tts.base import TTSProviderBase
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


class TTSProvider(TTSProviderBase):
    TTS_PARAM_CONFIG = [
        ("ttsRate", "speed", 0.5, 2, 1, lambda v: round(float(v), 2)),
    ]

    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)
        self.api_key = config.get("api_key")
        self.host = config.get("host", "api.minimaxi.com")
        self.api_url = f"https://{self.host}/v1/t2a_v2"
        self.model = config.get("model", "speech-02-turbo")
        if config.get("private_voice"):
            self.voice = config.get("private_voice")
        else:
            self.voice = config.get("voice_id", "female-shaonv")
        self.audio_file_type = config.get("format", "pcm")

        # 处理空字符串的情况
        speed = config.get("speed", "1.0")
        self.speed = float(speed) if speed else 1.0

        # 应用百分比调整（如果存在），否则使用公有化配置
        self._apply_percentage_params(config)

        self.output_file = config.get("output_dir", "tmp/")
        model_key_msg = check_model_key("TTS", self.api_key)
        if model_key_msg:
            logger.bind(tag=TAG).error(model_key_msg)

    async def text_to_speak(self, text, output_file):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "text": text,
            "stream": False,
            "output_format": "hex",
            "voice_setting": {
                "voice_id": self.voice,
                "speed": self.speed,
                "vol": 1,
                "pitch": 0,
            },
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": self.audio_file_type,
                "channel": 1,
            },
        }
        response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            resp_json = response.json()
            base_resp = resp_json.get("base_resp", {})
            if base_resp.get("status_code", 0) != 0:
                raise Exception(
                    f"MiniMax TTS业务错误: {base_resp.get('status_msg', '未知错误')}"
                )
            audio_hex = resp_json.get("data", {}).get("audio")
            if not audio_hex:
                raise Exception("MiniMax TTS返回数据缺少音频内容")
            audio_bytes = bytes.fromhex(audio_hex)
            if output_file:
                with open(output_file, "wb") as f:
                    f.write(audio_bytes)
            else:
                return audio_bytes
        else:
            raise Exception(
                f"MiniMax TTS请求失败: {response.status_code} - {response.text}"
            )
