import time
import json
import uuid
import random
import asyncio
import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler
from core.utils.dialogue import Message
from core.utils.util import audio_to_data
from core.providers.tts.dto.dto import SentenceType
from core.utils.wakeup_word import WakeupWordsConfig
from core.handle.sendAudioHandle import sendAudioMessage, send_tts_message
from core.utils.util import remove_punctuation_and_length, opus_datas_to_wav_bytes
from core.utils.tts import create_instance
from core.providers.tools.device_mcp import MCPClient, send_mcp_initialize_message

TAG = __name__

WAKEUP_CONFIG = {
    "refresh_time": 10,
    "responses": [
        "我一直都在呢，您请说。",
        "在的呢，请随时吩咐我。",
        "来啦来啦，请告诉我吧。",
        "您请说，我正听着。",
        "请您讲话，我准备好了。",
        "请您说出指令吧。",
        "我认真听着呢，请讲。",
        "请问您需要什么帮助？",
        "我在这里，等候您的指令。",
    ],
}

# 创建全局的唤醒词配置管理器
wakeup_words_config = WakeupWordsConfig()

# 用于防止并发调用wakeupWordsResponse的锁
_wakeup_response_lock = asyncio.Lock()


def _get_wakeup_cache_voice(conn: "ConnectionHandler"):
    """获取唤醒词缓存 key。
    注意：此 key 表示"唤醒词目标缓存槽"，不代表实际生成 provider。
    当 mimo_v25 生成失败时，回退音频也会写入同一 key 下。
    """
    mimo_config = conn.config.get("TTS", {}).get("mimo_v25")
    if mimo_config:
        model = mimo_config.get("model", "mimo-v2.5-tts")
        voice = mimo_config.get("voice", "default")
        fmt = mimo_config.get("response_format", "wav")
        style = mimo_config.get("style_instruction", "")
        style_hash = hashlib.md5(style.encode()).hexdigest()[:6]
        clone_file = mimo_config.get("clone_audio_file", "")
        clone_hash = hashlib.md5(clone_file.encode()).hexdigest()[:6] if clone_file else "none"
        opt_preview = "1" if mimo_config.get("optimize_text_preview", False) else "0"
        return f"mimo_v25_{model}_{voice}_{fmt}_{style_hash}_{clone_hash}_{opt_preview}"
    voice = getattr(conn.tts, "voice", "default")
    return voice or "default"


def _save_wakeup_response(wav_bytes, voice_key, text):
    file_path = wakeup_words_config.generate_file_path(voice_key)
    with open(file_path, "wb") as f:
        f.write(wav_bytes)
    wakeup_words_config.update_wakeup_response(voice_key, file_path, text)


async def handleHelloMessage(conn: "ConnectionHandler", msg_json):
    """处理hello消息"""
    audio_params = msg_json.get("audio_params")
    if audio_params:
        format = audio_params.get("format")
        conn.logger.bind(tag=TAG).debug(f"客户端音频格式: {format}")
        conn.audio_format = format
        conn.welcome_msg["audio_params"] = audio_params
    features = msg_json.get("features")
    if features:
        conn.logger.bind(tag=TAG).debug(f"客户端特性: {features}")
        conn.features = features
        if features.get("mcp"):
            conn.logger.bind(tag=TAG).debug("客户端支持MCP")
            conn.mcp_client = MCPClient()
            # 发送初始化
            asyncio.create_task(send_mcp_initialize_message(conn))

    await conn.websocket.send(json.dumps(conn.welcome_msg))


async def checkWakeupWords(conn: "ConnectionHandler", text):
    enable_wakeup_words_response_cache = conn.config[
        "enable_wakeup_words_response_cache"
    ]

    # 等待tts初始化，最多等待3秒
    start_time = time.time()
    while time.time() - start_time < 3:
        if conn.tts:
            break
        await asyncio.sleep(0.1)
    else:
        return False

    if not enable_wakeup_words_response_cache:
        return False

    _, filtered_text = remove_punctuation_and_length(text)
    if filtered_text not in conn.config.get("wakeup_words"):
        return False

    conn.just_woken_up = True
    await send_tts_message(conn, "start")

    # 获取唤醒词缓存 key
    voice_key = _get_wakeup_cache_voice(conn)
    conn.logger.bind(tag=TAG).info(f"checkWakeupWords voice_key={voice_key}")

    # 获取唤醒词回复配置
    response = wakeup_words_config.get_wakeup_response(voice_key)
    conn.logger.bind(tag=TAG).info(
        f"checkWakeupWords response={'命中' if response else '未命中'}"
    )
    if not response or not response.get("file_path"):
        response = {
            "voice": "default",
            "file_path": "config/assets/wakeup_words_short.wav",
            "time": 0,
            "text": "我在这里哦！",
        }

    # 获取音频数据
    opus_packets = await audio_to_data(response.get("file_path"), use_cache=False)
    # 播放唤醒词回复
    conn.client_abort = False

    # 将唤醒词回复视为新会话，生成新的 sentence_id，确保流控器重置
    conn.sentence_id = str(uuid.uuid4().hex)

    conn.logger.bind(tag=TAG).info(f"播放唤醒词回复: {response.get('text')}")
    await sendAudioMessage(conn, SentenceType.FIRST, opus_packets, response.get("text"))
    await sendAudioMessage(conn, SentenceType.LAST, [], None)

    # 补充对话
    conn.dialogue.put(Message(role="assistant", content=response.get("text")))

    # 检查是否需要更新唤醒词回复
    if time.time() - response.get("time", 0) > WAKEUP_CONFIG["refresh_time"]:
        if not _wakeup_response_lock.locked():
            async def _safe_wakeup_response():
                try:
                    await wakeupWordsResponse(conn)
                except Exception as e:
                    conn.logger.bind(tag=TAG).exception(
                        f"wakeupWordsResponse 执行失败: {e}"
                    )
            asyncio.create_task(_safe_wakeup_response())
    return True


async def wakeupWordsResponse(conn: "ConnectionHandler"):
    if not conn.tts:
        conn.logger.bind(tag=TAG).debug("wakeupWordsResponse: conn.tts 为 None，跳过")
        return

    try:
        # 尝试获取锁，如果获取不到就返回
        if not await _wakeup_response_lock.acquire():
            conn.logger.bind(tag=TAG).debug("wakeupWordsResponse: 获取锁失败，跳过")
            return

        # 从预定义回复列表中随机选择一个回复
        result = random.choice(WAKEUP_CONFIG["responses"])
        if not result or len(result) == 0:
            return

        voice_key = _get_wakeup_cache_voice(conn)
        conn.logger.bind(tag=TAG).info(
            f"开始生成唤醒词回复: voice_key={voice_key}, text={result}"
        )

        # 优先使用 mimo_v25 生成唤醒词回复
        mimo_config = conn.config.get("TTS", {}).get("mimo_v25")
        if mimo_config:
            try:
                mimo_tts = create_instance("mimo_v25", mimo_config, delete_audio_file=True)
                mimo_tts.conn = conn
                tts_result = await asyncio.to_thread(mimo_tts.to_tts, result)
                if not tts_result:
                    conn.logger.bind(tag=TAG).warning(
                        "mimo_v25 生成唤醒词回复返回空，回退到当前 TTS"
                    )
                else:
                    wav_bytes = opus_datas_to_wav_bytes(
                        tts_result, sample_rate=conn.sample_rate
                    )
                    _save_wakeup_response(wav_bytes, voice_key, result)
                    conn.logger.bind(tag=TAG).info(
                        f"mimo_v25 唤醒词回复生成成功: voice_key={voice_key}"
                    )
                    await mimo_tts.close()
                    return
            except Exception as e:
                conn.logger.bind(tag=TAG).warning(
                    f"mimo_v25 生成唤醒词回复失败，回退到当前 TTS: {e}"
                )

        # 回退到当前 TTS
        tts_result = await asyncio.to_thread(conn.tts.to_tts, result)
        if not tts_result:
            conn.logger.bind(tag=TAG).warning("当前 TTS 生成唤醒词回复返回空")
            return
        wav_bytes = opus_datas_to_wav_bytes(tts_result, sample_rate=conn.sample_rate)
        _save_wakeup_response(wav_bytes, voice_key, result)
        conn.logger.bind(tag=TAG).info(
            f"当前 TTS 唤醒词回复生成成功: voice_key={voice_key}"
        )
    finally:
        # 确保在任何情况下都释放锁
        if _wakeup_response_lock.locked():
            _wakeup_response_lock.release()
