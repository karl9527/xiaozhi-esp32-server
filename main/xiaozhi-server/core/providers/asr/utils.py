import re
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

EMOTION_EMOJI_MAP = {
    "HAPPY": "🙂",
    "SAD": "😔",
    "ANGRY": "😡",
    "NEUTRAL": "😶",
    "FEARFUL": "😰",
    "DISGUSTED": "🤢",
    "SURPRISED": "😲",
    "EMO_UNKNOWN": "😶",  # 未知情绪默认用中性表情
}
# EVENT_EMOJI_MAP = {
#     "<|BGM|>": "🎼",
#     "<|Speech|>": "",
#     "<|Applause|>": "👏",
#     "<|Laughter|>": "😀",
#     "<|Cry|>": "😭",
#     "<|Sneeze|>": "🤧",
#     "<|Breath|>": "",
#     "<|Cough|>": "🤧",
# }

def lang_tag_filter(text: str) -> dict | str:
    """
    解析 FunASR 识别结果，按顺序提取标签和纯文本内容

    Args:
        text: ASR 识别的原始文本，可能包含多种标签

    Returns:
        dict: {"language": "zh", "emotion": "SAD", "emoji": "😔", "content": "你好"} 如果有标签
        str: 纯文本，如果没有标签

    Examples:
        FunASR 输出格式：<|语种|><|情绪|><|事件|><|其他选项|>原文
        >>> lang_tag_filter("<|zh|><|SAD|><|Speech|><|withitn|>你好啊，测试测试。")
        {"language": "zh", "emotion": "SAD", "emoji": "😔", "content": "你好啊，测试测试。"}
        >>> lang_tag_filter("<|en|><|HAPPY|><|Speech|><|withitn|>Hello hello.")
        {"language": "en", "emotion": "HAPPY", "emoji": "🙂", "content": "Hello hello."}
        >>> lang_tag_filter("plain text")
        "plain text"
    """
    # 提取所有标签（按顺序）
    tag_pattern = r"<\|([^|]+)\|>"
    all_tags = re.findall(tag_pattern, text)

    # 移除所有 <|...|> 格式的标签，获取纯文本
    clean_text = re.sub(tag_pattern, "", text).strip()

    # 过滤掉模型内部特殊 token（如 SPECIAL_TOKEN_19、unk 等）
    valid_tags = [
        tag for tag in all_tags
        if not tag.startswith("SPECIAL_TOKEN") and not tag.startswith("unk")
    ]

    # 如果没有有效标签，直接返回纯文本
    if not valid_tags:
        return clean_text

    # 按照 FunASR 的固定顺序提取标签，返回 dict
    language = valid_tags[0] if len(valid_tags) > 0 else "zh"
    emotion = valid_tags[1] if len(valid_tags) > 1 else "NEUTRAL"
    # event = valid_tags[2] if len(valid_tags) > 2 else "Speech"  # 事件标签暂不使用

    result = {
        "content": clean_text,
        "language": language,
        "emotion": emotion,
        # "event": event,
    }

    # 添加 emoji 映射
    if emotion in EMOTION_EMOJI_MAP:
        result["emotion"] = EMOTION_EMOJI_MAP[emotion]
    # 事件标签暂不使用
    # if event in EVENT_EMOJI_MAP:
    #     result["event"] = EVENT_EMOJI_MAP[event]

    return result

