"""
Kimi k2.6 LLM Provider
- API文档: https://platform.moonshot.cn/
- 兼容OpenAI API格式，base_url: https://api.moonshot.cn/v1
- 默认模型: kimi-k2.6
"""

import httpx
import openai
from openai.types import CompletionUsage
from config.logger import setup_logging
from core.utils.util import check_model_key
from core.providers.llm.base import LLMProviderBase

TAG = __name__
logger = setup_logging()

# Kimi (moonshot.cn) 思考模式禁用参数
# THINKING_DISABLED_PARAMS = {"thinking": {"type": "disabled"}}

# Kimi API 默认配置
DEFAULT_BASE_URL = "https://api.foxming.com/v1"
DEFAULT_MODEL_NAME = "kimi-k2.6"


class LLMProvider(LLMProviderBase):
    """
    Kimi k2.6 LLM 提供商
    基于OpenAI兼容接口实现，支持流式响应和function calling
    """

    def __init__(self, config):
        """
        初始化Kimi LLM Provider

        Args:
            config: 配置字典，支持以下字段：
                - model_name: 模型名称，默认 "kimi-k2.6"
                - api_key: Moonshot API密钥
                - base_url / url: API基础地址，默认 "https://api.moonshot.cn/v1"
                - timeout: 超时配置（字典或数值）
                - max_tokens: 最大生成token数
                - temperature: 采样温度
                - top_p: 核采样参数
                - frequency_penalty: 频率惩罚
        """
        self.model_name = config.get("model_name", DEFAULT_MODEL_NAME)
        self.api_key = config.get("api_key")

        # 支持 base_url 或 url 配置（兼容两种写法）
        if "base_url" in config:
            self.base_url = config.get("base_url")
        elif "url" in config:
            self.base_url = config.get("url")
        else:
            self.base_url = DEFAULT_BASE_URL

        # 超时配置处理（支持字典格式或数值格式）
        timeout_config = config.get("timeout")
        if isinstance(timeout_config, dict):
            custom_timeout = httpx.Timeout(
                pool=timeout_config.get("pool", 2.0),
                connect=timeout_config.get("connect", 3.0),
                write=timeout_config.get("write", 5.0),
                read=timeout_config.get("read", 60.0)
            )
        elif isinstance(timeout_config, (int, float)) and timeout_config > 0:
            custom_timeout = httpx.Timeout(timeout_config)
        else:
            # 默认5分钟超时
            custom_timeout = httpx.Timeout(300)

        # 可选参数处理（max_tokens, temperature, top_p, frequency_penalty）
        param_defaults = {
            "max_tokens": int,
            "temperature": lambda x: round(float(x), 1),
            "top_p": lambda x: round(float(x), 1),
            "frequency_penalty": lambda x: round(float(x), 1),
        }

        for param, converter in param_defaults.items():
            value = config.get(param)
            try:
                setattr(self, param, converter(value) if value not in (None, "") else None)
            except (ValueError, TypeError):
                logger.bind(tag=TAG).warning(f"参数 {param} 值 '{value}' 无效，将使用默认值")
                setattr(self, param, None)

        # API密钥有效性检查
        model_key_msg = check_model_key("LLM", self.api_key)
        if model_key_msg:
            logger.bind(tag=TAG).error(model_key_msg)

        # 创建OpenAI兼容客户端
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=custom_timeout
        )

        logger.bind(tag=TAG).info(
            f"Kimi LLM Provider 初始化完成，模型: {self.model_name}，"
            f"base_url: {self.base_url}"
        )

    @staticmethod
    def normalize_dialogue(dialogue):
        """
        自动修复 dialogue 中缺失 content 的消息

        Args:
            dialogue: 对话消息列表

        Returns:
            修复后的对话消息列表
        """
        for msg in dialogue:
            if "role" in msg and "content" not in msg:
                msg["content"] = ""
        return dialogue

    def _apply_thinking_disabled(self, request_params: dict):
        """
        Kimi API 默认禁用思考模式
        通过 extra_body 参数传递 thinking 配置

        Args:
            request_params: 请求参数字典（会被原地修改）
        """
        request_params.setdefault("extra_body", {}).update(THINKING_DISABLED_PARAMS)
        logger.bind(tag=TAG).debug("已为 Kimi API 禁用思考模式")

    def _build_request_params(self, dialogue, **kwargs):
        """
        构建OpenAI兼容的请求参数

        Args:
            dialogue: 对话消息列表
            **kwargs: 可选的覆盖参数

        Returns:
            请求参数字典
        """
        dialogue = self.normalize_dialogue(dialogue)

        request_params = {
            "model": self.model_name,
            "messages": dialogue,
            "stream": True,
        }

        optional_params = {
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
        }

        for key, value in optional_params.items():
            if value is not None:
                request_params[key] = value

        self._apply_thinking_disabled(request_params)
        return request_params

    def response(self, session_id, dialogue, **kwargs):
        """
        流式响应生成器

        Args:
            session_id: 会话ID
            dialogue: 对话消息列表
            **kwargs: 可选参数覆盖（max_tokens, temperature, top_p, frequency_penalty）

        Yields:
            文本片段（生成器）
        """
        request_params = self._build_request_params(dialogue, **kwargs)

        try:
            responses = self.client.chat.completions.create(**request_params)
        except openai.AuthenticationError as e:
            logger.bind(tag=TAG).error(f"Kimi API 认证失败，请检查API密钥: {e}")
            yield "[Kimi API 认证失败，请检查API密钥]"
            return
        except openai.APIConnectionError as e:
            logger.bind(tag=TAG).error(f"Kimi API 连接失败: {e}")
            yield "[Kimi API 连接失败，请检查网络]"
            return
        except openai.RateLimitError as e:
            logger.bind(tag=TAG).error(f"Kimi API 请求频率超限: {e}")
            yield "[Kimi API 请求频率超限，请稍后再试]"
            return
        except openai.APIError as e:
            logger.bind(tag=TAG).error(f"Kimi API 请求失败: {e}")
            yield f"[Kimi API 请求失败: {e}]"
            return
        except Exception as e:
            logger.bind(tag=TAG).error(f"Kimi LLM 未知错误: {e}")
            yield f"[Kimi LLM 错误: {e}]"
            return

        is_active = True
        try:
            for chunk in responses:
                try:
                    delta = chunk.choices[0].delta if getattr(chunk, "choices", None) else None
                    content = getattr(delta, "content", "") if delta else ""
                except IndexError:
                    content = ""

                if content:
                    # 处理 <think> 标签：跳过思考过程内容
                    if "<think>" in content:
                        is_active = False
                        content = content.split("<think>")[0]
                    if "</think>" in content:
                        is_active = True
                        content = content.split("</think>")[-1]
                    if is_active:
                        yield content
        finally:
            responses.close()

    def response_with_functions(self, session_id, dialogue, functions=None, **kwargs):
        """
        支持function calling的流式响应

        Args:
            session_id: 会话ID
            dialogue: 对话消息列表
            functions: 工具/函数定义列表
            **kwargs: 可选参数覆盖

        Yields:
            (content, tool_calls) 元组
        """
        dialogue = self.normalize_dialogue(dialogue)

        request_params = {
            "model": self.model_name,
            "messages": dialogue,
            "stream": True,
            "tools": functions,
        }

        optional_params = {
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
        }

        for key, value in optional_params.items():
            if value is not None:
                request_params[key] = value

        self._apply_thinking_disabled(request_params)

        try:
            stream = self.client.chat.completions.create(**request_params)
        except openai.AuthenticationError as e:
            logger.bind(tag=TAG).error(f"Kimi API 认证失败: {e}")
            yield "[Kimi API 认证失败]", None
            return
        except openai.APIConnectionError as e:
            logger.bind(tag=TAG).error(f"Kimi API 连接失败: {e}")
            yield "[Kimi API 连接失败]", None
            return
        except openai.RateLimitError as e:
            logger.bind(tag=TAG).error(f"Kimi API 请求频率超限: {e}")
            yield "[Kimi API 请求频率超限]", None
            return
        except openai.APIError as e:
            logger.bind(tag=TAG).error(f"Kimi API 请求失败: {e}")
            yield f"[Kimi API 请求失败: {e}]", None
            return
        except Exception as e:
            logger.bind(tag=TAG).error(f"Kimi LLM function calling 未知错误: {e}")
            yield f"[Kimi LLM 错误: {e}]", None
            return

        try:
            for chunk in stream:
                if getattr(chunk, "choices", None):
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", "")
                    tool_calls = getattr(delta, "tool_calls", None)
                    yield content, tool_calls
                elif isinstance(getattr(chunk, "usage", None), CompletionUsage):
                    usage_info = getattr(chunk, "usage", None)
                    logger.bind(tag=TAG).info(
                        f"Token 消耗：输入 {getattr(usage_info, 'prompt_tokens', '未知')}，"
                        f"输出 {getattr(usage_info, 'completion_tokens', '未知')}，"
                        f"共计 {getattr(usage_info, 'total_tokens', '未知')}"
                    )
        finally:
            stream.close()