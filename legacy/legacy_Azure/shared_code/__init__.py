# shared_code/__init__.py

from .config import initialize_config
from .chat_service import ChatService

# 외부에서 import 할 수 있는 것들을 명시적으로 선언
__all__ = ['initialize_config', 'ChatService']