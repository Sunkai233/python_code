"""
智能体基类，提供通用的流式输出和工具调用追踪功能
"""
import json


class BaseAgent:
    """智能体基类，提供通用的输出和交互功能"""

    def __init__(self):
        """基类初始化"""
        self.graph = None  # 子类需要设置这个属性

    def chat_stream(self, user_message):
        """流式聊天生成器

        Args:
            user_message (str): 用户消息

        Yields:
            dict: 流式数据，格式为 {'type': 'start/chunk/end/error/tool_info', 'content': '...'}
        """
        try:
            print(f"🤖 智能体开始处理: '{user_message}'")

            # 发送开始信号
            yield {'type': 'start', 'content': ''}

            full_content = ""
            tool_call_sessions = {}  # 记录每个工具调用的状态

            # 使用智能体流式处理
            for chunk in self.graph.stream({"messages": [user_message]}, stream_mode="messages"):
                message, node_name = chunk
                message_type = message.__class__.__name__

                if message_type == 'AIMessageChunk':
                    # 处理AI的HTML内容
                    if hasattr(message, 'content') and message.content:
                        content = message.content
                        full_content += content
                        # 发送HTML内容块
                        yield {'type': 'chunk', 'content': content}

                    # 处理工具调用chunks
                    if hasattr(message, 'tool_call_chunks') and message.tool_call_chunks:
                        for tool_chunk in message.tool_call_chunks:
                            chunk_id = tool_chunk.get('id')
                            chunk_name = tool_chunk.get('name')
                            chunk_args = tool_chunk.get('args', '')
                            chunk_index = tool_chunk.get('index')

                            # 新的工具调用开始
                            if chunk_id and chunk_name:
                                session_key = f"{chunk_index}_{chunk_id}"
                                tool_call_sessions[session_key] = {
                                    'name': chunk_name,
                                    'args': chunk_args,
                                    'index': chunk_index,
                                    'session_key': session_key,
                                    'is_complete': False
                                }

                                # 如果已经有完整的参数，立即发送
                                if chunk_args and self._is_complete_json(chunk_args):
                                    tool_info = f"🔧 [{chunk_index}] 开始调用 {chunk_name}: {chunk_args}"
                                    tool_call_sessions[session_key]['is_complete'] = True
                                    yield {
                                        'type': 'tool_info',
                                        'content': tool_info,
                                        'tool_session': session_key,
                                        'action': 'complete_call'
                                    }
                                # 如果没有参数或参数不完整，先不发送，等待后续chunks
                                elif not chunk_args:
                                    # 暂时不发送，等待参数构建完成
                                    pass
                                else:
                                    # 参数不完整，等待后续chunks
                                    pass

                            # 继续构建参数
                            elif chunk_args and chunk_index is not None:
                                for session_key, session_data in tool_call_sessions.items():
                                    if (session_data['index'] == chunk_index and
                                            not session_data['is_complete']):
                                        session_data['args'] += chunk_args

                                        # 检查参数是否构建完成
                                        if self._is_complete_json(session_data['args']):
                                            # 参数构建完成，发送完整的工具调用信息
                                            tool_info = f"🔧 [{chunk_index}] 开始调用 {session_data['name']}: {session_data['args']}"
                                            session_data['is_complete'] = True
                                            yield {
                                                'type': 'tool_info',
                                                'content': tool_info,
                                                'tool_session': session_key,
                                                'action': 'complete_call'
                                            }
                                        break

                elif message_type == 'ToolMessage':
                    tool_name = getattr(message, 'name', 'unknown')
                    content = getattr(message, 'content', '')

                    # 发送工具完成信息
                    tool_complete_info = f"✅ {tool_name} 执行完成"
                    if content:
                        tool_complete_info += f": {content}"

                    yield {
                        'type': 'tool_info',
                        'content': tool_complete_info,
                        'action': 'complete'
                    }

            # 发送结束信号
            yield {'type': 'end', 'content': '', 'full_content': full_content}

        except Exception as e:
            print(f"❌ 智能体生成错误: {e}")
            error_html = f'<p style="color: red;">智能体处理错误: {str(e)}</p>'
            yield {'type': 'error', 'content': error_html}

    def chat_simple(self, user_message):
        """简单聊天（非流式）

        Args:
            user_message (str): 用户消息

        Returns:
            dict: 包含HTML内容和工具调用信息的字典
        """
        try:
            print(f"🤖 智能体简单模式处理: '{user_message}'")

            html_content = ""
            tool_info = []
            tool_call_sessions = {}

            for chunk in self.graph.stream({"messages": [user_message]}, stream_mode="messages"):
                message, node_name = chunk
                message_type = message.__class__.__name__

                if message_type == 'AIMessageChunk':
                    # 收集HTML内容
                    if hasattr(message, 'content') and message.content:
                        html_content += message.content

                    # 收集工具调用信息
                    if hasattr(message, 'tool_call_chunks') and message.tool_call_chunks:
                        for tool_chunk in message.tool_call_chunks:
                            chunk_id = tool_chunk.get('id')
                            chunk_name = tool_chunk.get('name')
                            chunk_args = tool_chunk.get('args', '')
                            chunk_index = tool_chunk.get('index')

                            if chunk_id and chunk_name:
                                session_key = f"{chunk_index}_{chunk_id}"
                                tool_call_sessions[session_key] = {
                                    'name': chunk_name,
                                    'args': chunk_args,
                                    'index': chunk_index
                                }
                                tool_info.append(f"🔧 开始调用 {chunk_name}")
                            elif chunk_args and chunk_index is not None:
                                for session_key, session_data in tool_call_sessions.items():
                                    if session_data['index'] == chunk_index:
                                        session_data['args'] += chunk_args
                                        break

                elif message_type == 'ToolMessage':
                    tool_name = getattr(message, 'name', 'unknown')
                    content = getattr(message, 'content', '')
                    tool_info.append(f"✅ {tool_name} 执行完成")

            print(f"✅ 智能体简单模式完成: {len(html_content)} 字符")

            return {
                'html_content': html_content,
                'tool_info': tool_info
            }

        except Exception as e:
            print(f"❌ 智能体简单模式错误: {e}")
            return {
                'html_content': f'<p style="color: red;">智能体处理错误: {str(e)}</p>',
                'tool_info': [f"❌ 错误: {str(e)}"]
            }

    def tool_call_tracking_output(self, user_message):
        """专门追踪工具调用参数构建过程的流式输出"""
        print("\n" + "=" * 60)
        print("🔧 工具调用参数构建追踪")
        print("=" * 60)

        tool_call_sessions = {}

        for chunk in self.graph.stream({"messages": [user_message]}, stream_mode="messages"):
            message, node_name = chunk
            message_type = message.__class__.__name__

            if message_type == 'AIMessageChunk':
                # 处理AI内容
                if hasattr(message, 'content') and message.content:
                    print(message.content, end='', flush=True)

                # 处理工具调用chunks
                if hasattr(message, 'tool_call_chunks') and message.tool_call_chunks:
                    for tool_chunk in message.tool_call_chunks:
                        chunk_id = tool_chunk.get('id')
                        chunk_name = tool_chunk.get('name')
                        chunk_args = tool_chunk.get('args', '')
                        chunk_index = tool_chunk.get('index')

                        # 新的工具调用开始
                        if chunk_id and chunk_name:
                            session_key = f"{chunk_index}_{chunk_id}"
                            tool_call_sessions[session_key] = {
                                'name': chunk_name,
                                'args': chunk_args,
                                'index': chunk_index
                            }
                            print(f"\n🔧 [{chunk_index}] 开始调用 {chunk_name}:")
                            print(f"   📝 构建参数: {chunk_args}", end='', flush=True)

                        # 继续构建参数
                        elif chunk_args and chunk_index is not None:
                            for session_key, session_data in tool_call_sessions.items():
                                if session_data['index'] == chunk_index:
                                    session_data['args'] += chunk_args
                                    print(chunk_args, end='', flush=True)
                                    break

            elif message_type == 'ToolMessage':
                tool_name = getattr(message, 'name', 'unknown')
                print(f"\n✅ {tool_name} 执行完成")
                content = getattr(message, 'content', '')
                print(content)

    def _is_complete_json(self, json_str):
        """检查JSON字符串是否完整

        Args:
            json_str (str): JSON字符串

        Returns:
            bool: 是否为完整的JSON
        """
        if not json_str:
            return False

        try:
            # 尝试解析JSON
            json.loads(json_str)
            return True
        except (json.JSONDecodeError, ValueError):
            # 检查是否是简单的完整结构
            json_str = json_str.strip()
            if json_str.startswith('{') and json_str.endswith('}'):
                # 简单检查括号是否平衡
                open_braces = json_str.count('{')
                close_braces = json_str.count('}')
                if open_braces == close_braces and open_braces > 0:
                    return True
            return False

    def get_agent_info(self):
        """获取智能体信息，子类可以重写"""
        return {
            'name': self.__class__.__name__,
            'description': '通用智能体基类',
            'tools_count': len(getattr(self, 'tools', [])),
            'has_graph': self.graph is not None
        }