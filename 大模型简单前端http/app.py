# app.py - HTML流式传输Flask对话应用
from flask import Flask, render_template, request, Response
from langchain_deepseek import ChatDeepSeek
import json

# 创建Flask应用
app = Flask(__name__)

# 初始化DeepSeek大模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=2048,  # 增加token限制以支持更丰富的HTML输出
    timeout=30.0,
    max_retries=2,
    api_key="sk-157abd02156e4718b1132b3ed03fd5ce"
)

# HTML格式提示词模板
HTML_SYSTEM_PROMPT = """你是一个专业的AI助手，你的回复需要使用结构化的HTML格式。

重要规则：
1. 你的回复必须是有效的HTML片段，不需要完整的HTML文档结构
2. 直接输出HTML内容，不要用代码块包裹
3. 使用适当的HTML标签来组织内容结构

可用的HTML标签和建议用法：
- <p>用于段落文本</p>
- <h1>, <h2>, <h3>用于标题层级
- <strong>用于重要内容强调</strong>
- <em>用于斜体强调</em>
- <ul><li>用于无序列表</li></ul>
- <ol><li>用于有序列表</li></ol>
- <blockquote>用于引用内容</blockquote>
- <code>用于行内代码</code>
- <pre><code>用于代码块</code></pre>
- <table>用于表格数据</table>

示例回复格式：
<h2>关于人工智能</h2>
<p>人工智能是一个快速发展的领域，主要包括以下几个方面：</p>
<ul>
<li><strong>机器学习</strong>：让机器从数据中学习</li>
<li><strong>深度学习</strong>：基于神经网络的学习方法</li>
<li><strong>自然语言处理</strong>：让机器理解和生成人类语言</li>
</ul>
<p>如果你想了解更多，我可以为你详细解释任何一个方面。</p>

请始终遵循这个格式，让你的回复既专业又易读。"""


@app.route('/')
def home():
    """主页 - 对话界面"""
    return render_template('chat.html')


@app.route('/stream_chat', methods=['POST'])
def stream_chat():
    """HTML流式对话API"""
    try:
        # 获取用户消息
        user_message = request.form.get('message', '').strip()

        print("=" * 50)
        print(f"🌊 收到HTML流式对话请求: '{user_message}'")

        if not user_message:
            return Response(
                f"data: {json.dumps({'type': 'error', 'content': '<p style=\"color: red;\">消息不能为空</p>'})}\n\n",
                mimetype='text/event-stream'
            )

        # 构建消息 - 使用HTML格式的系统提示词
        messages = [
            ("system", HTML_SYSTEM_PROMPT),
            ("human", user_message)
        ]

        print(f"🔍 开始HTML流式生成...")

        def generate():
            """HTML流式生成函数"""
            try:
                full_content = ""  # 保存完整HTML内容

                # 发送开始信号
                yield f"data: {json.dumps({'type': 'start', 'content': ''})}\n\n"

                for chunk in llm.stream(messages):
                    # 获取chunk内容
                    chunk_content = chunk.content if hasattr(chunk, 'content') else ""

                    # 累积完整内容
                    full_content += chunk_content

                    # 检查是否结束（通过metadata判断）
                    is_finished = False
                    if hasattr(chunk, 'response_metadata') and chunk.response_metadata:
                        if 'finish_reason' in chunk.response_metadata:
                            is_finished = True

                    # 如果内容为空且有结束标志，说明真正结束了
                    if chunk_content == "" and is_finished:
                        print(f"✅ HTML流式生成完成: {full_content}")

                        # 发送结束信号
                        yield f"data: {json.dumps({'type': 'end', 'content': '', 'full_content': full_content})}\n\n"
                        break

                    # 如果有HTML内容，发送这个chunk
                    if chunk_content:
                        # 确保JSON序列化安全 - 转义HTML中的特殊字符
                        safe_content = chunk_content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk_content})}\n\n"

            except Exception as e:
                print(f"❌ HTML流式生成错误: {e}")
                error_html = f'<p style="color: red;">HTML流式生成错误: {str(e)}</p>'
                yield f"data: {json.dumps({'type': 'error', 'content': error_html})}\n\n"

        # 返回流式响应
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',  # 禁用nginx缓冲
                'Access-Control-Allow-Origin': '*',  # CORS支持
                'Access-Control-Allow-Methods': 'POST',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )

    except Exception as e:
        print(f"❌ HTML流式对话错误: {e}")
        error_html = f'<p style="color: red;">HTML流式对话错误: {str(e)}</p>'
        return Response(
            f"data: {json.dumps({'type': 'error', 'content': error_html})}\n\n",
            mimetype='text/event-stream'
        )


if __name__ == '__main__':
    print("🚀 Flask HTML流式对话应用启动...")
    print("📍 访问地址: http://127.0.0.1:5000")
    print("🌊 HTML流式传输模式")
    print("📁 HTML模板路径: templates/chat.html")
    print("🎨 支持丰富的HTML格式输出")
    print("=" * 50)

    app.run(debug=True)# app.py - 流式传输Flask对话应用
from flask import Flask, render_template, request, Response
from langchain_deepseek import ChatDeepSeek
import json

# 创建Flask应用
app = Flask(__name__)

# 初始化DeepSeek大模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=1024,
    timeout=30.0,
    max_retries=2,
    api_key="sk-157abd02156e4718b1132b3ed03fd5ce"
)


@app.route('/')
def home():
    """主页 - 对话界面"""
    return render_template('chat.html')


@app.route('/stream_chat', methods=['POST'])
def stream_chat():
    """流式对话API"""
    try:
        # 获取用户消息
        user_message = request.form.get('message', '').strip()

        print("=" * 50)
        print(f"🌊 收到流式对话请求: '{user_message}'")

        if not user_message:
            return Response(
                f"data: {json.dumps({'type': 'error', 'content': '消息不能为空'})}\n\n",
                mimetype='text/event-stream'
            )

        # 构建消息
        messages = [
            ("system", "你是一个AI对话助手，请用友善、有帮助的语气回复用户。"),
            ("human", user_message)
        ]

        print(f"🔍 开始流式生成...")

        def generate():
            """流式生成函数"""
            try:
                full_content = ""  # 保存完整内容

                # 发送开始信号
                yield f"data: {json.dumps({'type': 'start', 'content': ''})}\n\n"

                for chunk in llm.stream(messages):
                    # 获取chunk内容
                    chunk_content = chunk.content if hasattr(chunk, 'content') else ""

                    print(f"🔍 收到chunk: '{chunk_content}'")

                    # 累积完整内容
                    full_content += chunk_content

                    # 检查是否结束（通过metadata判断）
                    is_finished = False
                    if hasattr(chunk, 'response_metadata') and chunk.response_metadata:
                        if 'finish_reason' in chunk.response_metadata:
                            is_finished = True
                            print(f"🔍 检测到结束信号: {chunk.response_metadata}")

                    # 如果内容为空且有结束标志，说明真正结束了
                    if chunk_content == "" and is_finished:
                        print(f"✅ 流式生成完成，总长度: {len(full_content)}")

                        # 发送结束信号
                        yield f"data: {json.dumps({'type': 'end', 'content': '', 'full_content': full_content})}\n\n"
                        break

                    # 如果有内容，发送这个chunk
                    if chunk_content:
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk_content})}\n\n"

            except Exception as e:
                print(f"❌ 流式生成错误: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': f'流式生成错误: {str(e)}'})}\n\n"

        # 返回流式响应
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'  # 禁用nginx缓冲
            }
        )

    except Exception as e:
        print(f"❌ 流式对话错误: {e}")
        return Response(
            f"data: {json.dumps({'type': 'error', 'content': f'出错了: {str(e)}'})}\n\n",
            mimetype='text/event-stream'
        )


if __name__ == '__main__':
    print("🚀 Flask流式对话应用启动...")
    print("📍 访问地址: http://127.0.0.1:5000")
    print("🌊 纯流式传输模式")
    print("📁 HTML模板路径: templates/chat.html")
    print("=" * 50)

    app.run(debug=True)