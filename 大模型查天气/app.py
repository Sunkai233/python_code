# app.py - Flask Web应用
from flask import Flask, render_template, request, Response, jsonify
from agent import create_weather_agent, format_stream_data
import json

# 创建Flask应用
app = Flask(__name__)

# 配置参数
CONFIG = {
    'API_KEY': 'sk-157abd02156e4718b1132b3ed03fd5ce',
    'DATABASE_URI': 'mysql+pymysql://root:Sunkai12@localhost:3306/langgraph_agent?charset=utf8mb4'
}

# 初始化天气智能体
agent = create_weather_agent(
    api_key=CONFIG['API_KEY'],
    database_uri=CONFIG['DATABASE_URI']
)


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
        print(f"🌊 Flask收到天气智能体流式对话请求: '{user_message}'")

        if not user_message:
            error_data = {
                'type': 'error',
                'content': '<p style="color: red;">消息不能为空</p>'
            }
            return Response(
                format_stream_data(error_data),
                mimetype='text/event-stream'
            )

        print(f"🔍 开始调用天气智能体...")

        def generate():
            """流式生成函数"""
            try:
                # 调用天气智能体的流式聊天
                for data in agent.chat_stream(user_message):
                    yield format_stream_data(data)

            except Exception as e:
                print(f"❌ Flask流式生成错误: {e}")
                error_data = {
                    'type': 'error',
                    'content': f'<p style="color: red;">Flask流式生成错误: {str(e)}</p>'
                }
                yield format_stream_data(error_data)

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
        print(f"❌ Flask流式对话错误: {e}")
        error_data = {
            'type': 'error',
            'content': f'<p style="color: red;">Flask流式对话错误: {str(e)}</p>'
        }
        return Response(
            format_stream_data(error_data),
            mimetype='text/event-stream'
        )


@app.route('/simple_chat', methods=['POST'])
def simple_chat():
    """简单对话API（非流式）"""
    try:
        user_message = request.form.get('message', '').strip()

        print(f"💬 Flask收到天气智能体简单对话请求: '{user_message}'")

        if not user_message:
            return jsonify({'success': False, 'message': '消息不能为空'})

        # 调用天气智能体的简单聊天
        result = agent.chat_simple(user_message)

        # 处理新的数据结构
        if isinstance(result, dict):
            return jsonify({
                'success': True,
                'html_content': result['html_content'],
                'tool_info': result['tool_info']
            })
        else:
            # 兼容旧格式
            return jsonify({'success': True, 'message': result})

    except Exception as e:
        print(f"❌ Flask简单对话错误: {e}")
        return jsonify({'success': False, 'message': f'出错了: {str(e)}'})


@app.route('/agent_info')
def agent_info():
    """获取天气智能体信息"""
    try:
        info = agent.get_model_info()
        return jsonify({'success': True, 'data': info})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/weather_test', methods=['POST'])
def weather_test():
    """天气查询测试接口"""
    try:
        user_message = request.form.get('message', '帮我查一下北京的天气')

        print(f"🌤️ 天气测试请求: '{user_message}'")

        # 使用工具调用追踪模式
        agent.tool_call_tracking_output(user_message)

        return jsonify({'success': True, 'message': '天气测试完成，请查看控制台输出'})

    except Exception as e:
        print(f"❌ 天气测试错误: {e}")
        return jsonify({'success': False, 'message': f'测试失败: {str(e)}'})


@app.route('/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'app': '天气智能体HTML流式对话应用',
        'version': '2.0.0',
        'features': ['weather_query', 'database_storage', 'real_time_search', 'html_streaming']
    })


@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({
        'success': False,
        'message': '页面未找到',
        'error_code': 404
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    print(f"❌ 服务器内部错误: {error}")
    return jsonify({
        'success': False,
        'message': '服务器内部错误',
        'error_code': 500
    }), 500


@app.before_request
def log_request():
    """请求日志"""
    if request.method == 'POST':
        print(f"📥 收到 {request.method} 请求: {request.endpoint}")


@app.after_request
def log_response(response):
    """响应日志"""
    if request.method == 'POST':
        print(f"📤 响应状态: {response.status_code}")
    return response


if __name__ == '__main__':
    print("🚀 天气智能体HTML流式对话应用启动...")
    print("📍 访问地址: http://127.0.0.1:5000")
    print("🌊 HTML流式传输模式")
    print("📁 HTML模板路径: templates/chat.html")
    print("🎨 支持丰富的HTML格式输出")
    print("🌤️ 集成天气查询功能")
    print("🔧 支持工具调用追踪")
    print("\n📋 可用接口:")
    print("  • GET  /           - 主页聊天界面")
    print("  • POST /stream_chat - HTML流式对话API")
    print("  • POST /simple_chat - 简单对话API")
    print("  • POST /weather_test- 天气查询测试")
    print("  • GET  /agent_info  - 天气智能体信息")
    print("  • GET  /health      - 健康检查")
    print("=" * 50)

    app.run(debug=True, host='0.0.0.0', port=5000)