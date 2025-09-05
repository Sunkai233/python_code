# app.py - Flask Web应用 - 风电场智能体版本
from flask import Flask, render_template, request, Response, jsonify
from agent import WindFarmAgent
import json

# 创建Flask应用
app = Flask(__name__)

# 配置参数
CONFIG = {
    'API_KEY': 'sk-157abd02156e4718b1132b3ed03fd5ce'
}


# 初始化风电场智能体
wind_agent = WindFarmAgent(api_key=CONFIG['API_KEY'])


def format_stream_data(data):
    """格式化流式数据为Server-Sent Events格式"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.route('/')
def home():
    """主页 - 风电场分析界面"""
    return render_template('chat.html')


@app.route('/stream_chat', methods=['POST'])
def stream_chat():
    """HTML流式对话API"""
    try:
        # 获取用户消息
        user_message = request.form.get('message', '').strip()

        print("=" * 50)
        print(f"🌪️ Flask收到风电场智能体流式对话请求: '{user_message}'")

        if not user_message:
            error_data = {
                'type': 'error',
                'content': '<p style="color: red;">消息不能为空</p>'
            }
            return Response(
                format_stream_data(error_data),
                mimetype='text/event-stream'
            )

        print(f"🔍 开始调用风电场智能体...")

        def generate():
            """流式生成函数"""
            try:
                # 调用风电场智能体的流式聊天
                for data in wind_agent.chat_stream(user_message):
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

        print(f"💬 Flask收到风电场智能体简单对话请求: '{user_message}'")

        if not user_message:
            return jsonify({'success': False, 'message': '消息不能为空'})

        # 调用风电场智能体的简单聊天
        result = wind_agent.chat_simple(user_message)

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
    """获取风电场智能体信息"""
    try:
        info = wind_agent.get_agent_info()
        return jsonify({'success': True, 'data': info})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/simulation_test', methods=['POST'])
def simulation_test():
    """风电场仿真测试接口"""
    try:
        user_message = request.form.get('message', '请运行一个标准的风电场仿真计算，使用默认的Horns Rev 1风电场布局')

        print(f"🌪️ 仿真测试请求: '{user_message}'")

        # 使用工具调用追踪模式
        wind_agent.tool_call_tracking_output(user_message)

        return jsonify({'success': True, 'message': '风电场仿真测试完成，请查看控制台输出'})

    except Exception as e:
        print(f"❌ 仿真测试错误: {e}")
        return jsonify({'success': False, 'message': f'测试失败: {str(e)}'})


@app.route('/analysis_test', methods=['POST'])
def analysis_test():
    """风电场分析测试接口"""
    try:
        user_message = request.form.get(
            'message',
            '请为simulation_result.nc生成所有类型的分析图表，包括功率热图、AEP对比、尾流损失分析等'
        )

        print(f"📊 分析测试请求: '{user_message}'")

        # 使用工具调用追踪模式
        wind_agent.tool_call_tracking_output(user_message)

        return jsonify({'success': True, 'message': '风电场分析测试完成，请查看控制台输出'})

    except Exception as e:
        print(f"❌ 分析测试错误: {e}")
        return jsonify({'success': False, 'message': f'测试失败: {str(e)}'})


@app.route('/full_analysis_test', methods=['POST'])
def full_analysis_test():
    """完整风电场分析测试接口"""
    try:
        user_message = request.form.get(
            'message',
            '请进行一次完整的风电场仿真分析，包括计算和所有图表生成，并提供专业的性能评估报告'
        )

        print(f"🔬 完整分析测试请求: '{user_message}'")

        # 使用工具调用追踪模式
        wind_agent.tool_call_tracking_output(user_message)

        return jsonify({'success': True, 'message': '完整风电场分析测试完成，请查看控制台输出'})

    except Exception as e:
        print(f"❌ 完整分析测试错误: {e}")
        return jsonify({'success': False, 'message': f'测试失败: {str(e)}'})


@app.route('/get_simulation_files')
def get_simulation_files():
    """获取可用的仿真文件列表"""
    try:
        import os
        pywake_dir = os.path.join(os.path.dirname(__file__), "pywake_me")

        if os.path.exists(pywake_dir):
            nc_files = [f for f in os.listdir(pywake_dir) if f.endswith('.nc')]
            return jsonify({
                'success': True,
                'files': nc_files,
                'directory': pywake_dir
            })
        else:
            return jsonify({
                'success': False,
                'message': 'pywake_me目录不存在',
                'files': []
            })

    except Exception as e:
        return jsonify({'success': False, 'message': str(e), 'files': []})


@app.route('/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'app': '风电场智能体HTML流式分析应用',
        'version': '1.0.0',
        'features': [
            'wind_farm_simulation',
            'pywake_analysis',
            'power_mapping',
            'aep_calculation',
            'wake_loss_analysis',
            'html_streaming',
            'professional_reporting'
        ],
        'tools': [
            'run_wind_farm_simulation',
            'generate_power_map_plot',
            'generate_aep_comparison_plot',
            'generate_turbine_aep_plot',
            'generate_wake_loss_heatmap',
            'generate_aep_windspeed_plot'
        ]
    })


@app.route('/system_status')
def system_status():
    """系统状态检查"""
    try:
        # 检查关键组件
        status = {
            'agent_initialized': wind_agent is not None,
            'tools_count': len(wind_agent.tools) if wind_agent else 0,
            'graph_ready': hasattr(wind_agent, 'graph') and wind_agent.graph is not None,
            'pywake_available': False,
            'simulation_files': []
        }

        # 检查PyWake模块
        try:
            from pywake_me import WindFarmSimulation
            status['pywake_available'] = True
        except ImportError:
            status['pywake_available'] = False

        # 检查仿真文件
        import os
        pywake_dir = os.path.join(os.path.dirname(__file__), "pywake_me")
        if os.path.exists(pywake_dir):
            status['simulation_files'] = [f for f in os.listdir(pywake_dir) if f.endswith('.nc')]

        return jsonify({
            'success': True,
            'status': status,
            'ready': all([
                status['agent_initialized'],
                status['graph_ready'],
                status['pywake_available']
            ])
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'系统状态检查失败: {str(e)}',
            'status': {}
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


@app.route('/static/plots/<filename>')
def serve_plot(filename):
    """服务图片文件"""
    import os
    from flask import send_from_directory

    static_plots_dir = os.path.join(os.path.dirname(__file__), 'static', 'plots')

    try:
        return send_from_directory(static_plots_dir, filename)
    except FileNotFoundError:
        return "图片未找到", 404


if __name__ == '__main__':
    print("🚀 风电场智能体HTML流式分析应用启动...")
    print("📍 访问地址: http://127.0.0.1:5000")

    app.run(debug=True, host='0.0.0.0', port=5000)