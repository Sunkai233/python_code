# app.py - 简单的Flask大模型对话应用
from flask import Flask, render_template, request, jsonify
from langchain_deepseek import ChatDeepSeek

# 创建Flask应用
app = Flask(__name__)

# 初始化DeepSeek大模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=1024,
    timeout=30.0,
    max_retries=2,
    api_key="sk-d0e8a80b05f84e6ca11034fb45c2078b"
)


@app.route('/')
def home():
    """主页 - 对话界面"""
    return render_template('chat.html')


@app.route('/chat', methods=['POST'])
def chat():
    """对话API"""
    try:
        # 获取用户消息
        user_message = request.form.get('message', '').strip()

        # 🔍 调试：打印接收到的数据
        print("=" * 50)
        print(f"🔍 收到POST请求")
        print(f"🔍 原始form数据: {dict(request.form)}")
        print(f"🔍 用户消息: '{user_message}'")
        print(f"🔍 消息长度: {len(user_message)}")

        if not user_message:
            print("❌ 消息为空")
            return jsonify({
                "success": False,
                "message": "消息不能为空"
            })

        print(f"✅ 开始调用大模型...")

        # 构建消息
        messages = [
            ("system", "你是一个对话模型"),
            ("human", user_message)
        ]

        # 调用大模型
        finan_response = llm.invoke(messages)
        ai_message = finan_response.content

        # 🔍 调试：打印提取的AI消息

        # 构建返回数据
        response_data = {
            "success": True,
            "message": ai_message
        }

        # 🔍 调试：打印返回给前端的数据
        print(f"🔍 返回给前端的数据: {response_data}")
        print("=" * 50)

        return jsonify(response_data)

    except Exception as e:
        # 🔍 调试：打印详细错误信息
        print("=" * 50)
        print(f"❌ 发生错误: {e}")
        print(f"❌ 错误类型: {type(e)}")
        import traceback
        print(f"❌ 错误堆栈: {traceback.format_exc()}")
        print("=" * 50)

        return jsonify({
            "success": False,
            "message": f"出错了: {str(e)}"
        })


if __name__ == '__main__':
    print("🚀 Flask对话应用启动...")
    print("📍 访问地址: http://127.0.0.1:5000")
    app.run(debug=True)