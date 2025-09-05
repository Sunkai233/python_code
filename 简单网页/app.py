# app.py - 极简Flask后端示例
from flask import Flask, render_template, request, jsonify

# 创建Flask应用
app = Flask(__name__)

# 简单的内存数据存储
students = [
    {"name": "张三", "age": 20},
    {"name": "李四", "age": 21},
    {"name": "王五", "age": 19}
]


@app.route('/')
def home():
    """主页 - 显示学生列表"""
    return render_template('chat.html', students=students)


@app.route('/add', methods=['POST'])
def add_student():
    """添加学生API"""
    # 获取表单数据
    name = request.form.get('name')
    age = request.form.get('age')

    # 简单验证
    if name and age:
        try:
            age = int(age)
            new_student = {"name": name, "age": age}
            students.append(new_student)
            return jsonify({"success": True, "message": f"学生 {name} 添加成功！"})
        except:
            return jsonify({"success": False, "message": "年龄必须是数字"})
    else:
        return jsonify({"success": False, "message": "姓名和年龄不能为空"})


@app.route('/delete', methods=['POST'])
def delete_student():
    """删除学生API"""
    name = request.form.get('name')

    # 查找并删除学生
    for i, student in enumerate(students):
        if student['name'] == name:
            students.pop(i)
            return jsonify({"success": True, "message": f"学生 {name} 删除成功！"})

    return jsonify({"success": False, "message": "学生不存在"})


if __name__ == '__main__':
    print("🚀 Flask应用启动...")
    print("📍 访问地址: http://127.0.0.1:5000")
    app.run(debug=True)