# app.py - 使用SQLAlchemy ORM的Flask学生管理系统
from flask import Flask, render_template, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String, DateTime, func
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import traceback

# 创建Flask应用
app = Flask(__name__)

# ==================== SQLAlchemy配置 ====================

# 创建基类
Base = declarative_base()


# 定义学生模型
class Student(Base):
    """学生数据模型"""
    __tablename__ = 'students'

    # 字段定义
    id = Column(Integer, primary_key=True, autoincrement=True)  # 主键，自增
    name = Column(String(100), nullable=False)  # 姓名，不能为空
    age = Column(Integer, nullable=False)  # 年龄，不能为空
    created_at = Column(DateTime, default=datetime.now)  # 创建时间
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)  # 更新时间

    def __repr__(self):
        """对象的字符串表示"""
        return f"<Student(id={self.id}, name='{self.name}', age={self.age})>"

    def to_dict(self):
        """将对象转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'age': self.age,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else '',
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S') if self.updated_at else ''
        }


# ==================== 数据库配置 ====================

# 数据库连接URI（请根据你的配置修改）
DATABASE_URI = 'mysql+pymysql://root:Sunkai12@localhost:3306/student_management?charset=utf8mb4'

# 可选的其他配置示例：
# DATABASE_URI = 'mysql+pymysql://root:password@192.168.110.131:3306/student_management?charset=utf8mb4'

try:
    # 创建数据库引擎
    engine = create_engine(
        DATABASE_URI,
        echo=True,  # 打印SQL语句（调试用）
        pool_size=10,  # 连接池大小
        max_overflow=20,  # 最大溢出连接数
        pool_recycle=3600,  # 连接回收时间（秒）
        pool_pre_ping=True  # 连接前ping测试
    )

    # 创建所有表（如果不存在）
    Base.metadata.create_all(engine)
    print("✅ 数据库表创建成功")

    # 创建会话工厂
    SessionLocal = sessionmaker(bind=engine)

except Exception as e:
    print(f"❌ 数据库连接失败: {e}")
    print("💡 请检查：")
    print("   1. MySQL服务是否运行")
    print("   2. 数据库连接信息是否正确")
    print("   3. 数据库是否存在（如果不存在请先创建）")
    exit(1)


# ==================== 数据库操作函数 ====================

def init_sample_data():
    """初始化示例数据"""
    session = SessionLocal()
    try:
        # 检查是否已有数据
        count = session.query(Student).count()
        if count == 0:
            print("📝 插入初始数据...")

            # 初始学生数据
            sample_students = [
                Student(name="张三", age=20),
                Student(name="李四", age=21),
                Student(name="王五", age=19),
                Student(name="赵六", age=22),
                Student(name="钱七", age=18)
            ]

            # 批量添加
            session.add_all(sample_students)
            session.commit()

            print(f"✅ 成功插入 {len(sample_students)} 条初始数据")
        else:
            print(f"📊 数据库中已有 {count} 条学生记录")

    except Exception as e:
        session.rollback()
        print(f"❌ 初始化数据失败: {e}")
    finally:
        session.close()


def get_all_students():
    """获取所有学生"""
    session = SessionLocal()
    try:
        # 查询所有学生，按创建时间倒序
        students = session.query(Student).order_by(Student.created_at.desc()).all()

        # 转换为字典列表
        students_list = [student.to_dict() for student in students]
        return students_list

    except Exception as e:
        print(f"❌ 查询学生失败: {e}")
        return []
    finally:
        session.close()


def add_student_to_db(name, age):
    """添加学生到数据库"""
    session = SessionLocal()
    try:
        # 检查学生是否已存在
        existing_student = session.query(Student).filter(Student.name == name).first()
        if existing_student:
            return False, f"学生 {name} 已存在"

        # 创建新学生对象
        new_student = Student(name=name, age=age)

        # 添加到会话并提交
        session.add(new_student)
        session.commit()

        print(f"✅ 成功添加学生: {name} (ID: {new_student.id})")
        return True, f"学生 {name} 添加成功！"

    except Exception as e:
        session.rollback()
        print(f"❌ 添加学生失败: {e}")
        return False, f"添加失败: {str(e)}"
    finally:
        session.close()


def delete_student_from_db(name):
    """从数据库删除学生"""
    session = SessionLocal()
    try:
        # 查找学生
        student = session.query(Student).filter(Student.name == name).first()

        if not student:
            return False, f"学生 {name} 不存在"

        # 删除学生
        session.delete(student)
        session.commit()

        print(f"✅ 成功删除学生: {name}")
        return True, f"学生 {name} 删除成功！"

    except Exception as e:
        session.rollback()
        print(f"❌ 删除学生失败: {e}")
        return False, f"删除失败: {str(e)}"
    finally:
        session.close()


def get_student_by_id(student_id):
    """根据ID获取学生信息"""
    session = SessionLocal()
    try:
        student = session.query(Student).filter(Student.id == student_id).first()
        if student:
            return student.to_dict()
        return None
    except Exception as e:
        print(f"❌ 查询学生失败: {e}")
        return None
    finally:
        session.close()


def update_student(student_id, name, age):
    """更新学生信息"""
    session = SessionLocal()
    try:
        student = session.query(Student).filter(Student.id == student_id).first()
        if not student:
            return False, "学生不存在"

        # 检查新姓名是否与其他学生重复
        if student.name != name:
            existing = session.query(Student).filter(
                Student.name == name, Student.id != student_id
            ).first()
            if existing:
                return False, f"姓名 {name} 已被其他学生使用"

        # 更新信息
        student.name = name
        student.age = age
        student.updated_at = datetime.now()

        session.commit()
        return True, "学生信息更新成功"

    except Exception as e:
        session.rollback()
        print(f"❌ 更新学生失败: {e}")
        return False, f"更新失败: {str(e)}"
    finally:
        session.close()


def get_students_stats():
    """获取学生统计信息"""
    session = SessionLocal()
    try:
        # 总学生数
        total_count = session.query(Student).count()

        # 平均年龄
        avg_age = session.query(func.avg(Student.age)).scalar()
        avg_age = round(avg_age, 1) if avg_age else 0

        # 最小最大年龄
        min_age = session.query(func.min(Student.age)).scalar() or 0
        max_age = session.query(func.max(Student.age)).scalar() or 0

        # 最新添加的学生
        latest_student = session.query(Student).order_by(Student.created_at.desc()).first()

        return {
            'total_count': total_count,
            'avg_age': avg_age,
            'min_age': min_age,
            'max_age': max_age,
            'latest_student': latest_student.name if latest_student else '无'
        }

    except Exception as e:
        print(f"❌ 获取统计信息失败: {e}")
        return {
            'total_count': 0,
            'avg_age': 0,
            'min_age': 0,
            'max_age': 0,
            'latest_student': '无'
        }
    finally:
        session.close()


# ==================== Flask路由 ====================

@app.route('/')
def home():
    """主页 - 显示学生列表"""
    students = get_all_students()
    stats = get_students_stats()

    print(f"📊 当前数据库中有 {len(students)} 名学生")
    return render_template('chat.html', students=students, stats=stats)


@app.route('/add', methods=['POST'])
def add_student():
    """添加学生API"""
    try:
        # 获取表单数据
        name = request.form.get('name', '').strip()
        age = request.form.get('age', '').strip()

        print(f"🔄 收到添加请求: 姓名={name}, 年龄={age}")

        # 数据验证
        if not name:
            return jsonify({"success": False, "message": "姓名不能为空"})

        if len(name) > 50:
            return jsonify({"success": False, "message": "姓名长度不能超过50个字符"})

        if not age:
            return jsonify({"success": False, "message": "年龄不能为空"})

        try:
            age = int(age)
            if age < 1 or age > 150:
                return jsonify({"success": False, "message": "年龄必须在1-150之间"})
        except ValueError:
            return jsonify({"success": False, "message": "年龄必须是数字"})

        # 调用数据库操作
        success, message = add_student_to_db(name, age)
        return jsonify({"success": success, "message": message})

    except Exception as e:
        print(f"❌ 添加学生API错误: {e}")
        print(traceback.format_exc())
        return jsonify({"success": False, "message": "服务器内部错误"})


@app.route('/delete', methods=['POST'])
def delete_student():
    """删除学生API"""
    try:
        name = request.form.get('name', '').strip()

        print(f"🔄 收到删除请求: 姓名={name}")

        if not name:
            return jsonify({"success": False, "message": "姓名不能为空"})

        # 调用数据库操作
        success, message = delete_student_from_db(name)
        return jsonify({"success": success, "message": message})

    except Exception as e:
        print(f"❌ 删除学生API错误: {e}")
        return jsonify({"success": False, "message": "服务器内部错误"})


@app.route('/api/students', methods=['GET'])
def get_students_api():
    """获取所有学生的API接口"""
    try:
        students = get_all_students()
        stats = get_students_stats()

        return jsonify({
            "success": True,
            "data": students,
            "stats": stats
        })
    except Exception as e:
        print(f"❌ 获取学生API错误: {e}")
        return jsonify({"success": False, "message": "获取数据失败"})


@app.route('/api/student/<int:student_id>', methods=['GET'])
def get_student_api(student_id):
    """获取单个学生信息"""
    try:
        student = get_student_by_id(student_id)
        if student:
            return jsonify({"success": True, "data": student})
        else:
            return jsonify({"success": False, "message": "学生不存在"})
    except Exception as e:
        print(f"❌ 获取学生信息错误: {e}")
        return jsonify({"success": False, "message": "获取学生信息失败"})


# ==================== 错误处理 ====================

@app.errorhandler(500)
def internal_error(error):
    """处理500错误"""
    print(f"❌ 服务器错误: {error}")
    return jsonify({
        "success": False,
        "message": "服务器内部错误，请检查数据库连接"
    }), 500


@app.errorhandler(404)
def not_found(error):
    """处理404错误"""
    return jsonify({
        "success": False,
        "message": "页面未找到"
    }), 404


# ==================== 应用启动 ====================

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Flask + SQLAlchemy ORM 应用启动中...")
    print(f"🗄️  数据库: {DATABASE_URI.split('@')[1].split('?')[0]}")
    print("📊 使用 SQLAlchemy ORM 进行数据操作")
    print("=" * 70)

    try:
        # 初始化示例数据
        init_sample_data()

        print("📍 访问地址: http://127.0.0.1:5000")
        print("🔧 调试模式: 开启")
        print("💡 SQLAlchemy特性:")
        print("   - ORM对象关系映射")
        print("   - 自动SQL生成")
        print("   - 连接池管理")
        print("   - 事务支持")
        print("=" * 70)

        # 启动Flask应用
        app.run(debug=True, host='127.0.0.1', port=5000)

    except Exception as e:
        print(f"❌ 应用启动失败: {e}")
        print("💡 请检查：")
        print("   1. MySQL服务正在运行")
        print("   2. 数据库连接信息正确")
        print("   3. 数据库已创建")
        print("   4. 已安装依赖: pip install flask sqlalchemy pymysql")