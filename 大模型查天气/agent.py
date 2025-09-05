# agent.py - AI智能体模块
import os
import json
import requests
from typing import Union, Optional
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.prebuilt import create_react_agent
from base_agent import BaseAgent

# 设置环境变量
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "me_test1"
os.environ["DEEPSEEK_API_KEY"] = "sk-157abd02156e4718b1132b3ed03fd5ce"

# =============================================================================
# 数据库模型定义
# =============================================================================
Base = declarative_base()


class Weather(Base):
    __tablename__ = 'weather'
    city_id = Column(Integer, primary_key=True)
    city_name = Column(String(50))
    main_weather = Column(String(50))
    description = Column(String(100))
    temperature = Column(Float)
    feels_like = Column(Float)
    temp_min = Column(Float)
    temp_max = Column(Float)


# =============================================================================
# Pydantic模型定义
# =============================================================================
class WeatherLoc(BaseModel):
    location: str = Field(description="The location name of the city")


class WeatherInfo(BaseModel):
    """Extracted weather information for a specific city."""
    city_id: int = Field(..., description="The unique identifier for the city")
    city_name: str = Field(..., description="The name of the city")
    main_weather: str = Field(..., description="The main weather condition")
    description: str = Field(..., description="A detailed description of the weather")
    temperature: float = Field(..., description="Current temperature in Celsius")
    feels_like: float = Field(..., description="Feels-like temperature in Celsius")
    temp_min: float = Field(..., description="Minimum temperature in Celsius")
    temp_max: float = Field(..., description="Maximum temperature in Celsius")


class QueryWeatherSchema(BaseModel):
    """Schema for querying weather information by city name."""
    city_name: str = Field(..., description="The name of the city to query weather information")


class SearchQuery(BaseModel):
    query: str = Field(description="Questions for networking queries")


# =============================================================================
# 工具函数定义
# =============================================================================
def get_weather(loc):
    """
    Function to query current weather.
    :param loc: Required parameter, of type string, representing the specific city name for the weather query.
    :return: The result of the OpenWeather API query for current weather.
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": loc,
        "appid": "887d13ad15ef6fd15a6f488b34760b6c",
        "units": "metric",
        "lang": "zh_cn"
    }

    response = requests.get(url, params=params)
    data = response.json()
    return json.dumps(data)


@tool(args_schema=WeatherInfo)
def insert_weather_to_db(city_id, city_name, main_weather, description, temperature, feels_like, temp_min, temp_max):
    """Insert weather information into the database."""
    from .database import get_session  # 延迟导入避免循环依赖

    session = get_session()
    try:
        weather = Weather(
            city_id=city_id,
            city_name=city_name,
            main_weather=main_weather,
            description=description,
            temperature=temperature,
            feels_like=feels_like,
            temp_min=temp_min,
            temp_max=temp_max
        )
        session.merge(weather)
        session.commit()
        return {"messages": [f"天气数据已成功存储至Mysql数据库。"]}
    except Exception as e:
        session.rollback()
        return {"messages": [f"数据存储失败，错误原因：{e}"]}
    finally:
        session.close()


@tool(args_schema=QueryWeatherSchema)
def query_weather_from_db(city_name: str):
    """Query weather information from the database by city name."""
    from .database import get_session  # 延迟导入避免循环依赖

    session = get_session()
    try:
        weather_data = session.query(Weather).filter(Weather.city_name == city_name).first()
        if weather_data:
            return {
                "city_id": weather_data.city_id,
                "city_name": weather_data.city_name,
                "main_weather": weather_data.main_weather,
                "description": weather_data.description,
                "temperature": weather_data.temperature,
                "feels_like": weather_data.feels_like,
                "temp_min": weather_data.temp_min,
                "temp_max": weather_data.temp_max
            }
        else:
            return {"messages": [f"未找到城市 '{city_name}' 的天气信息。"]}
    except Exception as e:
        return {"messages": [f"查询失败，错误原因：{e}"]}
    finally:
        session.close()


@tool(args_schema=SearchQuery)
def fetch_real_time_info(query):
    """Get real-time Internet information"""
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "num": 1,
    })
    headers = {
        'X-API-KEY': 'cd872fca99047eb9165242365c65b858bc8970c0',
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, data=payload)
    data = json.loads(response.text)
    if 'organic' in data:
        return json.dumps(data['organic'], ensure_ascii=False)
    else:
        return json.dumps({"error": "No organic results found"}, ensure_ascii=False)


# =============================================================================
# 数据库管理类
# =============================================================================
class DatabaseManager:
    """数据库管理器"""

    def __init__(self, database_uri=None):
        self.database_uri = database_uri or 'mysql+pymysql://root:Sunkai12@localhost:3306/langgraph_agent?charset=utf8mb4'
        self.engine = create_engine(self.database_uri)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def get_session(self):
        """获取数据库会话"""
        return self.Session()


# =============================================================================
# 智能体类定义
# =============================================================================
class WeatherAgent(BaseAgent):
    """天气查询智能体，继承自BaseAgent"""

    def __init__(self, api_key=None, database_uri=None):
        """初始化天气智能体

        Args:
            api_key (str): DeepSeek API密钥
            database_uri (str): 数据库连接URI
        """
        super().__init__()  # 调用父类初始化

        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")

        # 初始化数据库
        self.db_manager = DatabaseManager(database_uri)

        # 初始化LLM
        self.llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2048,
            timeout=30.0,
            max_retries=2,
            api_key=self.api_key,
        )

        # 系统提示词
        self.system_prompt = self._get_system_prompt()

        # 工具列表
        self.tools = [fetch_real_time_info, get_weather, insert_weather_to_db, query_weather_from_db]

        # 创建ReAct智能体 - 设置父类需要的graph属性
        self.graph = create_react_agent(self.llm, tools=self.tools, prompt=self.system_prompt)

    def _get_system_prompt(self):
        """获取系统提示词"""
        return """你的回复需要使用结构化的HTML格式。

## 重要规则：
1. 你的回复必须是有效的HTML片段，不需要完整的HTML文档结构
2. 直接输出HTML内容，不要用代码块包裹
3. 使用适当的HTML标签来组织内容结构
4. 优先使用工具获取准确、实时的信息
5. 我get_weather用的https://api.openweathermap.org/data/2.5/weather这个网站，请使用这个网站的城市名查询：比如北京是beijing

## HTML格式规范：
### 可用的HTML标签和建议用法：
- `<h1>`, `<h2>`, `<h3>` 用于标题层级
- `<p>` 用于段落文本
- `<strong>` 用于重要内容强调
- `<em>` 用于斜体强调
- `<ul><li>` 用于无序列表
- `<ol><li>` 用于有序列表
- `<table><tr><td>` 用于表格数据展示
- `<blockquote>` 用于引用天气预报信息
- `<code>` 用于行内数据值
- `<span style="color: #color;">` 用于颜色标记（温度用红/蓝，湿度用蓝色等）

### 天气信息展示格式示例：
```html
<h2>🌤️ 北京天气实况</h2>
<table border="1" style="border-collapse: collapse; width: 100%;">
<tr>
    <td><strong>🌡️ 温度</strong></td>
    <td><span style="color: #e74c3c;">25.8°C</span></td>
</tr>
<tr>
    <td><strong>🌫️ 天气状况</strong></td>
    <td>多云</td>
</tr>
<tr>
    <td><strong>💧 湿度</strong></td>
    <td><span style="color: #3498db;">65%</span></td>
</tr>
</table>

<h3>📊 多城市天气对比</h3>
<ul>
<li><strong>最高温度</strong>：<span style="color: #e74c3c;">上海 28.5°C</span></li>
<li><strong>最低温度</strong>：<span style="color: #3498db;">哈尔滨 15.2°C</span></li>
<li><strong>最佳天气</strong>：<span style="color: #27ae60;">昆明 晴朗</span></li>
</ul>
```

请始终遵循这个格式和工具使用策略，让你的回复既专业又实用。"""
# =============================================================================
# 工厂函数
# =============================================================================
def create_weather_agent(api_key=None, database_uri=None):
    """创建天气智能体实例

    Args:
        api_key (str): DeepSeek API密钥
        database_uri (str): 数据库连接URI

    Returns:
        WeatherAgent: 天气智能体实例
    """
    return WeatherAgent(api_key, database_uri)


def format_stream_data(data_dict):
    """格式化流式数据为SSE格式

    Args:
        data_dict (dict): 数据字典

    Returns:
        str: SSE格式的数据
    """
    return f"data: {json.dumps(data_dict)}\n\n"


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    # 创建智能体
    agent = create_weather_agent()

    print("=== 测试天气智能体 ===")

    # 测试简单模式
    print("\n1. 测试简单模式:")
    result = agent.chat_simple("北京现在的天气怎么样？")
    print(f"结果: {result}")

    # 测试流式模式
    print("\n2. 测试流式模式:")
    for data in agent.chat_stream("帮我查一下上海的天气"):
        print(f"流式数据: {data}")

    # 测试工具调用追踪
    print("\n3. 测试工具调用追踪:")
    agent.tool_call_tracking_output("帮我对比查一下重庆、宁波，海南三个城市的天气")
