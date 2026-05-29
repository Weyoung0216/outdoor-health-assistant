import os
from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
import random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
import requests

import streamlit as st

# 优先从 st.secrets 读取，其次从环境变量，最后从配置文件（本地开发兼容）
try:
    QWEATHER_API_KEY = st.secrets["QWEATHER_API_KEY"]
except Exception:
    QWEATHER_API_KEY = os.getenv("QWEATHER_API_KEY")
    if not QWEATHER_API_KEY:
        from utils.config_handler import agent_conf
        QWEATHER_API_KEY = agent_conf.get("qweather_api_key")


try:
    API_HOST = st.secrets["WEATHER_API_HOST"]
except Exception:
    API_HOST = os.getenv("WEATHER_API_HOST")
    if not API_HOST:
        # 本地开发时的默认值（可选）
        API_HOST = "https://qc4ewukxnu.re.qweatherapi.com"

# 用于存储用户当前所在城市（可由前端动态设置）
_current_user_city = None

def set_user_city(city: str):
    global _current_user_city
    _current_user_city = city

def get_user_city() -> str:
    return _current_user_city if _current_user_city else "深圳"  # 默认值


def _get_location_id(city_name: str) -> str | None:
    if not QWEATHER_API_KEY:
        logger.error("和风天气 API Key 未配置")
        return None
    geo_url = f"{API_HOST}/geo/v2/city/lookup"
    params = {"location": city_name, "key": QWEATHER_API_KEY}
    try:
        resp = requests.get(geo_url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == "200" and data.get("location"):
            return data["location"][0]["id"]
        else:
            logger.warning(f"未找到城市 {city_name} 的 location_id, code={data.get('code')}")
            return None
    except Exception as e:
        logger.error(f"获取 location_id 失败: {e}")
        return None


rag = RagSummarizeService()

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]

external_data = {}


@tool(description="从运动健康知识库中检索跑步技巧、损伤预防、不同天气下的运动指南、运动营养等专业内容")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description="获取指定城市的实时天气（温度、湿度、风力、天气现象），用于判断运动适宜性")
def get_weather(city: str) -> str:
    if not QWEATHER_API_KEY:
        return "天气服务未配置，请联系管理员。"
    location_id = _get_location_id(city)
    if not location_id:
        return f"未找到城市「{city}」的天气信息。"
    weather_url = f"{API_HOST}/v7/weather/now"
    params = {"location": location_id, "key": QWEATHER_API_KEY}
    try:
        resp = requests.get(weather_url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == "200":
            now = data["now"]
            return (f"{city}当前天气：{now['text']}，气温{now['temp']}℃，体感{now['feelsLike']}℃，"
                    f"湿度{now['humidity']}%，风向{now['windDir']}，风力{now['windScale']}级")
        else:
            logger.error(f"和风天气 API 错误码: {data.get('code')}")
            return f"查询{city}天气失败，请稍后重试。"
    except requests.exceptions.Timeout:
        logger.error(f"查询{city}天气超时")
        return f"查询{city}天气超时，请稍后再试。"
    except Exception as e:
        logger.error(f"查询{city}天气异常: {e}")
        return f"查询{city}天气时出现错误。"


@tool(description="获取用户当前所在城市名称")
def get_user_location() -> str:
    return get_user_city()


@tool(description="获取用户的唯一标识ID")
def get_user_id() -> str:
    return random.choice(user_ids)


@tool(description="获取当前月份，格式YYYY-MM")
def get_current_month() -> str:
    return random.choice(month_arr)


def generate_external_data():
    """从CSV加载用户运动记录，CSV列顺序：用户ID,用户画像,运动摘要,装备/健康指标,对比/建议,时间"""
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])
        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"运动数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            # 跳过标题行
            next(f)
            for line in f:
                arr = line.strip().split(",")
                if len(arr) < 6:
                    continue
                user_id = arr[0].strip('"')
                profile = arr[1].strip('"')
                activity_summary = arr[2].strip('"')
                health_metrics = arr[3].strip('"')
                suggestion = arr[4].strip('"')
                record_date = arr[5].strip('"')

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][record_date] = {
                    "用户画像": profile,
                    "运动摘要": activity_summary,
                    "健康指标": health_metrics,
                    "对比建议": suggestion,
                }


@tool(description="获取用户在指定月份的运动记录（活动类型、时长、心率、表现评分、天气状况）")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()
    try:
        # month 格式 YYYY-MM，CSV中的日期是 YYYY-MM-DD，所以需要匹配前缀
        matched = {}
        for date_str, record in external_data.get(user_id, {}).items():
            if date_str.startswith(month):
                matched[date_str] = record
        if matched:
            # 返回该月所有记录的字符串表示（可按需格式化）
            return str(matched)
        else:
            logger.warning(f"未找到用户 {user_id} 在 {month} 的运动记录")
            return ""
    except Exception as e:
        logger.error(f"获取运动记录失败: {e}")
        return ""


@tool(description="准备生成运动月报的上下文，调用后切换至报告模式。仅当用户要求生成报告时调用")
def fill_context_for_report():
    return "fill_context_for_report已调用"


if __name__ == '__main__':
    # 测试天气
    print(get_weather.func("武汉"))