from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )
        # 新增：存储对话历史
        self.conversation_history = []

    def execute_stream(self, query: str):
        # 将用户消息加入历史
        self.conversation_history.append({"role": "user", "content": query})

        # 构造输入（包含完整历史）
        input_dict = {
            "messages": self.conversation_history.copy()
        }

        full_response = ""
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                content = latest_message.content.strip()
                full_response += content + "\n"
                yield content + "\n"

        # 将助手回复加入历史
        if full_response.strip():
            self.conversation_history.append({"role": "assistant", "content": full_response.strip()})

    def clear_history(self):
        """清除对话历史（可在前端添加按钮调用）"""
        self.conversation_history = []
