from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from utils.long_term_memory import add_memory, retrieve_memories
import uuid


class ReactAgent:
    def __init__(self, user_id: str = None):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )
        # 用户标识（用于长期记忆分区）
        self.user_id = user_id or str(uuid.uuid4())

        # 短期记忆：原始消息窗口
        self.short_messages = []          # 存储 {"role": "user"/"assistant", "content": ...}
        self.summary_messages = []        # 存储压缩生成的摘要 {"role": "system", "content": "摘要..."}
        self.window_size = 8              # 原始消息窗口最大数量
        self.compress_trigger = 4         # 当超过窗口时，压缩最旧的几条

        # 长期记忆消息（每次检索后暂存）
        self.long_term_msg = None

    def _compress_messages(self):
        """当 short_messages 超过 window_size 时，将最旧的 compress_trigger 条消息压缩成一条摘要"""
        if len(self.short_messages) <= self.window_size:
            return
        to_compress = self.short_messages[:self.compress_trigger]
        conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in to_compress])
        summary_prompt = f"请将以下对话压缩成一段简洁的摘要（保留关键信息，如用户偏好、已解决的问题等），字数控制在150字以内：\n{conversation_text}\n摘要："
        try:
            response = chat_model.invoke(summary_prompt)
            summary = response.content.strip()
        except Exception as e:
            summary = f"摘要生成失败：{str(e)}"
        # 创建摘要消息
        summary_msg = {"role": "system", "content": f"【对话摘要】{summary}"}
        # 更新列表
        self.short_messages = self.short_messages[self.compress_trigger:]
        self.summary_messages.append(summary_msg)

        # 将摘要存入长期记忆（跨会话保留）
        add_memory(self.user_id, summary, {"type": "conversation_summary"})

    def _inject_long_term_memory(self, query: str):
        """根据当前用户问题检索长期记忆，并构造系统消息"""
        memories = retrieve_memories(self.user_id, query, k=3)
        if memories:
            memory_text = "\n".join([f"- {m}" for m in memories])
            self.long_term_msg = {"role": "system", "content": f"【长期记忆】该用户之前提到过：\n{memory_text}\n请参考这些信息提供个性化回答。"}
        else:
            self.long_term_msg = None

    def _rebuild_conversation_history(self):
        """构建完整历史：长期记忆 + 摘要 + 短期原始消息"""
        history = []
        if self.long_term_msg:
            history.append(self.long_term_msg)
        history.extend(self.summary_messages)
        history.extend(self.short_messages)
        self.conversation_history = history

    def execute_stream(self, query: str):
        # 1. 根据本次用户问题检索长期记忆
        self._inject_long_term_memory(query)

        # 2. 将用户消息加入短期窗口
        self.short_messages.append({"role": "user", "content": query})
        # 3. 检查是否需要压缩
        self._compress_messages()
        # 4. 重建完整历史
        self._rebuild_conversation_history()
        # 5. 构造输入并流式输出
        input_dict = {"messages": self.conversation_history.copy()}
        full_response = ""
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                content = latest_message.content.strip()
                full_response += content + "\n"
                yield content + "\n"
        # 6. 将助手回复加入短期窗口
        if full_response.strip():
            self.short_messages.append({"role": "assistant", "content": full_response.strip()})
            # 再次检查压缩（因为刚加了一条回复）
            self._compress_messages()
            self._rebuild_conversation_history()

    def clear_history(self):
        """清除所有短期记忆（保留长期记忆）"""
        self.short_messages = []
        self.summary_messages = []
        self.long_term_msg = None
        self._rebuild_conversation_history()