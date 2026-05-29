import time
import streamlit as st
from agent.react_agent import ReactAgent
from agent.tools import agent_tools
import uuid

# ------------------------------
# 全国省市数据（省份 -> 城市列表）
# ------------------------------
CHINA_CITIES = {
    "北京市": ["北京市"],
    "天津市": ["天津市"],
    "上海市": ["上海市"],
    "重庆市": ["重庆市"],
    "河北省": ["石家庄市", "唐山市", "秦皇岛市", "邯郸市", "邢台市", "保定市", "张家口市", "承德市", "沧州市", "廊坊市", "衡水市"],
    "山西省": ["太原市", "大同市", "阳泉市", "长治市", "晋城市", "朔州市", "晋中市", "运城市", "忻州市", "临汾市", "吕梁市"],
    "辽宁省": ["沈阳市", "大连市", "鞍山市", "抚顺市", "本溪市", "丹东市", "锦州市", "营口市", "阜新市", "辽阳市", "盘锦市", "铁岭市", "朝阳市", "葫芦岛市"],
    "吉林省": ["长春市", "吉林市", "四平市", "辽源市", "通化市", "白山市", "松原市", "白城市", "延边朝鲜族自治州"],
    "黑龙江省": ["哈尔滨市", "齐齐哈尔市", "鸡西市", "鹤岗市", "双鸭山市", "大庆市", "伊春市", "佳木斯市", "七台河市", "牡丹江市", "黑河市", "绥化市", "大兴安岭地区"],
    "江苏省": ["南京市", "无锡市", "徐州市", "常州市", "苏州市", "南通市", "连云港市", "淮安市", "盐城市", "扬州市", "镇江市", "泰州市", "宿迁市"],
    "浙江省": ["杭州市", "宁波市", "温州市", "嘉兴市", "湖州市", "绍兴市", "金华市", "衢州市", "舟山市", "台州市", "丽水市"],
    "安徽省": ["合肥市", "芜湖市", "蚌埠市", "淮南市", "马鞍山市", "淮北市", "铜陵市", "安庆市", "黄山市", "滁州市", "阜阳市", "宿州市", "六安市", "亳州市", "池州市", "宣城市"],
    "福建省": ["福州市", "厦门市", "莆田市", "三明市", "泉州市", "漳州市", "南平市", "龙岩市", "宁德市"],
    "江西省": ["南昌市", "景德镇市", "萍乡市", "九江市", "新余市", "鹰潭市", "赣州市", "吉安市", "宜春市", "抚州市", "上饶市"],
    "山东省": ["济南市", "青岛市", "淄博市", "枣庄市", "东营市", "烟台市", "潍坊市", "济宁市", "泰安市", "威海市", "日照市", "临沂市", "德州市", "聊城市", "滨州市", "菏泽市"],
    "河南省": ["郑州市", "开封市", "洛阳市", "平顶山市", "安阳市", "鹤壁市", "新乡市", "焦作市", "濮阳市", "许昌市", "漯河市", "三门峡市", "南阳市", "商丘市", "信阳市", "周口市", "驻马店市"],
    "湖北省": ["武汉市", "黄石市", "十堰市", "宜昌市", "襄阳市", "鄂州市", "荆门市", "孝感市", "荆州市", "黄冈市", "咸宁市", "随州市", "恩施土家族苗族自治州"],
    "湖南省": ["长沙市", "株洲市", "湘潭市", "衡阳市", "邵阳市", "岳阳市", "常德市", "张家界市", "益阳市", "郴州市", "永州市", "怀化市", "娄底市", "湘西土家族苗族自治州"],
    "广东省": ["广州市", "韶关市", "深圳市", "珠海市", "汕头市", "佛山市", "江门市", "湛江市", "茂名市", "肇庆市", "惠州市", "梅州市", "汕尾市", "河源市", "阳江市", "清远市", "东莞市", "中山市", "潮州市", "揭阳市", "云浮市"],
    "广西壮族自治区": ["南宁市", "柳州市", "桂林市", "梧州市", "北海市", "防城港市", "钦州市", "贵港市", "玉林市", "百色市", "贺州市", "河池市", "来宾市", "崇左市"],
    "海南省": ["海口市", "三亚市", "三沙市", "儋州市"],
    "四川省": ["成都市", "自贡市", "攀枝花市", "泸州市", "德阳市", "绵阳市", "广元市", "遂宁市", "内江市", "乐山市", "南充市", "眉山市", "宜宾市", "广安市", "达州市", "雅安市", "巴中市", "资阳市", "阿坝藏族羌族自治州", "甘孜藏族自治州", "凉山彝族自治州"],
    "贵州省": ["贵阳市", "六盘水市", "遵义市", "安顺市", "毕节市", "铜仁市", "黔西南布依族苗族自治州", "黔东南苗族侗族自治州", "黔南布依族苗族自治州"],
    "云南省": ["昆明市", "曲靖市", "玉溪市", "保山市", "昭通市", "丽江市", "普洱市", "临沧市", "楚雄彝族自治州", "红河哈尼族彝族自治州", "文山壮族苗族自治州", "西双版纳傣族自治州", "大理白族自治州", "德宏傣族景颇族自治州", "怒江傈僳族自治州", "迪庆藏族自治州"],
    "西藏自治区": ["拉萨市", "日喀则市", "昌都市", "林芝市", "山南市", "那曲市", "阿里地区"],
    "陕西省": ["西安市", "铜川市", "宝鸡市", "咸阳市", "渭南市", "延安市", "汉中市", "榆林市", "安康市", "商洛市"],
    "甘肃省": ["兰州市", "嘉峪关市", "金昌市", "白银市", "天水市", "武威市", "张掖市", "平凉市", "酒泉市", "庆阳市", "定西市", "陇南市", "临夏回族自治州", "甘南藏族自治州"],
    "青海省": ["西宁市", "海东市", "海北藏族自治州", "黄南藏族自治州", "海南藏族自治州", "果洛藏族自治州", "玉树藏族自治州", "海西蒙古族藏族自治州"],
    "宁夏回族自治区": ["银川市", "石嘴山市", "吴忠市", "固原市", "中卫市"],
    "新疆维吾尔自治区": ["乌鲁木齐市", "克拉玛依市", "吐鲁番市", "哈密市", "昌吉回族自治州", "博尔塔拉蒙古自治州", "巴音郭楞蒙古自治州", "阿克苏地区", "克孜勒苏柯尔克孜自治州", "喀什地区", "和田地区", "伊犁哈萨克自治州", "塔城地区", "阿勒泰地区"],
    "香港特别行政区": ["香港特别行政区"],
    "澳门特别行政区": ["澳门特别行政区"],
    "台湾省": ["台北市", "新北市", "桃园市", "台中市", "台南市", "高雄市", "基隆市", "新竹市", "嘉义市", "新竹县", "苗栗县", "彰化县", "南投县", "云林县", "嘉义县", "屏东县", "宜兰县", "花莲县", "台东县", "澎湖县"]
}
PROVINCES = list(CHINA_CITIES.keys())

# ------------------------------
# 页面配置
# ------------------------------
st.set_page_config(page_title="户外运动健康助手", page_icon="🏃")

# ------------------------------
# 初始化 session_state
# ------------------------------
if "user_city" not in st.session_state:
    st.session_state.user_city = "北京市"

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "agent" not in st.session_state:
    st.session_state.agent = ReactAgent(user_id=st.session_state.user_id)

if "message" not in st.session_state:
    st.session_state["message"] = [
        {"role": "assistant",
         "content": "嗨！我是您的户外运动健康助手。我可以帮您查询天气是否适合运动、提供运动知识、生成您的运动月报。请问有什么可以帮您？ 💪"}
    ]
    st.session_state["message"].append(
        {"role": "assistant",
         "content": f"您当前所在城市：{st.session_state.user_city}（如需更改，请在左侧侧边栏选择并确认）"}
    )

# ------------------------------
# 侧边栏：地区设置 + 清除历史
# ------------------------------
with st.sidebar:
    st.header("🏙️ 地区设置")
    st.sidebar.write(f"用户ID: {st.session_state.user_id[:8]}...")
    selected_province = st.selectbox("省份 / 直辖市 / 自治区", PROVINCES, index=PROVINCES.index("北京市"))
    city_list = CHINA_CITIES.get(selected_province, [selected_province])
    selected_city = st.selectbox("城市", city_list, index=0)

    if st.button("✅ 确认并更新城市"):
        old_city = st.session_state.user_city
        st.session_state.user_city = selected_city
        agent_tools.set_user_city(selected_city)
        st.session_state["message"].append(
            {"role": "assistant",
             "content": f"您已将所在城市从 {old_city} 更改为 {selected_city}。后续天气查询将基于新城市。"}
        )
        st.success(f"城市已更新为：{selected_city}")

    st.divider()

    if st.button("🗑️ 清除对话历史"):
        st.session_state.agent.clear_history()
        st.session_state["message"] = [
            {"role": "assistant", "content": "对话已重置。我是您的运动健康助手，请随时提问。"}
        ]
        st.rerun()

# 确保 agent_tools 中的城市始终是最新的
agent_tools.set_user_city(st.session_state.user_city)

# ------------------------------
# 主界面
# ------------------------------
st.title("🏃 户外运动健康助手")
st.markdown(f"📍 您当前所在城市：**{st.session_state.user_city}**")
st.divider()

# 显示历史消息
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入
prompt = st.chat_input("例如：今天适合户外运动吗？")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("正在分析，请稍后..."):
        res_stream = st.session_state.agent.execute_stream(prompt)

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        if response_messages:
            st.session_state["message"].append({"role": "assistant", "content": response_messages[-1]})
        else:
            st.session_state["message"].append({"role": "assistant", "content": "抱歉，未能获取回复，请稍后再试。"})
        st.rerun()