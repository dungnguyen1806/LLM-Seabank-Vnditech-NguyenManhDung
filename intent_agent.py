import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.oauth2 import service_account
from utils.jsonProcess import load_conversation_history, save_conversation_history

import os
from dotenv import load_dotenv
import json
import logging

load_dotenv()

DEFAULT_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_PATH")
credentials = service_account.Credentials.from_service_account_file(DEFAULT_CREDENTIALS_PATH)
DEFAULT_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
DEFAULT_LOCATION = os.getenv("VERTEX_AI_LOCATION")
DEFAULT_HISTORY_PATH = os.getenv("HISTORY")
DEFAULT_MEMORY = os.getenv("MEMORY")


vertexai.init(project=DEFAULT_PROJECT, location=DEFAULT_LOCATION, credentials=credentials)   

DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME")
DEFAULT_MODEL_CONFIG = {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 500,
} 

def generate_intent_from_question(question: str,
                                  history_path: str = DEFAULT_HISTORY_PATH,
                                  user_info: dict = {"user_role": "Khối Khách Hàng Doanh Nghiệp", "user_datamart_schema": "ENT_DTM"},
                                  vertexai=None,
                                  model_name: str = DEFAULT_MODEL_NAME,
                                  config: dict = DEFAULT_MODEL_CONFIG) -> dict:
    """
    Generates a question with detected intent using Vertex AI LLM.
    """
    if vertexai is None:
        raise ValueError("vertexai instance must be provided")
    
    conversation_start = 0
    user_role = user_info.get("user_role")
    user_datamart_schema = user_info.get("user_datamart_schema")

    conversation_history = load_conversation_history(history_path=DEFAULT_HISTORY_PATH, memory=DEFAULT_MEMORY)

    formatted_history = json.dumps(conversation_history, ensure_ascii=False, indent=2)

    starter_prompt = f"""
    Bạn là một trợ lý AI dữ liệu có khả năng hiểu Xử lý Ngôn ngữ Tự nhiên (NLU) và phát hiện ý định. Nhiệm vụ của bạn là phân tích đầu vào của người dùng. 
    Người đặt câu hỏi là nhân viên {user_role} thuộc ngân hàng Seabank, mục tiêu là truy cập cơ sở dữ liệu datamart schema {user_datamart_schema} của Seabank.

    1. Xác định ý định của đầu vào và phân loại câu hỏi của người dùng: Xác định mục tiêu hoặc hành động của người dùng từ đầu vào. Nếu câu hỏi người dùng không có ý định truy vấn cơ sở dữ liệu của Seabank, định hướng lại cho người dùng hoặc từ chối trả lời câu hỏi ngoài lĩnh vực. Nếu câu hỏi của người dùng có ý định truy vấn cơ sở dữ liệu của Seabank, hãy viết lại câu hỏi của người dùng thành một câu hỏi không có thành phần thừa không liên quan đến ý định của người dùng.

    2. Đầu ra: 

    In nội dung văn bản của đối tượng JSON với cấu trúc sau:
    {{
        "intent": "[RAG intent hoặc Non-RAG intent]",
        "question": "[Nội dung câu hỏi đã được viết lại hoặc null]",
        "prompt": "[Câu trả lời định hướng người dùng hoặc null]"
    }}

    - Giá trị phần tử intent chỉ có thể là: "RAG intent" hoặc "Non-RAG intent". Giá trị phần tử question là câu hỏi đã được viết lại của người dùng hoặc null nếu câu hỏi không có ý định. Giá trị phần tử prompt là câu trả lời định hướng người dùng hoặc null nếu người dùng có ý định.

    Câu hỏi: "{question}"
    """

    continuous_prompt = f"""
    Bạn là một trợ lý AI dữ liệu có khả năng hiểu Xử lý Ngôn ngữ Tự nhiên (NLU) và phát hiện ý định. Nhiệm vụ của bạn là phân tích đầu vào của người dùng.
    Đây là cuộc hội thoại giữa nhân viên {user_role} thuộc ngân hàng Seabank, mục tiêu là truy cập cơ sở dữ liệu datamart schema {user_datamart_schema} của Seabank.
    Ở dưới là lịch sử hội thoại dưới dạng JSON:

    {formatted_history}

    1. Xác định ý định của đầu vào và phân loại câu hỏi của người dùng: Xác định mục tiêu hoặc hành động của người dùng từ đầu vào. Nếu câu hỏi người dùng không có ý định truy vấn cơ sở dữ liệu của Seabank, định hướng lại cho người dùng hoặc từ chối trả lời câu hỏi ngoài lĩnh vực. Nếu câu hỏi của người dùng có ý định truy vấn cơ sở dữ liệu của Seabank, hãy sử dụng lịch sử hội thoại và viết lại câu hỏi của người dùng thành một câu hỏi: 
        a. Không có thành phần thừa không liên quan đến ý định của người dùng.
        b. Sừ dụng thông tin từ các cuộc hội thoại gần đây để làm rõ ý định của người dùng
    
    2. Đầu ra: 

    In nội dung văn bản của đối tượng JSON với cấu trúc sau:
    {{
        "intent": "[RAG intent hoặc Non-RAG intent]",
        "question": "[Nội dung câu hỏi đã được viết lại hoặc null]",
        "prompt": "[Câu trả lời định hướng người dùng hoặc null]"
    }}

    - Giá trị phần tử intent chỉ có thể là: "RAG intent" hoặc "Non-RAG intent". Giá trị phần tử question là câu hỏi đã được viết lại của người dùng hoặc null nếu câu hỏi không có ý định. Giá trị phần tử prompt là câu trả lời định hướng người dùng hoặc null nếu người dùng có ý định.

    Câu hỏi: "{question}"

    """

    if os.path.exists(history_path) and os.path.getsize(history_path) == 0:
        prompt = starter_prompt
    else: 
        prompt = continuous_prompt
    
    try:
        model = vertexai.generative_models.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(**config),
        )
        output = "\n".join(x for x in response.text.splitlines() if "```" not in x)
        parse_response = json.loads(output)

        if (parse_response.get("intent") == "RAG intent"):
            conversation_history.append({"user": question, "assistant": output})
            save_conversation_history(conversation_history)

        return parse_response

    except Exception as e:
        error_message = f"Lỗi khi xác định intent: {e}"
        logging.error(error_message)
        raise 

print(generate_intent_from_question("Cho tôi hỏi bảng C có những trường nào", vertexai = vertexai))
