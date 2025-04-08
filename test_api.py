from google import genai
import json

def intent_detect(question):

    client = genai.Client(api_key = "AIzaSyCX9gc40pdxln7jzbtc0RRlmHpQXNFlJ7s")

    user_role = "Khối Khách Hàng Cá Nhân"
    user_datamart_schema = "RETAIL_DTM"

    v_prompt = f'''
    Bạn là một trợ lý AI dữ liệu có khả năng hiểu Xử lý Ngôn ngữ Tự nhiên (NLU) và phát hiện ý định. Nhiệm vụ của bạn là phân tích đầu vào của người dùng. Người đặt câu hỏi là nhân viên {user_role} thuộc ngân hàng Seabank, mục tiêu là truy cập cơ sở dữ liệu datamart schema {user_datamart_schema} của Seabank.

    1. Xác định ý định của đầu vào và phân loại câu hỏi của người dùng: Xác định mục tiêu hoặc hành động của người dùng từ đầu vào. Nếu câu hỏi người dùng không có ý định truy vấn cơ sở dữ liệu của Seabank, trả lời rằng tôi sẽ không trả lời câu hỏi ngoài lĩnh vực.

    2. Đầu ra: 
    - Nếu câu hỏi liên quan đến lĩnh vực phân tích dữ liệu, in nội dung văn bản thô của đối tượng JSON với cấu trúc sau:    
    {{
        "intent": "RAG intent"
    }}

    - Nếu câu hỏi không liên quan đến lĩnh vực phân tích dữ liệu, in nội dung văn bản của đối tượng JSON với cấu trúc sau:
    {{
        "intent": "Non-RAG intent",
        "prompt": "[Câu trả lời định hướng người dùng]"
    }}

    - Giá trị phần tử intent chỉ có thể là: "RAG intent" hoặc "Non-RAG intent".

    Câu hỏi: "{question}"
'''

    # English version

#     prompt = f'''
#     You are an data AI assistant that understands Natural Language Understanding and intent detection. Your task is to analyze user input and extract structured information.

#     1. Identify the intent of the user input: Determine the user's goal or action from the input. If the user's question does not have an intent to query Seabank's database, do not extract parameters but respond that you will not answer questions outside the scope.
#     2. Extract parameters: identify specific details or entities relevant to the input. These can be explicit (directly mentioned), implicit (implied by context) or derived from other information.
#     3. Handle missing information: If any parameters are missing or unclear, ask for clarification from user with a concise, specific question.
#     4. Output: 
#     - If the question has an intent to query Seabank's database, print raw text content of a JSON object with the following structure:
#     {{
#         "intent": "[The indentified intent]",
#         "parameters": {{
#             "[parameter name 1]": "[extracted value 1]",
#             "[parameter name 2]": "[extracted value 2]",
#             // ... more if needed
#         }},
#         "missing_parameters": [
#             "[List any missing or unclear parameters]"
#             ],
#         "prompt": "[Clarification question for missing parameters, or empty string if none]"
#     }}
#     - Otherwise, print raw text content of a JSON object with the following structure:
#     {{
#         "intent": "[The identified intent]",
#         "prompt": "[Response to guide the user]"
#     }}

#     User input: "{question}"
# '''

    response = client.models.generate_content(
        model="gemini-2.5-pro-exp-03-25",
        contents=v_prompt,
    )

    # output = response.text
    output = "\n".join(x for x in response.text.splitlines() if "```" not in x)
    
    parse_output = json.loads(output)

    return parse_output
    
print(intent_detect("cho tôi thông tin về tình hình kinh doanh của BIDV"))