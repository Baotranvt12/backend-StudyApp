from openai import OpenAI

openai = OpenAI(
    api_key="rt4TcoQRkgGx6YAlEPArqiVKJd1lPAxx",
    base_url="https://api.deepinfra.com/v1/openai"
)

messages = [
    {
        "role": "system",
        "content": "Bạn là cố vấn học tập."
    },
    {
        "role": "user",
        "content": "Tạo lộ trình học toán lớp 10 trong 4 tuần."
    }
]

try:
    response = openai.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",  # ✅ tên model đúng
        messages=messages,
        stream=False
    )

    print("✅ Kết quả:\n")
    print(response.choices[0].message.content)

except Exception as e:
    print("❌ Gọi GPT thất bại:", e)
