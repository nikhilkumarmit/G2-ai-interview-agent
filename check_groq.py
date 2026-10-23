import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "Reply with exactly: Groq connection successful"}
    ],
)

print(response.choices[0].message.content)
