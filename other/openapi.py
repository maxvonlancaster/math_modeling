from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key="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",
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5-turbo",
)