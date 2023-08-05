import openai

openai.api_key = "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"

conversation_history = []

while True:
    user_input = input("Player: ")
    conversation_history.append(f"Player: {user_input}")

    prompt = "\n".join(conversation_history)

    response = openai.Completion.create(
        engine="text-davinci-003",  # You can use the GPT-3 engine
        prompt=prompt,
        max_tokens=50
    )

    npc_response = response.choices[0].text.strip()
    print("NPC:", npc_response)

    conversation_history.append(f"NPC: {npc_response}")
