from google import genai

client = genai.Client(api_key = "AIzaSyCX9gc40pdxln7jzbtc0RRlmHpQXNFlJ7s")

print("List of models that support generateContent:\n")
for m in client.models.list():
    for action in m.supported_actions:
        if action == "generateContent":
            print(m.name)

print("List of models that support embedContent:\n")
for m in client.models.list():
    for action in m.supported_actions:
        if action == "embedContent":
            print(m.name)