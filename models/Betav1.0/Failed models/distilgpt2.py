generator = pipeline("text-generation", model="distilgpt2", device=mps_device)  # -1 CPU, 0 GPU

results = generator(
    "বারান্দায় দীঁড়িয়ে সামিয়া ও নয়ন কথা বলছে। সামিয়া: আগামী শুক্রবার আমার ছোটো বোনের জন্মদিন। তুমি এদিন বিকেলে",
    max_new_tokens=30,  # Generate up to 30 new tokens (excluding input length)
    num_return_sequences=2,
    truncation=True,
    pad_token_id=50256  # Explicit padding token
)

for i, result in enumerate(results):
    print(f"Generated Text {i+1}: {result['generated_text']}")