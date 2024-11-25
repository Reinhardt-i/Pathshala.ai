import transformers


from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("asif00/mistral-bangla")
model = AutoModelForCausalLM.from_pretrained("asif00/mistral-bangla")


prompt = """Below is an instruction in Bengali language that describes a task, paired with an input also in Bengali language that provides further context. Write a response in Bengali language that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}
"""


def generate_response(question, context):
    inputs = tokenizer([prompt.format(question, context, "")], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    response_start = responses.find("### Response:") + len("### Response:")
    response = responses[response_start:].strip()
    return response


question = "ভারতীয় বাঙালি কথাসাহিত্যিক মহাশ্বেতা দেবীর মৃত্যু কবে হয় ?"
context = "২০১৬ সালের ২৩ জুলাই হৃদরোগে আক্রান্ত হয়ে মহাশ্বেতা দেবী কলকাতার বেল ভিউ ক্লিনিকে ভর্তি হন। সেই বছরই ২৮ জুলাই একাধিক অঙ্গ বিকল হয়ে তাঁর মৃত্যু ঘটে। তিনি মধুমেহ, সেপ্টিসেমিয়া ও মূত্র সংক্রমণ রোগেও ভুগছিলেন।"
answer = generate_response(question, context)
print(answer)