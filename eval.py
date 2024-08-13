from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
import json
from tqdm import tqdm
import copy


def read_raw_data(file_path):
    raw_data = []
    with open(file_path, 'r') as f:
        json_list = list(f)
    for json_str in json_list:
        raw_data.append(json.loads(json_str))
    return raw_data

def split_one_chat_as_two(raw_dataset):
    chat_mul2 = []
    for item in raw_dataset:
        # a list of 5 dicts
        chat_list = item["messages"]
        assert len(chat_list) == 5, "length should be 5"
        chat_1 = chat_list[:3]
        chat_2 = chat_list
        chat_mul2.append({"messages": chat_1})
        chat_mul2.append({"messages": chat_2})
    assert len(chat_mul2) == 2 * len(raw_dataset)
    return chat_mul2


def format_dataset(raw_dataset, fmt_tokenizer):
    message_list_infer = []
    message_list_label = []
    message_list_trct = []
    for entry in raw_dataset:
        message_list_infer.append(entry["messages"][:-1])     # mask the last assistant response as label
        message_list_label.append(entry["messages"][-1])
        truncate_chat = [entry["messages"][0], entry["messages"][3]]  # sys + last user
        message_list_trct.append(truncate_chat)
    dataset = Dataset.from_dict({"complete_chat": message_list_infer, "truncate_chat": message_list_trct, "label": message_list_label})
    dataset = dataset.map(lambda x: {"formatted_chat": fmt_tokenizer.apply_chat_template(x["truncate_chat"], tokenize=False, add_generation_prompt=True)})
    return dataset


if __name__ == "__main__":
    finetuned_path = "/222043014/ciphersafe/finetune_nips/ft_model/checkpoint-327"
    test_dataset_path = "/222043014/ciphersafe/finetune_nips/fine-tune/test.jsonl"
    llama3_fmt_tokenizer_path = "/222043014/ciphersafe/llms/Meta-Llama-3-8B-Instruct"

    llama3_ft_model = AutoModelForCausalLM.from_pretrained(finetuned_path, device_map='auto')
    llama3_tokenizer = AutoTokenizer.from_pretrained(finetuned_path)
    llama3_format_tokenizer = AutoTokenizer.from_pretrained(llama3_fmt_tokenizer_path)

    raw_data = read_raw_data(file_path=test_dataset_path)
    # eval_data = split_one_chat_as_two(raw_data)
    eval_data_formatted = format_dataset(raw_data, llama3_format_tokenizer)
    print(eval_data_formatted)

    # # ret_list = []

    ret_list = []
    for sample in tqdm(eval_data_formatted["formatted_chat"][:], desc='Infering answers: '):
        inputs = llama3_tokenizer(sample, return_tensors="pt")
        inputs = inputs.to("cuda")
        # print(inputs)
        # break
        outputs = llama3_ft_model.generate(**inputs, max_new_tokens=256)[0]
        decoded_text = llama3_tokenizer.decode(outputs)
        gen_text = decoded_text.split("<|start_header_id|>assistant<|end_header_id|>")[1].strip()
        ret_list.append(gen_text)
        # print(decoded_text)
    with open("test_answers_with_context_trct.json", "w") as f:
        json.dump(ret_list, f, indent=2)
    eval_data_formatted.to_json("test_data_trct.jsonl")
    
    # default_sit = raw_data[0]["messages"][:2]
    # default_prompt = llama3_format_tokenizer.apply_chat_template(default_sit, add_generation_prompt=True, tokenize=False)
    # # print(default_prompt)
    # ret_list = []
    # for iter in tqdm(range(100)):
    #     inputs = llama3_tokenizer(default_prompt, return_tensors="pt")
    #     inputs = inputs.to("cuda")
    #     outputs = llama3_ft_model.generate(**inputs, max_new_tokens=256)[0]
    #     decoded_text = llama3_tokenizer.decode(outputs)
    #     gen_text = decoded_text.split("<|start_header_id|>assistant<|end_header_id|>")[1].strip()
    #     ret_list.append(gen_text)
    # with open("test_answers_wo_context.json", "w") as f:
    #     json.dump(ret_list, f, indent=2)