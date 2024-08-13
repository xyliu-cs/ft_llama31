import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset

def read_raw_data(file_path):
    raw_data = []
    with open(file_path, 'r') as f:
        json_list = list(f)
    for json_str in json_list:
        this_dict = json.loads(json_str)
        if this_dict not in raw_data:
            raw_data.append(json.loads(json_str))
    return raw_data

def format_dataset(raw_dataset, fmt_tokenizer):
    message_list = []
    for entry in raw_dataset:
        message_list.append(entry["messages"])
    dataset = Dataset.from_dict({"chat": message_list})
    dataset = dataset.map(lambda x: {"formatted_chat": fmt_tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})
    return dataset


if __name__ == "__main__":
    # data_list = read_raw_data('/222043014/ciphersafe/finetune_nips/fine-tune/train.jsonl')

    
    # formatting_tokenizer_path = "/222043014/ciphersafe/llms/Meta-Llama-3.1-8B-Instruct"

    # tokenizer = AutoTokenizer.from_pretrained(formatting_tokenizer_path)
    # tokenizer.add_special_tokens({"pad_token": '<|finetune_right_pad_id|>'})
    # tokenizer.padding_side = "right"

    # # model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    # train_dataset = format_dataset(data_list[:5], tokenizer)
    # instruction_template = "<|start_header_id|>user<|end_header_id|>"
    # response_template = "<|start_header_id|>assistant<|end_header_id|>"
    

    # collator = DataCollatorForCompletionOnlyLM(
    #     instruction_template=instruction_template, 
    #     response_template=response_template, 
    #     tokenizer=tokenizer
    # )


    # examples = [train_dataset["formatted_chat"][0]]
    # print(examples)
    # encodings = [tokenizer(e) for e in examples]

    # dataloader = DataLoader(encodings, collate_fn=collator, batch_size=1)
    # batch = next(iter(dataloader))
    # labels = batch["labels"][0]
    # not_masked_labels = [id for id in labels if id != -100]
    # print(batch["labels"])
    # converted_back_tokens = tokenizer.convert_ids_to_tokens(not_masked_labels)
    # print(converted_back_tokens)

    with open