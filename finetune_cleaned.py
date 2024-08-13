from datasets import Dataset
import json
import torch.utils
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
from peft import LoraConfig
from accelerate import PartialState


def read_raw_data(file_path):
    raw_data = []
    with open(file_path, 'r') as f:
        json_list = list(f)
    for json_str in json_list:
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
    train_dataset_path = "/222043014/ciphersafe/finetune_nips/fine-tune/train.jsonl"
    formatting_tokenizer_path = "/222043014/ciphersafe/llms/Meta-Llama-3.1-8B-Instruct"
    llama3_model_path = "/222043014/ciphersafe/llms/Meta-Llama-3.1-8B"


    llama3_fmt_tokenizer = AutoTokenizer.from_pretrained(formatting_tokenizer_path)
    llama3_fmt_tokenizer.add_special_tokens({"pad_token": '<|finetune_right_pad_id|>'})
    llama3_fmt_tokenizer.padding_side = "right"

    raw_data = read_raw_data(file_path=train_dataset_path)
    ft_dataset = format_dataset(raw_data, llama3_fmt_tokenizer)

    print(ft_dataset)
    # assert len(ft_dataset) == 866

    # may need further investigation
    instruction_template = "<|start_header_id|>user<|end_header_id|>"
    response_template = "<|start_header_id|>assistant<|end_header_id|>"
    

    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template, 
        response_template=response_template, 
        tokenizer=llama3_fmt_tokenizer
    )

    target_mods = [
        "self_attn.q_proj",
        "self_attn.v_proj",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
        "mlp.up_proj"
    ]

    lora_config = LoraConfig(
        r=96, 
        lora_alpha=16, 
        lora_dropout=0.05, 
        target_modules=target_mods, 
        bias="none"
    )
    
    # start parallel
    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(llama3_model_path, device_map={'':device_string}, torch_dtype=torch.float16)
    model.resize_token_embeddings(len(llama3_fmt_tokenizer), pad_to_multiple_of=8)
    model.config.pad_token_id = llama3_fmt_tokenizer.pad_token_id

    sft_config = SFTConfig(
        output_dir='./ft_model_new',
        dataset_text_field="formatted_chat", 
        max_seq_length=1024,
        per_device_train_batch_size=2, 
        num_train_epochs=3, 
        fp16=True, 
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={'use_reentrant':False},
        ddp_find_unused_parameters=False
    )
    
    trainer = SFTTrainer(
        peft_config=lora_config,
        model=model,
        train_dataset=ft_dataset,
        args=sft_config,
        data_collator=collator
    )

    trainer.train()