import json


def read_jsonl_data(file_path):
    raw_data = []
    with open(file_path, 'r') as f:
        json_list = list(f)
    for json_str in json_list:
        raw_data.append(json.loads(json_str))
    return raw_data


def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def post_process(indexed_ans_list: list) -> list:
    ret_ans = []
    for ans in indexed_ans_list:
        if '<|eot_id|>' in ans:
            ret_ans.append(ans.split('<|eot_id|>')[0]) # truncate the output here
        else:
            ret_ans.append(ans)
    assert len(ret_ans) == len(indexed_ans_list)
    return ret_ans


def compare_to_label(indexed_ans_list, indexed_label_list) -> float:
    assert len(indexed_ans_list) == len(indexed_label_list), "Length not match"
    correct = 0
    unmatch = []
    for id in range(len(indexed_ans_list[:10])):
        print(indexed_ans_list[id])
        print('--')
        print(indexed_label_list[id])
        print('==')
        
        if indexed_ans_list[id] == indexed_label_list[id]:
            correct += 1
        else:
            unmatch.append(id+1)
    print("Total answer evaluated: ", len(indexed_ans_list))
    print("Accuracy: ", round(correct/len(indexed_ans_list), 2))
    print("Failure ids: ", unmatch)
    return unmatch


if __name__ == "__main__":
    test_data_file = '/222043014/ciphersafe/finetune_nips/fine-tune/test.jsonl'
    train_data_file = '/222043014/ciphersafe/finetune_nips/fine-tune/train.jsonl'

    # test_answer_ft = '/222043014/ciphersafe/finetune_nips/test_answers.json'
    test_data = read_jsonl_data(test_data_file)
    train_data = read_jsonl_data(train_data_file)

    count = 0
    for test_case in test_data:
        if test_case in train_data:
            count += 1
    print(count)
    # test_answer = read_json(test_answer_ft)
    # test_answer_pcd = post_process(test_answer)
    # test_answer_label = [item["label"]["content"] for item in test_data]
    # # compare_to_label(test_answer_pcd, test_answer_label)
    # print(test_data[321]["chat"])
    # print(test_data[55]["chat"])
    # print(test_data[321]["chat"] == test_data[55]["chat"])
    # print(test_data[321]["label"]["content"].replace('\n', ' '))
    # print(test_data[55]["label"]["content"].replace('\n', ' '))
    # print(test_data[321]["label"]["content"] == test_data[55]["label"]["content"])


    







