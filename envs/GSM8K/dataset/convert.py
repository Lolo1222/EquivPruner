import json

def convert_json_to_jsonl(input_file, output_file):
    """
    将 JSON 文件转换为 JSONL 文件，其中每行都是一个包含 "problem" 和 "answer" 字段的字典。

    Args:
        input_file (str): 输入 JSON 文件的路径。
        output_file (str): 输出 JSONL 文件的路径。
    """
    with open(input_file, 'r', encoding='utf-8') as f_in:
        data = json.load(f_in)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in data:
            output_item = {"problem": item["text"], "answer": item["label"]}
            json.dump(output_item, f_out, ensure_ascii=False)
            f_out.write('\n')

if __name__ == '__main__':
    input_file = 'envs/GSM8K/dataset/train.json'
    output_file = 'envs/GSM8K/dataset/train.jsonl'
    convert_json_to_jsonl(input_file, output_file)
    print(f"已将 {input_file} 转换为 {output_file}")