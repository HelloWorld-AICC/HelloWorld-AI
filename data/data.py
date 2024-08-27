
import json
import os
import torch
from torch.utils.data import Dataset

# JSON 파일에 데이터를 한 줄씩 추가하는 함수
def save_to_json_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


class CustomDataset(Dataset):
    def __init__(self, tokenizer):
        IGNORE_INDEX=-100
        self.inp = []

        PROMPT = '''당신은 유능한 AI 어시스턴트 입니다. [대화 내용]을 보고, [요약문]을 생성해주세요.\n'''

        with open("configs/summary_config.json", "r") as f:
            data = json.load(f)

        def make_chat(inp):
            chat = ["[대화 내용]"]

            for cvt in inp['conversation']:
                speaker = cvt['speaker']
                utterance = cvt['utterance']
                
                # 비어있는 문장 제거
                if len(utterance) == 0:
                    continue

                chat.append(f"{speaker}: {utterance}")
                
            chat = "\n".join(chat)

            question = f"[요약문]\n"
            chat = chat + "\n\n" + question

            return chat
        
        for example in data:
            chat = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            self.inp.append(source)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx]


class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
