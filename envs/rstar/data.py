from pathlib import Path
import jsonlines
from torch.utils.data import Dataset


def get_train_test_dataset(*args, **kwargs):
    env_dir = Path(__file__).parent.parent
    # test_ds = JsonlMathDataset(env_dir / "MATH/dataset/test10.jsonl")
    test_ds = JsonlMathDataset(env_dir / "MATH/dataset/test500.jsonl")
    train_ds = JsonlMathDataset(env_dir / "MATH/dataset/train.jsonl")
    return train_ds, test_ds


class JsonlMathDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = []
        with jsonlines.open(data_path, "r") as reader:
            for obj in reader:
                # 验证数据格式
                if self._validate_data_format(obj):
                    self.data.append(obj)
                else:
                    print(f"Warning: Skipping invalid data entry: {obj}")
    
    def _validate_data_format(self, obj):
        required_fields = ["problem", "answer"]
        return all(field in obj for field in required_fields)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return {"question": x["problem"], "answer": x["solution"]}
