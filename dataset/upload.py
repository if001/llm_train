# from datasets import load_dataset
import glob
from datasets import Dataset, DatasetDict

def main():
    file_paths = glob.glob("./dataset/artemis/data*.txt")

    data = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            data.append({"text": content})

    ds = Dataset.from_list(data)
    for v in ds:
        print(v)
        print('-'*20)

    ds.push_to_hub('if001/artemis')


if __name__ == '__main__':
    main()