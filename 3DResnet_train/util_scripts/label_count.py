import json
import collections

def label_count(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
        labels = data["labels"]
        counter = collections.Counter()
        data_base = data["database"]
        for key, item in data_base.items():
            annotations = item["annotations"]
            label = annotations["label"]
            counter[label] += 1
        for label in labels:
            print(label, counter[label])
        max_count = max(counter.values())
        weights = [max_count / counter[label] for label in labels]
        print(",".join([f"{w:.2f}" for w in weights]))

if __name__ == "__main__":
    import sys
    label_count(sys.argv[1])