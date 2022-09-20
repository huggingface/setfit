import argparse
import json


def read_tsv(filename, max_num_lines):
    train_json_filename = filename.replace(".tsv", ".jsonl")

    with open(filename, 'r') as f_in, open(train_json_filename, 'w+') as f_out:
        f_in.readline()
        for idx, line in enumerate(f_in.readlines()):
            if idx < max_num_lines:
                tab_split = line.strip('\n').split('\t')
                dict_json = {"TEXT1": tab_split[0], "LBL": tab_split[1]}

                f_out.write(json.dumps(dict_json) + '\n')
            else:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filepath", required=True)
    parser.add_argument('-m', "--max_num_lines", type=int, default=32)
    args = parser.parse_args()

    read_tsv(args.filepath, args.max_num_lines)