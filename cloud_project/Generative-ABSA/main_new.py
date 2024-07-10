import argparse
import os
import logging
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration

logger = logging.getLogger(__name__)


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='tasd', type=str, required=True,
                        help="The name of the task, selected from: [uabsa, aste, tasd, aope]")
    parser.add_argument("--dataset", default='rest15', type=str, required=True,
                        help="The name of the dataset, selected from: [laptop14, rest14, rest15, rest16]")
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--paradigm", default='extraction', type=str, required=True,
                        help="The way to construct target sentence, selected from: [annotation, extraction]")
    parser.add_argument("--input_file", default=None, type=str,
                        help="Path to the JSON file containing input texts")

    args = parser.parse_args()

    # Set up output directory
    output_dir = f"./outputs/{args.task}/{args.dataset}/{args.paradigm}"
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir

    return args


def read_texts_from_json(file_path):
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data['text']  # assuming 'text' is the key for each text data in JSON
            texts.append(text)
    return texts


def main():
    args = init_args()
    print("=" * 30, f"NEW EXP: {args.task.upper()} on {args.dataset}", "=" * 30)

    # Read texts from JSON file
    if args.input_file:
        texts = read_texts_from_json(args.input_file)
        print(f"Total examples = {len(texts)}")

        # Initialize tokenizer and model
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)

        # Process each text and generate outputs
        for text in texts:
            print(f"Input : {text}")
            inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
            outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4,
                                     early_stopping=True)
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Output: {decoded_output}")
            print()


if __name__ == "__main__":
    main()
