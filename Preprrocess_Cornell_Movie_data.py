import os
import re


def load_lines(file_path):
    with open(file_path, encoding='iso-8859-1') as f:
        lines = f.readlines()
    id_to_line = {}
    for line in lines:
        parts = line.split(" +++$+++ ")
        line_id = parts[0].strip()
        text = parts[-1].strip()
        id_to_line[line_id] = text
    return id_to_line


def load_conversations(file_path, id_to_line):
    with open(file_path, encoding='iso-8859-1') as f:
        lines = f.readlines()
    conversations = []
    for line in lines:
        parts = line.split(" +++$+++ ")
        line_ids_str = parts[-1].strip()
        line_ids = re.findall(r"L\d+", line_ids_str)
        conversation = [id_to_line[line_id] for line_id in line_ids]
        conversations.append(conversation)
    return conversations


def save_conversations(conversations, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for conversation in conversations:
            for line in conversation:
                f.write(line + "\n")
            f.write("\n")


if __name__ == "__main__":
    output_path = "preprocessed_conversations.txt"

    lines_file = "movie_lines.txt"
    conversations_file = "movie_conversations.txt"

    id_to_line = load_lines(lines_file)
    conversations = load_conversations(conversations_file, id_to_line)
    save_conversations(conversations, output_path)
