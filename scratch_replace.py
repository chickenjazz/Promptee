import os

replacements = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B": "Qwen2.5-3B",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen2.5-1.5B-Instruct": "Qwen2.5-3B-Instruct",
    "Qwen2.5-1.5B": "Qwen2.5-3B",
    "Qwen-Instruct-7B": "Qwen-Instruct-3B"
}

skip_dirs = {".git", ".venv", "node_modules", ".next"}

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        new_content = content
        for old_str, new_str in replacements.items():
            new_content = new_content.replace(old_str, new_str)
            
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated: {filepath}")
    except Exception as e:
        print(f"Error {filepath}: {e}")

if __name__ == "__main__":
    search_dirs = ["architecture", "dataset_builder", "Frontend", "rules", "tools", "training", "tests"]
    root_files = ["README.md", "findings.md", "task_plan.md", "read-training-dataset-builder-py-as-a-velvet-wirth.md"]
    
    for filepath in root_files:
        if os.path.exists(filepath):
            process_file(filepath)
            
    for directory in search_dirs:
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for file in files:
                if file.endswith((".py", ".md", ".tsx", ".ts")):
                    process_file(os.path.join(root, file))
