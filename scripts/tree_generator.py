import os
import fnmatch

def ignore_pattern():
    return [
        ".git",
        "venv/",
        ".env",
        "__pycache__/",
        "logs/",
        ".ipynb_checkpoints/",
        "*.ipynb_checkpoints",
        ".vscode/",
        ".idea/",
        "*.swp",
        "*.swo",
        "final_model/",
    ]


def should_ignore(name, path, patterns):
    for pattern in patterns:
        # Directory pattern
        if pattern.endswith("/"):
            if os.path.isdir(path) and fnmatch.fnmatch(name, pattern[:-1]):
                return True

        # File or wildcard pattern
        elif fnmatch.fnmatch(name, pattern):
            return True

    return False


def generate_tree(
    start_path,
    prefix="",
    level=0,
    max_depth=3,
    patterns=None
):
    if patterns is None:
        patterns = []

    if level > max_depth:
        return

    try:
        entries = os.listdir(start_path)
    except PermissionError:
        return

    # Apply gitignore-like filtering
    filtered = []
    for entry in entries:
        path = os.path.join(start_path, entry)
        if not should_ignore(entry, path, patterns):
            filtered.append(entry)

    # Sort: directories first
    filtered.sort(
        key=lambda x: (not os.path.isdir(os.path.join(start_path, x)), x.lower())
    )

    total = len(filtered)

    for index, entry in enumerate(filtered):
        path = os.path.join(start_path, entry)
        connector = "├── " if index < total - 1 else "└── "
        print(prefix + connector + entry)

        if os.path.isdir(path):
            extension = "│   " if index < total - 1 else "    "
            generate_tree(
                path,
                prefix + extension,
                level + 1,
                max_depth,
                patterns
            )


if __name__ == "__main__":
    ROOT = "."
    MAX_DEPTH = 3

    patterns = ignore_pattern()

    print(ROOT)
    generate_tree(ROOT, max_depth=MAX_DEPTH, patterns=patterns)