#!/usr/bin/env python
import json
import sys
import os


def clean_notebook(notebook_path):
    """Clean a notebook by removing widget state data that causes issues with mkdocs-jupyter."""
    print(f"Cleaning notebook: {notebook_path}")

    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Make a backup
    backup_path = f"{notebook_path}.bak"
    if not os.path.exists(backup_path):
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, ensure_ascii=False, indent=2)
        print(f"Created backup at: {backup_path}")

    # Clean widget data from metadata
    cleaned = False
    for cell in notebook.get("cells", []):
        if "outputs" in cell:
            for output in cell["outputs"]:
                if (
                    "data" in output
                    and "application/vnd.jupyter.widget-view+json" in output["data"]
                ):
                    # Remove problematic widget view
                    output["data"].pop("application/vnd.jupyter.widget-view+json", None)
                    cleaned = True

        # Clean cell metadata
        if "metadata" in cell and "widgets" in cell["metadata"]:
            cell["metadata"].pop("widgets", None)
            cleaned = True

    # Clean notebook metadata
    if "metadata" in notebook:
        if "widgets" in notebook["metadata"]:
            notebook["metadata"].pop("widgets", None)
            cleaned = True

    if cleaned:
        # Save the cleaned notebook
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, ensure_ascii=False, indent=2)
        print(f"Cleaned and saved: {notebook_path}")
        return True
    else:
        print(f"No widget data found in: {notebook_path}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path) and path.endswith(".ipynb"):
            clean_notebook(path)
        elif os.path.isdir(path):
            # Clean all notebooks in the directory
            count = 0
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".ipynb"):
                        notebook_path = os.path.join(root, file)
                        if clean_notebook(notebook_path):
                            count += 1
            print(f"Cleaned {count} notebooks in {path}")
        else:
            print(f"Error: {path} is not a valid notebook or directory")
    else:
        print("Usage: python clean_notebook.py <notebook_path_or_directory>")
