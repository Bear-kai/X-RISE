import os
import shutil
from pathlib import Path

def replace_files():
    print('current directory:', os.getcwd())
    source_dir = Path("third/robotwin")
    target_dir = Path("../..")

    for root, dirs, files in os.walk(source_dir):
        rel_path = Path(root).relative_to(source_dir)
        target_root = target_dir / rel_path
        for file in files:
            source_file = Path(root) / file
            target_file = target_root / file
            shutil.copy2(source_file, target_file)
            print(f"replace: {source_file} -> {target_file}")


if __name__ == "__main__":
    replace_files()
    print("Replace files done.")