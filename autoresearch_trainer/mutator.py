import re
import os
from typing import Dict, Any

def mutate_config(file_path: str, mutations: Dict[str, Any]) -> bool:
    """Mutate global constants in a python file using regex."""
    if not os.path.exists(file_path):
        return False
        
    with open(file_path, "r") as f:
        content = f.read()
        
    new_content = content
    for key, value in mutations.items():
        # Match "KEY = value" or "KEY: type = value"
        # Handles numbers, strings, tuples
        pattern = rf"^({key}\s*(?::\s*[\w\[\], ]+)?\s*=\s*).*?$"
        
        # Replacement value formatting
        if isinstance(value, str):
            val_str = f'"{value}"'
        else:
            val_str = str(value)
            
        new_content = re.sub(pattern, rf"\g<1>{val_str}", new_content, flags=re.MULTILINE)
        
    if new_content != content:
        with open(file_path, "w") as f:
            f.write(new_content)
        return True
        
    return False
