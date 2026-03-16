import os
from autoresearch_trainer.mutator import mutate_config

config_file = "test_config_mutation.py"
content = """
EMBEDDING_LR = 0.6
DEPTH = 8
"""
with open(config_file, "w") as f:
    f.write(content)

print(f"Original content:\n{content}")

mutations = {"EMBEDDING_LR": 0.42, "DEPTH": 12}
print(f"Applying mutations: {mutations}")
mutate_config(config_file, mutations)

with open(config_file, "r") as f:
    new_content = f.read()

print(f"New content:\n{new_content}")

if "EMBEDDING_LR = 0.42" in new_content and "DEPTH = 12" in new_content:
    print("Mutation verification succeeded!")
else:
    print("Mutation verification failed!")

os.remove(config_file)
