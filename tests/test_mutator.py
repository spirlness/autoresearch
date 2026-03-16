from autoresearch_trainer.mutator import mutate_config

def test_mutate_config(tmp_path):
    config_file = tmp_path / "config.py"
    content = """
EMBEDDING_LR = 0.6
MATRIX_LR = 0.04
DEPTH = 8
"""
    config_file.write_text(content)
    
    # Mutate
    mutate_config(str(config_file), {"EMBEDDING_LR": 0.5, "DEPTH": 12})
    
    new_content = config_file.read_text()
    assert "EMBEDDING_LR = 0.5" in new_content
    assert "DEPTH = 12" in new_content
    assert "MATRIX_LR = 0.04" in new_content
    assert "EMBEDDING_LR = 0.6" not in new_content

def test_mutate_config_type_hint(tmp_path):
    config_file = tmp_path / "config.py"
    content = "LR: float = 0.1"
    config_file.write_text(content)
    
    mutate_config(str(config_file), {"LR": 0.2})
    
    new_content = config_file.read_text()
    assert "LR: float = 0.2" in new_content

def test_mutate_config_float_formatting(tmp_path):
    config_file = tmp_path / "config.py"
    content = "LR = 0.1"
    config_file.write_text(content)
    
    mutate_config(str(config_file), {"LR": 0.0001})
    
    new_content = config_file.read_text()
    assert "LR = 0.0001" in new_content
