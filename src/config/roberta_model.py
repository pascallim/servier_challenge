from transformers import RobertaConfig

def get_roberta_config(vocab_size: int):
    return RobertaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=130,
        hidden_size=192,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1
    )
