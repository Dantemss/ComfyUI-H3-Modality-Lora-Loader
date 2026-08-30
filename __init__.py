from .nodes.h3_modality_lora_loader import H3ModalityLoraLoader

NODE_CLASS_MAPPINGS = {
    "H3ModalityLoraLoader": H3ModalityLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ModalityLoraLoader": "H3 Modality LoRA Loader",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
