# ComfyUI-H3-Modality-Lora-Loader

## Purpose

This node allows you to load up to 10 MiniMax H3 LoRAs in a single batch.  
For each batch, you can select if the LoRAs should affect audio, video, text (token refiner and conditioning), or any combination of them.


This way you can prevent video-only LoRAs from affecting audio, and prevent audio-only LoRAs from affecting video.  
Note that there's still a bit of an effect due to self-attention but it should be much smaller than the usual effect.

## Performance impact

Batching provides better performance compared to single-LoRA nodes in a vacuum, although the modality filtering costs some performance.  
Each node adds 2 matrix multiplications per module affected by the LoRA, up to 528 on the stock model, which exposes 264 modules.

The node uses masks that, while small, could cause OOM due to VRAM fragmentation.  
You can try running ComfyUI with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512` to reduce VRAM fragmentation.  
This seems to be most relevant if you keep changing the LoRA strengths.

## Installation

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/Dantemss/ComfyUI-H3-Modality-Lora_Loader.git
```

## Notes

Unknown LoRA targets are skipped with a warning message. Let me know if I missed any.

## Unit Testing

```sh
python -m pytest --import-mode=importlib
```

## Acknowledgements

UI Based on the excellent LoRA loader stack node by Plaguekind:  
https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes
