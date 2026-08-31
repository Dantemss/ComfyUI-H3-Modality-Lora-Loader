# ComfyUI-H3-Modality-Lora-Loader

## Purpose

This node allows you to load up to 10 MiniMax H3 LoRAs in a single batch.  
For each batch, you can select if the LoRAs should affect audio, video, text (token refiner and conditioning), or any combination of them.


This way you can prevent video-only LoRAs from affecting audio, and prevent audio-only LoRAs from affecting video.  
Note that there's still a bit of an effect due to self-attention but it should be much smaller than the usual effect.

## Installation

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/Dantemss/ComfyUI-H3-Modality-Lora_Loader.git
```
Or find ComfyUI-H3-Modality-Lora_Loader in ComfyUI Manager.

## Performance Impact

Batching provides better performance compared to single-LoRA nodes in a vacuum, although the modality filtering costs some performance.  
Each node adds 2 matrix multiplications per module affected by the LoRA, up to 528 on the stock model, which exposes 264 modules.

## Known Issues

Drag and drop has several bugs:
- wrong row is dragged
- inaccurate drop position
- rows drop above the add lora button
- drop indicator is not showing
- rows disabled after dropping

LoRA refresh button is not working. "None" entries may appear in the LoRA list after clicking it.

The node uses masks that, while small, could cause OOM due to VRAM fragmentation, which will either manifest as an OOM error or as severe slowdown.  
This seems to be most relevant if you keep changing the LoRA strengths.  
Unloading models and clearing the node cache, or simply restarting ComfyUI may be required from time to time.  
Running ComfyUI with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` may or may not help a little bit.


## Notes

Unknown LoRA targets are skipped with a warning message. Let me know if I missed any.

## Unit Testing

```sh
python -m pytest --import-mode=importlib
```

## Acknowledgements

UI based on the excellent LoRA Loader Stack node by PlagueKind:  
https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes
