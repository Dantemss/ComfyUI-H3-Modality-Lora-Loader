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

## Performance impact in addition to base LoRA loader nodes

Batching provides better performance compared to single-LoRA nodes in a vacuum, although the modality filtering costs some performance.  
Each node adds 2 matrix multiplications per module affected by the LoRA per step, up to 528 per step on the stock model, which exposes 264 modules.

## Known issues

No known issues for now.

## Notes

Unknown LoRA targets are skipped with a warning message. Let me know if I missed any.

## Unit testing

```sh
python -m pytest --import-mode=importlib
```

## Acknowledgements

UI based on the excellent LoRA Loader Stack node by PlagueKind:  
https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes
