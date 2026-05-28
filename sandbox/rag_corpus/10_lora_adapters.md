# Low-Rank Adaptation (LoRA)

LoRA freezes pretrained weights and injects trainable low-rank factors into selected linear layers. For weight matrix W, the update ΔW = BA uses small rank r, updating only adapter parameters during fine-tuning.
