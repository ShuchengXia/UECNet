# UECNet: UECNet: A Unified Framework for Exposure Correction Utilizing Region-Level Prompts
The implementation of the paper "UECNet: A Unified Framework for Exposure Correction Utilizing Region-Level Prompts"
## Abstract ##
>In real-world scenarios, complex illumination often causes improper exposure in images. Most existing correction methods assume uniform exposure degradation across the entire image, leading to suboptimal performance when multiple exposure degradations coexist in a single image. To address this limitation, we propose UECNet, a Unified Exposure Correction Network guided by region-level prompts. Specifically, we first derive five degradation-specific text prompts through prompt tuning. These prompts are fed into our Exposure Prompts Generation (EPG) module, which generates spatially adaptive, regionlevel descriptors to characterize local exposure properties. To effectively integrate these region-specific descriptors into the exposure correction pipeline, we design a Prompt-guided Token Mixer (PTM) module. The PTM enables global interactive modeling between highdimensional visual features and region-level prompts, thereby dynamically steering the correction process. UECNet is built by incorporating EPG and PTM into a U-shaped Transformer backbone. Furthermore, we introduce SICE-DE (SICE-based Diverse Exposure), a new benchmark dataset reorganized from the well-known SICE dataset, to facilitate effective training and comprehensive evaluation. SICE-DE covers six distinct exposure conditions, including challenging severe over/underexposure and non-uniform exposure. Extensive experiments demonstrate that the proposed UECNet consistently outperforms state-of-the-art methods on multiple exposure correction benchmarks.

## SICE-DE Dataset ##
The SICE-DE test dataset can be download [here](https://drive.google.com/drive/folders/1fYRR9Tt58tMsqJx50w28pIEfakm5oflv?usp=drive_link)
(We will make all the SICE-DE datasets available after the paper is accepted.)

## Test ##
Run the test.py to get the restored images and evaluation values.

The pretrained model weights can be download [here](https://drive.google.com/drive/folders/1WwDLiWVj0lWYLoXp-79czSdmvqr2L-Ev?usp=drive_link) 

## Citation ##
If you find our repo useful for your research, please consider citing this paper.

