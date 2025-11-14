# ProstateCancerPrognosis

## Environments and Requirements
Windows  
Anaconda  
GPU: RTX 3080Ti 12GB  
CUDA version: cu12.9  
python version: 3.13  

pytorch 2.8.0  
lifelines  
numpy  
torchsummary  
SimpleITK  
scipy  
json  
wandb: it is to monitor the training curves, users should sign in wandb and get their own API key  


## Dataset 
Data from the CHIMERA challenge  
[1]	Schouten, D., Spaans, R., Faryna, K., Khalili, N., Litjens, G., Dille, B., Chia, C., Vendittelli, P., & Zuiverloon, T. (2025). Combining HIstology, Medical imaging and molEcular data for medical pRognosis and diAgnosis (CHIMERA). Medical Image Computing and Computer Assisted Intervention 2025 (MICCAI). Zenodo.  


## Preprocessing  

1. Normalization MRI data  
2. Quantification of textual informaiton of Clinical Data
Please see the details in the report and the code Pre_processing.py


## Training  
To train the model in the report, run this code:  
Train.py  

You can download trained models via:  
1. This repository
2. Hugging Face  https://huggingface.co/sunyanming18mtl/ProstateCancerPrognosis


## Inference and evaluation  
To infer and evaluate the testing cases, run this code:  
Inference_Overall.py  


## Results  
Folds	| Fold 0	| Fold 1	| Fold 2	| Fold 3	| Fold 4	| Entire dataset  
| ----| --------|--------|---------|-------| -------|--------
C-index	| 0.86	| 0.67	| 0.94	| 0.63	| 0.78	| 0.77  









