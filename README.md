# Deep-learning-enhanced digital twinning of complex composite structures
A framework of using Deep learning(DL) & Gaussian process regression(GPR) as a digital twin for complex composite materials.

## Introduction
- Prediction of deformation field and stress field for layered complex composites;
- Prediction with confidence interval through multitask Gaussian process regression and auto-coding technique;
- Almost real-time prediction capability making the method a possible digital twin of the in-service sealing structure;
- Multi-scale prediction of the macro contact force and micro deformation and stress field.

## Environment Requests
- python39
- pytorch (torch 1.12.1+cu116 used)
- [gpytorch](https://github.com/cornellius-gp/gpytorch) (1.22.1 used)

## Data Samples and pretrained models
The training data samples and the pre-trained models can be found in the [cloud Drive](https://drive.google.com/drive/folders/1CBYx2--yq1O6mtDW2fm0awCrN627bzU8?usp=share_link)

## Workflow
- ### Training
1. Download training samples from the [cloud drive](https://drive.google.com/drive/folders/1CBYx2--yq1O6mtDW2fm0awCrN627bzU8?usp=share_link).
2. Modify the code according to the directory of the training set and test set in ```COORD\train_coord.py``` or ```STRESS\train_stress.py```.
3. Run ```COORD\train_coord.py``` or ```STRESS\train_stress.py``` to train the auto-encoder or the DNN. 
4. Save Coord Code and Stress Code by modify ```save_code=True```  in ```COORD\train_coord.py``` or ```STRESS\train_stress.py``` to save the codes with a trained model.
5. Run ```STRESS\Continuum\C2S_Code.py``` to build code to code continuum law between coord code and stress code.
+ Note: The training process is optional. Pre-trained parameters are available in the [cloud drive](https://drive.google.com/drive/folders/1CBYx2--yq1O6mtDW2fm0awCrN627bzU8?usp=share_link)

- ### Evaluating
1. If you want only the predicted shape and the stress, just run ```COORD\train_coord.py``` or ```STRESS\train_stress.py``` and set ```display=True``` and running with a pre-trained model.
2. To evaluate a shape or stress field with a confidence interval, run ```COORD\GPR.py``` or ```STRESS\GPR.py```.


## Cite this work

