### Effi-Net-Classi
I just implemented PyTorch based classification problem with the EfficientNet V2 model.

- [X] Download script for CIFAR-10 dataset from torchvision.dataset package.
- [X] Train/Fine-tuning script with EfficinetNet_v2 PyTorch vision model.
- [ ] Evaluation script for fine tuned efficientnet_v2 model.
  - [X] Accuracy
  - [ ] Precision
  - [ ] Recall
  - [ ] Confusion Matrix
- [ ] Inference script for fine tuned efficientnet_v2 model.

> Input image shape should be the same as the models input else getting **low accuracy in prediction**. [**IMP**]

### Things to Note:
1. While loading fine-tuned paramteres to the model: (Keep these points in mind while loading the paramters).
   1. First modify the models as necessary (like output tensor size).
   2. Second load the state_dict() from the checkpoints.
   3. Both should be done in **cpu** itself. After that change the device to **GPU**. [**IMP**] 
   4. while inferencing do not need to load the pretrained weights (**IMP**) -> check `eval.py`