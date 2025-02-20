### Effi-Net-Classi
I just implemented PyTorch based classification problem with the EfficientNet V2 model.

- [X] Download script for CIFAR-10 dataset from torchvision.dataset package.
- [X] Train/Fine-tuning script with EfficinetNet_v2 PyTorch vision model.
  - [ ] Script to stop training if no more parameters are learned.**IMP**
- [ ] Evaluation script for fine tuned efficientnet_v2 model.
  - [X] Accuracy
  - [ ] Precision
  - [ ] Recall
  - [ ] Confusion Matrix
- [X] Inference script for fine tuned efficientnet_v2 model.
  - [X] argparse based execution -> used `argparse` library.
  - [X] non-test data's are working.

> **V.IMP Note:** If the model is small then train the entire model's parameters; If the model is large then train the models by freezing few layers. **`The model should have sufficient parameters to train on`** 

> Input image shape should be the same as the models input else getting **low accuracy in prediction**. [**IMP**] **-> eg (224,224)** <br>
> Training the entire model without freezing the top layers **-> improves the accuracy and training**. [**IMP**]

### Things to Note:
1. While loading fine-tuned paramteres to the model: (Keep these points in mind while loading the paramters).
   1. First modify the models as necessary (like output tensor size).
   2. Second load the state_dict() from the checkpoints.
   3. Both should be done in **cpu** itself. After that change the device to **GPU**. [**IMP**] 
   4. while inferencing do not need to load the pretrained weights (**IMP**) -> check `eval.py`
2. Output layer must be modified based on the training dataset(based on number of classes) -> atleast in classifiction problem.
3. In inference the **input image shape** is important. (convert the desired shape before passing into the model for inference). [**V.IMP**] => **[batch_size, channels, height, weight]** -> this order