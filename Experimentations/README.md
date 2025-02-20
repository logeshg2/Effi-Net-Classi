## Experimentation models -> only classification

#### Same CIFAR-10 dataset

> `.\checkpoints\test3.pt` -> best **model** till now (84.36% accuracy in evaluation).

### New Implementation:
1. Batch-size of 256 is very effective.
2. Modular code is used where a train function is used -> check `engine.py`. 
3. I have put the evaluation script inside `engine.py` itself.
4. Removed the freezing of model layers -> which resulted in **better performance(training and accuracy)**.