# HYPOTHESIS

In convolutional neural networks with a large number of epochs, cross-validation can be avoided, since it is possible to optimally change the parameters during training using the so-called effective learning rate.

# WHAT WAS DONE

An attempt was made to repeat the experiment [1] Samsung AI under the leadership of Dmitry Vetrov in “home” conditions. It was supposed to detect the convergence of the learning rate over 10,000 epochs to one "effective" value. For this, the effective learning rate was used, which is recalculated every epoch by dividing the initial learning rate by the current norm of the model weights. Thus, the lower the complexity of the model, the higher the learning rate, and vice versa. This approach was described by Twan van Laarhoven [2]. To implement the experiment, the CIFAR10 base dataset and the resnet18 standard model were chosen, no augmentation was carried out, L2 regularization was used. Training was carried out on 8 sets of hyperparameters for effective learning rate and on 8 combinations of optimizers & schedulars, in total: 16 sets of hyperparameters were trained for 10,000 epochs.

During the experiment, practice was gained in hypothesizing, experiment design, training neural networks on pytorch, as well as visualizing and analyzing data using matplotlib.

# INTERPRETATION OF THE RESULTS

The experiment showed that the values ​​of the effective learning rate do not converge to one optimal value; moreover, they slightly deviate from the initialized values. "Sawtooth" behavior was detected, as in the experiment [1], but only at high values ​​of the initial learning rate and a smaller L2-regularization parameter. Thus, the experiment showed that the method of optimizing the learning rate by dividing it by the weight norm does not have the expected advantage, the absence of the need for cross-validation, since it depends on the initial learning rate and the value of L2 regularization. It can be assumed that with a different model or dataset, the effect can still be detected, but this is the subject of study for other experiments.

However, it is worth noting that the accuracy on the test set was consistently higher than with randomly selected combinations of optimizer & schedular & lr & reg. This suggests that with limited resources, the effective learning rate can be useful and can be added to the validation parameter set. Also, in several experiments, an effect was detected, during which the accuracy on the test sample drops sharply, and after the fall it increases. The same effect was also shown in the webinar [1], however, it is not possible to derive practical significance in the general case, and this detection only shows the unusual nature of neural networks.


# LINKS

[1] Dmitry Vetrov, Webinar "Unusual properties of the loss function in deep learning. Dmitry Vetrov, Higher School of Economics. Part 1", link: https://clck.ru/sTjkH
[2] Twan van Laarhoven, paper: "L2 Regularization versus Batch and Weight Normalization", ref: https://arxiv.org/abs/1706.05350