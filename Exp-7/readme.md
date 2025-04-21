## **Experiment 7**: WAP to retrain a pretrained ImageNet model to classify a medical image dataset (HAM10000 Dataset)

---

## **Model Description**

In this experiment, we retrain **DenseNet121**, a convolutional neural network that was originally trained on the ImageNet dataset. DenseNet121 uses dense connections where each layer receives input from all preceding layers, improving feature propagation and reducing redundancy.

We fine-tune this model to classify skin lesion images from the HAM10000 dataset into one of 7 categories.

---

## **Code Description**

The code follows these main steps:

1. **Data Preparation**

   - Reads the metadata CSV and maps image IDs to their file paths.
   - Filters out entries whose images do not exist on disk.
   - Adds numeric labels for classification.
   - Splits the dataset into training and validation sets using stratified sampling.
   - Applies preprocessing and data augmentation using `ImageDataGenerator`.

2. **Model Creation**

   - Loads the DenseNet121 model without the top layers (`include_top=False`) and with pretrained ImageNet weights.
   - Adds a global average pooling layer followed by a dense layer with softmax activation for 7-class classification.
   - Freezes the base model to train only the custom top layers initially.

3. **Training**

   - **Phase 1:** Trains only the top layers (base model frozen) for 5 epochs using Adam optimizer with learning rate `1e-3`.
   - **Phase 2:** Unfreezes all layers and fine-tunes the entire model for 10 more epochs with a lower learning rate (`1e-5`).

4. **Evaluation**

   - Combines training histories of both phases.
   - Plots training and validation accuracy/loss curves.
   - Predicts on the validation set.
   - Generates and visualizes a confusion matrix.
   - Prints classification report for performance metrics.
   - Displays sample validation images with true and predicted labels.

---

