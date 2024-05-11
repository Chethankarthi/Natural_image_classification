This project focuses on building a deep learning model for image classification using the Intel Image Classification dataset. The dataset consists of images belonging to six different categories: buildings, forest, glacier, mountain, sea, and street.

Objective
The main objective of this project is to develop a deep learning model capable of accurately classifying images into their respective categories.

Dataset
The dataset used for this project is the Intel Image Classification dataset, which contains a total of 14,034 images divided into training and validation sets.

Training Set: 14,034 images
Validation Set: 3,000 images
Model Architecture
For this project, we utilized the ResNet-18 architecture, a pre-trained convolutional neural network (CNN) architecture, and fine-tuned it for our image classification task. The last fully connected layer of the ResNet-18 model was modified to output predictions for the six image categories present in the dataset.

Training
The model was trained for 50 epochs using the Adam optimizer with a learning rate of 0.001. During training, the loss was monitored to track the model's performance on the training set.

Evaluation
After training, the model was evaluated on the validation set to assess its performance. The accuracy, confusion matrix, and classification report were computed to evaluate the model's classification performance.

Results
Test Accuracy: 89.5%
Classification Report:
markdown
Copy code
              precision    recall  f1-score   support

   buildings       0.94      0.85      0.89       437
      forest       0.93      0.99      0.96       474
     glacier       0.83      0.87      0.85       553
    mountain       0.87      0.80      0.83       525
         sea       0.92      0.94      0.93       510
      street       0.90      0.92      0.91       501

    accuracy                           0.90      3000
   macro avg       0.90      0.90      0.90      3000
weighted avg       0.90      0.90      0.89      3000
Visualizations
We included visualizations such as a confusion matrix and sample images with their predicted and true labels to provide insights into the model's performance and predictions.

Model Saving
Finally, the trained model was saved as intel_image_classification_model.pth for future use or deployment.