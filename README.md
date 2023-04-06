# AMLS_II_assignment22-23
## Description
-   Kaggle challenge: https://www.kaggle.com/competitions/cassava-leaf-disease-classification
-   5-class classification problem
-   21397 images (800*600)
-   The student try to compare the performance of 4 models (VGG16, VGG19, ResNet18 and ViT)

## Dataset
-   Download the dataset named "cassava-leaf-disease-classification" and put it in the folder "Datasets".

## Role of each file
-   main.py: main function that can activate 4 models
-   preprocess.py: perform preprocessing to the original dataset
-   VGG16.py: construct a VGG16 model
-   VGG19.py: construct a VGG19 model
-   ResNet.py: construct a ResNet18 model
-   ViT.py: construct a ViT model

## Packet required
-   The required environment has been export to file "env.yaml". Use the following conda instruction to finish the environment setting.
-   conda env create -f env.yaml