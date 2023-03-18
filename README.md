# Satellite_Image_Classification

Land Cover Classification using Deep Learning: In this project, we aim to classify satellite images into 10 different land cover categories using a deep learning model.

## Dataset

The dataset used in this project is the Sentinel-2 satellite images covering 13 spectral bands and consisting out of 10 classes with in total 27,000 labeled and geo-referenced images. It was was preprocessed using the ImageDataGenerator class from the keras library.
## Approach

The approach taken in this project involved 
- preprocess satellite imagery dataset with the ImageDataGenerator class from the keras library.
- training several machine learning models, including a **shallow CNN** and **VGG** classifier.
- models were evaluated primarily on **per-class accuracy**

## Results

Overall, **VGG** performed better achieving **validation accuracy** of 90% and **test accuracy** of 80% respectively.

![Confusion Matrix for VGG](https://user-images.githubusercontent.com/73485842/226115940-3e8645e4-407d-42fe-80aa-0c15d1eab1d0.png)

Shortly, let's explore the class with the highest misclassified images. 
From the confusion matrix above, 64 *PermanentCrop* images misclassified as *HerbaceousVegetation*.  Exploring these misclassified images (in the notebook), it is evident that the misclassified *PermanentCrop* lack in the quality and there is low variability from the *HerbaceousVegetation* images.


## Constraints

There are several constraints for this solution in terms of the data quality, data variability and vague requirements.

- **Data quality**: The images used in this project are of low resolution, which limits the ability of the models to achieve high accuracy. Additionally, it is important to note that the presence of clouds in the images can impact the performance of the models.

- **Data variability**: The classes with the lowest recall are *PermanentCrop* (0.69), *AnnualCrop*(0.88) and *Pasture* (0.84. Upon further examination of the images in these classes, it is evident that they belong to the agricultural umbrella class and may not exhibit significant variability.

- **Vague requirements**: Due to the lack of clear requirements, it is uncertain which architecture is best suited for completing the task, which class prediction to prioritize, and subsequently which evaluation metrics to use for the models.

## Improvements

There are several potential improvements that can be made to this solution. Looking at the per-class classification, the model seems to achieve poor performance on *PermanentCrop* and *AnnualCrop*. Here are some possible solutions that can be explored:

- Increase model complexity and fine-tune hyperparameters based on specific requirements.
- Use image augmentation techniques to up-sample images in the underrepresented classes.
- Integrate a new model that specializes in separating these classes, considering that they are of high importance.
- Train the dataset with the pre-trained models, i.e  ResNet-50

## Project Navigation
### Training
1. **Training** [Jupyter notebook]: [satellite_classification 2.ipynb](https://github.com/Mariner07/Satellite_Image_Classification/blob/main/satellite_classification%202.ipynb). Include all steps from loading the dataset, to saving the model and evaluation metrics.
2. **Training Dataset**: [2750](https://github.com/Mariner07/Satellite_Image_Classification/tree/main/2750)

### Inference (testing)
1. **Inference** [Jupyter notebook]:[inference folder](https://github.com/Mariner07/Satellite_Image_Classification/tree/main/inference) with respected  [inference 2.ipynb](https://github.com/Mariner07/Satellite_Image_Classification/blob/main/inference/inference%202.ipynb) 
2. **Test dataset**: [test_data](https://github.com/Mariner07/Satellite_Image_Classification/tree/main/inference/test_data)
3. **Test results** : [test_results_in_csv](https://github.com/Mariner07/Satellite_Image_Classification/blob/main/inference/classification_results.csv) 

### Saved models
1. **Saved models**: [saved models](https://github.com/Mariner07/Satellite_Image_Classification/tree/main/saved_models) with respected model, model weights, architecture and class names.
