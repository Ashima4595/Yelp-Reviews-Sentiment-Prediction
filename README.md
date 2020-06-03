# Yelp Reviews Sentiment Predictor 

The goal of this project is to train a binary classifier to predict text sentiment for positivity or negativity. The project uses techniques like text cleaning, text processing, sentiment analysis and machine learning modeling. 

## Getting Started

### Dependencies
Project is run using following system specifications and dependencies:
- Windows 10
- Python 3.8.2
- Pandas
- Numpy
- NLTK
- sklearn

### Inside the Repository
The repository contains all the files required to run this project. These files are used for training and testing the model which can be further used for predicting the sentiment of textual information.
- Training File - training_file.txt : This file contains training data which consists of textual information. This text contains yelp reviews along with the star rating for the reviews.
- Training Label - training_label_file.txt : This file contains training lables. The training labels contain 1 and 0 which represent positive review and negative review respectively based on the textual information stored in the training file. 
- Vector Model - sentiment_analyzer_count_vector.pickle : This file contains Count Vectorizer model to create vectors for textual data to feed into machine learning model while testing. 
- Classifier Model - sentiment_analyzer_gbm.pickle : This file contain GBM model to classify textual information into positive and negative sentiment while testing.
- Testing file- sample_testing.txt : This file contains sample text for testing the model.
- Testing results - output.txt : This file contains prediction results for sample testing data.
- Jupyter Notebook - Yelp_Reviews_Sentiment_Predictor.ipynb : This notebook contains implementation of this project. It includes data processing, data cleaning, sentiment scoring and machine learning modeling steps.
- Script- sentiment_predictor.py : This python file contains code for training and testing the model for sentiment analysis.

## Training the Model
I have already trained all the models and stored the weights required for testing in the repository.
But you can use the script in the "training" mode to train the models with your training data, make sure it follows the format of the "training_file.txt" and your labels the format of ""training_label_file.txt"".
```
# First Input to the function = Mode: training
# Second Input to the function = File name containing training data : training_file.txt
# Third Input to the function = File name containing labels of the training data : training_label_file.txt
# Example:
sentiment_predictor.py 'training' training_file.txt training_label_file.txt
```

## Testing the Model
The repository contains one testing sample file along with the weights of the model.
You can call the function using following lines of code.

```
# First Input to the function = Mode: testing
# Second Input to the function = File name containing testing data : sample_testing.txt
# Example:
sentiment_predictor.py testing sample_testing.txt
```
Output:
Output File Sample: The output consists of the predictions for the testing data(1 represents Positive Sentiment, 0 represents Negative Sentiment).
```
1
1
1
1
1
1
1
1
1
1
1
1
1
1
0
```

### Additional Notes
 The dataset for this project was taken from [Kaggle](https://www.kaggle.com/yelp-dataset/yelp-dataset). If you have anything you'd like to discuss about the project, feel free to contact me at ashimamunjal04@gmail.com.