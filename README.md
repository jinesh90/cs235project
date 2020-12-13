# cs235project
CS235 project for twitter sentiment analysis COVID-19 related tweets based on BERT deep neural network 

# How to run this project ?

To run this project, It needs some python packages to be pre-installed, It is recommended that this project should run on
python 3.6 or greater version. 

For training data and twitter data, please see folder to download appropriate files saved in S3.

If you want to train complete model, It is recommended you have GPU based machine 
with NVIDIA graphic card installed. CPU based machine will take longer time (~ 12-15 hours).

To install necessary packages, download `requirement.txt` file and run 
`pip install -r requirement.txt` this will install necessary packages.

afterwards, tune your model parameters from `config/parameters.py` file, by default it will take file's parameters as it is.

upon set up params, run main.py for training model, once model successfully trained, it will be saved as `bin/<model>.bin`


# Document Structure.
1. <h4> app </h4>
    this folder is application for commandline quick sentiment test as well as web application.to run standalone web application 
    
    `python app.py`
    
    after successfully application run, open browser and type http://localhost:5000, you will see text block. type any 
    text for finding sentiment, It loads pre-trained model and draw sentiment out of it [negative, neutral, positive]
 
2. <h4>bin</h4>
    folder where trained model saved. some of the trained model in s3. if you want to use those model download from here:
    `https://cs235project.s3.amazonaws.com/models/dropout_sentiment.bin`

3. <h4>config</h4>
   project configuration file. model hyperparams, data path and all kinds of constants defined here.It has various files
   based on different parameters.

4. <h4>data</h4>
   model training data, note here this data is tagged data from google playstore or yelp. where there are two main field.
   user reviews (in text format), user sentiment (0 for negative, 1 for neutral, 2 for positive.) This is supervised learning 
    so this data is used for model fine tuning and transfer learning. this should be in .csv file.

5. <h4>dataprocesssor</h4>
   this is folder has data preprocessing file.convert raw text data into deep learning model input object that compatible
   to BERT input layer.

6. </h4>evaluation</h4> 
   this folder has  model evaluation file. after training model, It will generate confusion matrix and get test data accuracy.

7. <h4>model</h4>
   various model architectures abstract class, like simple model, model with dropout layer or model with two deep neural layers etc.

8. <h4>twitter</h4>
   twitter raw tweets data related to covid19, after model training and fine tuning. tag sentiment on raw covid19 related
   tweets.
   
9. <h4>main.py</h4>
   main file for start training model.

10. <h4>builder.py</h4>
    supportive file for main.py
 
# Useful data links

- COVID19 related tweets data ( Full ).
    - `https://covid-data-for-project.s3.amazonaws.com/coviddata.zip`

- COVID19 tweets data samples.
    - `https://cs235project.s3.amazonaws.com/twitter_sentiment_data/2020-[mm]-clean.xlsx`
    
    replace mm to month digit. for example January replace 01. project contains data from January 2020- October 2020.
  
    
- COVID19 tagged tweet example by model.
    - `https://cs235project.s3.amazonaws.com/twitter_sentiment_tag/2020-[mm]-tagged.csv`
   