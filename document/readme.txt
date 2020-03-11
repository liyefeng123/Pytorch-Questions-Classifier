In the submited zip file, there are three folders:
-document: a document containing a description for each function, a README file instructing how to use the code.
-data: training, dev, test, configuration files (excluding word embeddings), and some extra files needed for our models
-src: our code - question_classifier.py

For Running 
1.In the terminal, go to the folder-src where the code is located 
for example: cd C:\Users\Alienware\PycharmProjects\TextMining-CW1\src

2.For training, run: % python question_classifier.py train -config [configuration_file_path]
Every model have one config file, you can change some parameters in config file. And remember to
write the right config file path for training the model you choose.
for example: python question_classifier.py train -config ../data/bow_pre_train.config

3.For testing, run % python question_classifier.py test -config [configuration_file_path]
Every model have one config file, remember to write the right config file path for testing the model you choose.
for example: python question_classifier.py test -config ../data/bow_pre_train.config

For config files
1.bilstm_pre_train.config -- is the config file for the bilstm model with pre trained weight. 
2.bilstm_random.config -- is the config file for the bilstm model with random word vector.
3.bow_pre_train.config -- is the config file for the bag of word model with pre trained weight.
4.bow_random.config -- is the config file for the bag of word model with random word vector. 
5.ensemble_pre_train.config -- is the config file for the model combining the sentence vector from bag of word model and bilstm model.





