# Final project for post-graduate class Open Source Linux Software(WHU)
## Introduction
We have built a novel project of machine learning. Specificly it belongs to sentence classification, which is a substaintial and key task in 
many Natural Language Processing(NLP), such as sentiment analysis, text entailment and entity-relation extraction. <br><br>
We utilized two kinds of dataset as data source to be classified. One is made by ourselves and they're sentences which are extracted from 
listed company annoucements by keywords and may express relation among indutries, the other is a public dataset for sentiment classification published by Yelp.<br><br>
The rest of the README will show you some main parts of the project in detail.
## Data source
The sentences to be classified consist of two types. Firstly, we used sentences extracted from <b>listed company announments</b> by some self-defined for relation classification. Secondly, we downloaded <b>Yelp</b> reviews which are comments of some restaurants from costomers for sentiment classification.
## Data preprocessing
In order to extract sentences we needed for classification, the first thing we did was data preprocessing. <br><br>
For sentences from list company announcements, we read each announment files and split a document into many sentence, then, we traversed 
each sentence and judged whether it contained keyword. We wrote those containing keywords into files. After that, we also need to label the 
sentences according to whether they really express industry relation. which is time-costing but unavoidable work in supervised learning.<br><br>
For sentences from Yelp, we just need to extract them and their labels from json. And write the sentences into files with their labels in right order. It's more convenient compared with data above.
## Word embedding
One of the most important procedures in classification is how to represent features of samples. In NLP, word embedding is a revolutionary tool for text representation as each word can be distributed a vector which is embedded word semantic.<br><br>
For sentences of relation classification(self-made), we used all the sentences to train word embedding with Python gensim. <br>
For sentences of sentiment classification(public), we used pre-trained word embedding of <b>Glove</b>. <br><br>
The first dataset is Chinese dataset, and there is rarely public pre-trained Chinese word embedding, so we needed to train it by ourselves. 
By the way, the quality of word embedding trained by our own text may be not vary satisifying and adaptable as scale of our text is limited. But it really convenient and fast.
However, there're several public pre-trained English word embeddings, such as Glove, FastText. We used Glove here.
## Sentence feature vector
After we got word embedding, we needed to take into consideration how to construct sentence feature vectors based on word embedding. Here we used a simple but effetive way to do it.<br><br>
Considering a sentence contains n words, we sum up all the vectors of n words with different weights. The weights can be TF-IDF or calculated based on word frequence. The reason why we represented sentences like this is based on the hypothesis that word embedding can represent word semantic and the sum of them can also embed sentence semantic in a way. And weight of each word is ralated to the importance of the word in the sentence, so we can try to increase effect of key words for sentence feature vectors.

## Training and testing
We splitted sentence vectors into training set and testing set with proportion 4 : 1 using cross validation. We utilized precision, recall and f1-score as metric. 
We chosed several popular and useful classifiers: <b>Logistic, SVM, Random Forest, Decision Tree</b> and <b>Extra Tree</b> to do classification on two dataset. Conveniently, Python sklearn provides all we need.
## Results
The results of our classification is as follow:(Here we only show <b>accuracy</b> for simplification)<br><br>
&emsp;&emsp;&emsp;&emsp;&emsp;LR&emsp;&emsp;&emsp;&emsp;&emsp;SVM&emsp;&emsp;&emsp;&emsp;&emsp;DT&emsp;&emsp;&emsp;&emsp;&emsp;RF&emsp;&emsp;&emsp;&emsp;&emsp;ET<br>
Dataset1&emsp;0.8815&emsp;&emsp;&emsp;0.9120&emsp;&emsp;&emsp;0.9294&emsp;&emsp;&emsp;0.9314&emsp;&emsp;&emsp;0.9353<br>
Dataset2&emsp;0.6083&emsp;&emsp;&emsp;0.5692&emsp;&emsp;&emsp;0.4232&emsp;&emsp;&emsp;0.5223&emsp;&emsp;&emsp;0.5137<br>
## Acknowledgement
This project belongs to <b>Wuhan University Open-source Software Engineering</b>
（该项目属于武汉大学开源软件工程项目）
