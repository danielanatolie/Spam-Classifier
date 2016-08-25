# Naive Bayes Spam Classifier 
import os
import io 
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)
    return DataFrame(rows, index=index)


# Function returns message path and message contents
def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message
            
# Creating an empty Panda dictionary containing two empty lists
# message: text of email | class: spam or not
data = DataFrame({'message': [], 'class': []})

# Appending already marked spam emails:
data = data.append(dataFrameFromDirectory('/Users/DanLam/Dropbox/ridentcode/DataScience/DataScience/emails/spam', 'spam'))
# Appending already marked regular (ham) emails:
data = data.append(dataFrameFromDirectory('/Users/DanLam/Dropbox/ridentcode/DataScience/DataScience/emails/ham', 'ham'))

print data.head()


vectorizer = CountVectorizer()
# User vectorizer.fit to take all of the 'message' contents and convert them into numbers and how many times each word occurs
counts = vectorizer.fit_transform(data['message'].values)
classifier = MultinomialNB()

#targets is the classification type of each email
targets = data['class'].values

# Create a model using Naive Bayes based on the current data provided
classifier.fit(counts, targets) 

#TEST 1: 
testOne = ['Free Viagra now!!!','UBC Department of Computer Science']
testOne_counts = vectorizer.transform(testOne)
predictions = classifier.predict(testOne_counts)
print predictions 

