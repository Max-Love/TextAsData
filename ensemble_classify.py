#Generated with assistance from copilot.
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score
from sklearn.svm import SVC
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from numpy import nan, mean, ravel

#Reduce the four label columns to a single label column with values 0-3.
def create_label(df):
    result = []
    count = 0
    for row in df.itertuples(index=True):
        found = False
        
        if str(row.ALIGNED) == 'X' or str(row.ALIGNED) == '1':
            if found:
                print(count)
                raise Exception("duplicate labeling")
            result.append(0)
            found = True
        if str(row.NOT_ALIGNED)  == 'X' or str(row.NOT_ALIGNED)  == '1':
            if found:
                print(count)
                raise Exception("duplicate labeling")
            result.append(1)
            found = True
        if str(row.NEUTRAL_IRRELEVANT)  == 'X' or str(row.NEUTRAL_IRRELEVANT)  == '1':
            if found:
                print(count)
                raise Exception("duplicate labeling")
            result.append(2)
            found = True
        if str(row.BORDER_CASE)  == 'X' or str(row.BORDER_CASE)  == '1':
            if found:
                print(count)
                raise Exception("duplicate labeling")
            result.append(3)
            found = True
        if not found:
            print(count)
            raise Exception("missing labeling")
        count += 1
        
    df_little = pd.DataFrame(result,columns=['LABEL'])
    return df_little


#Read in coding data and convert to dataframe
df = pd.read_csv('.//big_coding_ZCH.csv',encoding_errors='ignore')
columns = ['ALIGNED','NOT_ALIGNED','NEUTRAL_IRRELEVANT','BORDER_CASE']
big_y = df[columns]


# Split the data into text and label
X = df['TEXT'][0:]
y = create_label(big_y)
flat_y = ravel(y)

#Option- write out new csv
#z = pd.concat([X,y],axis=1)
#z.to_csv('.//training_data.csv',index=False)


# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the training data, transform the test data
X_tfidf = vectorizer.fit_transform(X)


#set up model
model = RandomForestClassifier(random_state=42)
#model = AdaBoostClassifier(random_state=42)
#model = BaggingClassifier(random_state=42)
#model = ExtraTreesClassifier(random_state=42)
#model = GradientBoostingClassifier(random_state=42)
#model = StackingClassifier()
#model = SVC(kernel='linear')
#model = MultinomialNB(alpha=.2)
#model = PassiveAggressiveClassifier()
#model=GaussianProcessClassifier()

#set up cross folds
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Perform cross-validation
scorer = {'f1': make_scorer(f1_score, average='macro'), 
'accuracy': make_scorer(accuracy_score), 
'precision': make_scorer(precision_score, average='macro')}
cv_scores = cross_validate(model, X_tfidf, flat_y, cv=kf, scoring=scorer, return_train_score=False)

# Output results
print('F1 Scores: ', cv_scores['test_f1'])
print('Accuracy Scores: ', cv_scores['test_accuracy'])
print('Precision Scores: ', cv_scores['test_precision'])

print(f'Mean F1 Score: {cv_scores["test_f1"].mean()}')
print(f'Mean Accuracy Score: {cv_scores["test_accuracy"].mean()}')
print(f'Mean Precision Score: {cv_scores["test_precision"].mean()}')

