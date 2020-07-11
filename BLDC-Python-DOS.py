#!/usr/bin/env python
# Testing for git
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
bld = pd.read_csv("bank-loan.csv")


# In[2]:


bld.head(5)


# ## For Input data

# In[6]:

print("For predicting an input case")
print("Sample case for Default: 41	,3	,17	,12	,176	.9.3	,11.359392	,5.008608	")
print("Sample case forNon-default: 41	,2	,5	,5	,25	,10.2	,0.392700	,2.157300	")

age = int(input('Enter age'))
ed = int(input('Enter education'))
employ = int(input('Enter employ'))
address = int(input('Enter address'))
income = int(input('Enter income'))
debtinc = float(input('Enter debtinc'))
creddebt = float(input('Enter creddebt'))
othdebt = float(input('Enter othdebt'))

# lis =  ['age', 'ed', 'employ', 'address', 'income', 'debtinc', 'creddebt',
#        'othdebt']

# 41	3	17	12	176	9.3	11.359392	5.008608	1.0
# 41	2	5	5	25	10.2	0.392700	2.157300	0.0


# In[7]:



bld.loc[len(bld)] = [str(age), str(ed), str(employ), str(address), str(income), str(debtinc), str(creddebt), str(othdebt), "NaN"]
print(bld.loc[len(bld)-1:,:])

lis = ['age', 'ed', 'employ', 'address', 'income']
for i in lis:
    bld[i] = bld[i].astype('int64')
lis = ['debtinc', 'creddebt', 'othdebt', 'default']        
for i in lis:
    bld[i] = bld[i].astype('float')
bld.dtypes


# In[8]:


# temp_df = bld.copy()
# Imputed the last 150 rows of default in bld datafame.
from fancyimpute import KNN
temp_df=pd.DataFrame(KNN(k=3).fit_transform(bld), columns=bld.columns)

indexNames = temp_df[temp_df['default'] > 0.5].index
for i in range(0,len(temp_df['default'])):
    if i in indexNames:
        temp_df['default'][i] = 1.0
    else:
        temp_df['default'][i] = 0.0

        
for i in range(0,len(bld['default'])):
    bld['default'][i] = temp_df['default'][i]


# In[9]:


bld.loc[len(bld)-1:,:]
# bld.dtypes


# ## Outlier Analysis

# In[10]:


# Outlier Analysis 
import matplotlib.pyplot as plt

# df=bld.copy()
# plt.show(plt.boxplot(bld['age']))
# A-Note: Outlier Analysis can only be applied on continuous numeric variables.

## For the last default values
# from fancyimpute import KNN
# bld=pd.DataFrame(KNN(k=3).fit_transform(bld), columns=bld.columns)


colnames=['age', 'ed', 'employ', 'address', 'income', 'debtinc', 'creddebt',
       'othdebt']

for column in colnames: 
    q75, q25 = np.percentile(bld.loc[:,column],[75,25])

#   Calculate IQR
    iqr = q75 - q25

    # Calculate inner and outer fence.
    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)
    # print(minimum,maximum)

    # Drop
#     indexNames = bld[bld[column] < minimum].index
#     bld.drop(indexNames , inplace=True)
    
#     indexNames = bld[bld[column] > maximum].index
#     bld.drop(indexNames , inplace=True)
    
#     replace by nan.
    bld.loc[bld[column] < minimum ,:column] = np.nan
    bld.loc[bld[column] > maximum ,:column] = np.nan


bld=pd.DataFrame(KNN(k=3).fit_transform(bld), columns=bld.columns)
    
    
list=['age', 'ed', 'employ', 'address', 'income','default']
# list=['ed','default']

for i in list:
    print(i)
    bld[i] = bld[i].astype('int64')
# bld['ed'] = bld['ed'].astype('int64')
        

# B- nan occupied rows
for column in colnames: 
    Total=bld.loc[bld[column].isna()==True]
    print("NaN Occupied ",column," rows :",len(Total)) 
bld.shape    


# ## Feature Extraction

# In[11]:


# ### Standardisation of variables.
cnames=['age', 'employ', 'address', 'income', 'debtinc', 'creddebt',
       'othdebt']
for i in cnames:
    print(i)
    bld[i] = (bld[i] - bld[i].mean())/bld[i].std()
bld.head(10)

# ### Normalisation of variables.
# for i in cnames:
#     print(i)
#     bld[i] = (bld[i] - np.min(bld[i]))/(np.max(bld[i]) - np.min(bld[i]))
bld.head(10)


# ## Correation Analysis

# In[12]:


# ### Correlation Analysis : Only for continuous numeric variables.
#   # Correlation plot.NOTE:cp is only for continuous numeric variables.
#   # Extreme Blue:highly positively correlated.
#   # Extreme Red :highly negatively correlated.
colnames=['age', 'employ', 'address', 'income', 'debtinc', 'creddebt',
       'othdebt']
df_corr = bld.loc[:,colnames]
# print(df_corr.shape)


import matplotlib.pyplot as plt
# # Set the height and width of the plot
f, ax = plt.subplots(figsize=(7,5))

# # Generate correlation matrix.
corr = df_corr.corr()

# # Plot using seaborn library.
# # 'mask' -creates individual blocks for correlation matrix

import seaborn as sns
plt.show(sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220,10,as_cmap=True),
                     square=True, ax=ax))

# bld = bld.drop(['debtinc',"age","address"],axis=1)
bld = bld.drop(['debtinc'],axis=1)


# # Receiver Operating Characteristic Curves

# In[13]:


# roc curve and auc
#  Receiver Operating Characteristic curve
def roc_curve(x_train, x_test, y_train, y_test, model , model_name):
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from matplotlib import pyplot


#     # Dividing data into train and test.
#     X = bld.values[:,0:len(bld.columns)-1]
#     Y = bld.values[:,len(bld.columns)-1]

#     # split into train/test sets
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.37,random_state=0)

    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_test))]

    if model_name == "LR Model":
        model = LogisticRegression(solver='lbfgs')
        model.fit(x_train, y_train)

    # predict probabilities
    lr_probs = model.predict_proba(x_test)

    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]

    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)

    # summarize scores
    print(model_name, 'No Skill: ROC AUC=%.3f' % (ns_auc))

    print(model_name, ' ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label=model_name)

    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')

    # show the legend
    pyplot.legend()

    # show the plot
    pyplot.show()
    
    return lr_auc   
     


# # Precision Recall Curves

# In[39]:


def pr_curve(x_train, x_test, y_train, y_test, model , model_name):
    # precision-recall curve and f1
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import f1_score
    from sklearn.metrics import auc
    from matplotlib import pyplot
    # fit a model
   
    if model_name == "LR Model":
        model = LogisticRegression(solver='lbfgs')
        model.fit(x_train, y_train)
    
    # predict probabilities
    lr_probs = model.predict_proba(x_test)
    
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    
    # predict class values
    yhat = model.predict(x_test)
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
    lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)
    
    # summarize scores
    print(model_name,': f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    
    # plot the precision-recall curves
    no_skill = len(y_test[y_test==1]) / len(y_test)
    if model_name != 'LR Model':
        pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(lr_recall, lr_precision, marker='.', label=model_name)
    
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    
    # show the legend
    pyplot.legend()
    
    # show the plot
    pyplot.show()
    
    return lr_auc


# # Logistic Regression Model

# In[40]:


# B- nan occupied rows
# a=bld_logit.loc[bld_logit['default'].isna()==True]
# len(a)

# B-Creating logistic data and saving the target variable first.
bld_logit=pd.DataFrame(bld['default'])

# B-joining the target variable('default') with the rest of continuous variables excluding 'education'.
cnames=['employ', 'creddebt','othdebt']
# cnames=['debtinc','creddebt','othdebt']

bld_logit=bld_logit.join(bld[cnames])


# B-Joining with the categorical variable, by dividing each category(Q. Why do we need to do this)
temp = pd.get_dummies(bld['ed'], prefix='ed')
bld_logit = bld_logit.join(temp)
# bld_logit = bld_logit.drop(['ed_0'],axis=1)


bld_logit


# ## For input data

# In[41]:


input_sample=bld_logit.loc[len(bld_logit)-1:,:]
input_sample= input_sample.drop(['default'], axis=1)


# Dropping the input sample
bld_logit= bld_logit.drop(bld_logit.index[[850]])


# In[42]:


# B-Slicing the df into train and test.
Sample_index = np.random.rand(len(bld_logit)) < 0.7
train = bld_logit[Sample_index]
test = bld_logit[~Sample_index]

x_train = train.drop(['default'], axis=1)
x_test = test.drop(['default'], axis=1)
y_train = train[['default']]
y_test = test[['default']]

print(x_train.shape,
x_test.shape,
y_train.shape,
y_test.shape)
print(train.shape,test.shape)


# In[43]:


# B-Alloting column indexes to independent variables.
train_cols = train.columns[1:]
len(train_cols)
train_cols


# In[44]:


# B-Building logistic regression model.
import statsmodels.api as sm
LR_model = sm.Logit(train['default'],train[train_cols], random_state=0).fit()
LR_model.summary()


# In[45]:


# B-Predicting for test data.
test['Actual_prob']=LR_model.predict(test[train_cols])
# test.loc[0:5,'ed_5.0':'Actual_prob']

test['Actual_val']=1
# B-Selecting the index of rows with prob less than 0.5 which return an list of indexes.
idx=test.index[test['Actual_prob']<0.6]
idx
for x in idx:
    test.at[x,'Actual_val']=0

test['Actual_val'].head(20)


# ## For input data

# In[46]:


test[train_cols]


# In[47]:


new_output = LR_model.predict(input_sample)
print(new_output)
if new_output.item() < 0.5:
    print("Default value from Logistic Regression is",0)
else :
    print("Default value from Logistic Regression is",1)
    


# In[48]:


# Building confusion matrix
from sklearn.metrics import confusion_matrix

CM = confusion_matrix(test['default'],test['Actual_val'])

CM = pd.crosstab(test['default'],test['Actual_val'])
print(CM)


# test.head(20)
# Let us save TN,TP,FN,FP.
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

accuracy = ((TP + TN)*100) / (TP + TN + FN + FP)
print("Accuracy from Logistic regression", accuracy)

# Recall
recall=(TP*100)/(FN + TP)
print("Recall ", recall)
Model_name="LR Model"

############################# ROC ###################################

roc_auc = roc_curve(x_train, x_test, y_train, y_test, LR_model , Model_name)


############################ PR ##################################

pr_auc = pr_curve(x_train, x_test, y_train, y_test, LR_model , Model_name)

############################# Gain Chart #####################################


from sklearn.model_selection import train_test_split

# bld=bld.drop(bld.index([[850]]))
# Dividing data into train and test.
X = bld.values[:,0:len(bld.columns)-1]
Y = bld.values[:,len(bld.columns)-1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, 
test_size=0.3)

from sklearn.linear_model import LogisticRegression
import scikitplot as skplt

lr = LogisticRegression()
lr = lr.fit(x_train, y_train)
y_probas = lr.predict_proba(x_test)
skplt.metrics.plot_cumulative_gain(y_test, y_probas)
plt.show()
############################ Lift chart ###################################

import numpy as np
skplt.metrics.plot_lift_curve(y_test, y_probas)
plt.show()


print(x_train.shape,
x_test.shape,
y_train.shape,
y_test.shape)

###################### Storing Accuracy and recall  #######################

df1 = pd.DataFrame(columns=['Model_name','Accuracy','Recall','Roc_auc','Pr_auc'])
idx=len(df1)
df1.loc[idx,'Model_name'] = Model_name
df1.loc[idx,'Accuracy'] = accuracy
df1.loc[idx,'Recall'] = recall
df1.loc[idx,'Roc_auc'] = roc_auc
df1.loc[idx,'Pr_auc'] = pr_auc


# ### Decision Tree Model

# In[49]:


################ Decision Tree Model #################
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

# Dividing data into train and test.
X = bld.values[:len(bld)-1,0:len(bld.columns)-1]
Y = bld.values[:len(bld)-1,len(bld.columns)-1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# Decision tree( criterion ='entropy' means the C.50 model for classification)
DT_model = tree.DecisionTreeClassifier(criterion='entropy', random_state=0).fit(x_train, y_train)

# B-Predicting new test cases. Predicted values of default.
y_pred = DT_model.predict(x_test)
# y_pred.shape

# # B-Creating a dot file to visualize tree. #http://webgraphviz.com/
# dotfile = open("bld.dot",'w')

# # B-exporting DT.
# temp_df = bld.loc[:, bld.columns != 'default']
# df = tree.export_graphviz(DT_model, out_file=dotfile, feature_names=temp_df.columns)

# B-Building Confusion matrix.
CM = confusion_matrix(y_test, y_pred)
CM = pd.crosstab(y_test, y_pred, rownames=['y_test'], colnames=['y_pred'])
print(CM)

# A-column represents predicted and row represents actual values.

# Let us save TP,TN,FP,FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

# Alternatively, Accuracy -
accuracy= ((TP + TN)*100) / (TP + TN + FN + FP)
print("Accuracy from DT :",accuracy)

# Recall.
recall=(TP*100)/(FN + TP)
print("False Negative rate:", recall)

Model_name="DT Model"
########################## ROC ############################

roc_auc=roc_curve(x_train, x_test, y_train, y_test, DT_model , Model_name)

############################ PR ##############################

pr_auc=pr_curve(x_train, x_test, y_train, y_test, DT_model , Model_name)

############################# Gain Chart #####################################

import scikitplot as skplt

y_probas = DT_model.predict_proba(x_test)
skplt.metrics.plot_cumulative_gain(y_test, y_probas)
plt.show()

############################ Lift chart ###################################

import numpy as np
skplt.metrics.plot_lift_curve(y_test, y_probas)
plt.show()

###################### Storing Accuracy and recall  #######################

idx=len(df1)
df1.loc[idx,'Model_name'] = Model_name
df1.loc[idx,'Accuracy'] = accuracy
df1.loc[idx,'Recall'] = recall
df1.loc[idx,'Roc_auc'] = roc_auc
df1.loc[idx,'Pr_auc'] = pr_auc


# ## For new input

# In[25]:


a = bld.loc[len(bld)-1:,:].drop(['default'], axis=1)
new_output=DT_model.predict(a)
print("DT Model")
print("Default value for input data is:",int(new_output[0]))


# 
# 
# ### Random Forest Classifier

# In[50]:


############ Random Forest Classifier ###########
from sklearn.model_selection import train_test_split
# Dividing data into train and test.
x = bld.values[:len(bld)-1,0:len(bld.columns)-1]
y = bld.values[:len(bld)-1,len(bld.columns)-1]

# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
x_train,x_test,y_train,y_test = train_test_split(
    x, y, random_state=0, test_size=0.36)

from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators=88, max_depth=9, random_state=0).fit(x_train,y_train)

RF_Predictions = RF_model.predict(x_test)



# build confusion matrix.
CM = pd.crosstab(y_test,RF_Predictions,rownames=['y_test'],colnames=['RF_Predictions'])
# Let us save TP,TN,FN,FP
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]
print(CM)

# Alternatively, Accuracy -
accuracy= ((TP + TN)*100) / (TP + TN + FN + FP)
print("Accuracy from Random Forest :", accuracy)

# Recall.
recall = (TP*100)/(FN + TP)
print("Recall from Random Forest:", recall)

Model_name = "RF Model"

############################# ROC ################################

roc_auc=roc_curve(x_train, x_test, y_train, y_test, RF_model , Model_name)

############################ PR ##############################

pr_auc=pr_curve(x_train, x_test, y_train, y_test, RF_model , Model_name)

############################# Gain Chart #####################################

import scikitplot as skplt

y_probas = RF_model.predict_proba(x_test)
skplt.metrics.plot_cumulative_gain(y_test, y_probas)
plt.show()

############################ Lift chart ###################################

import numpy as np
skplt.metrics.plot_lift_curve(y_test, y_probas)
plt.show()

###################### Storing Accuracy and recall  #######################

idx=len(df1)
df1.loc[idx,'Model_name'] = Model_name
df1.loc[idx,'Accuracy'] = accuracy
df1.loc[idx,'Recall'] = recall
df1.loc[idx,'Roc_auc'] = roc_auc
df1.loc[idx,'Pr_auc'] = pr_auc


# ## For input data

# In[51]:


a = bld.loc[len(bld)-1:,:].drop(['default'], axis=1)
new_output=RF_model.predict(a)
print("RF Model")
print("Default value for input data is:",int(new_output[0]))


# 
# 
# ### KNN Model

# In[52]:


###################### KNN Model ###########################

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# B-Dividing data into train and test.

X = bld.values[:len(bld)-1,0:len(bld.columns)-1]
Y = bld.values[:len(bld)-1,len(bld.columns)-1]

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
KNN_model =  KNeighborsClassifier(n_neighbors=1 ).fit(x_train,y_train)
KNN_Predictions = KNN_model.predict(x_test)
KNN_Predictions.shape

# A-Predict test cases
# A-KNN_Predictions


CM = pd.crosstab(y_test, KNN_Predictions)

# Let us save  TP, TN, FP, FN.
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]
print(CM)

# Check accuracy of model.
accuracy = ((TP + TN)*100) / (TP + TN + FN + FP)
print("KNN accuracy:", accuracy)

# TNR
recall = (TP*100)/(FN + TP)
print("Recall      :" ,recall)

Model_name = "KNN Model"

################################# ROC #################################

roc_auc=roc_curve(x_train, x_test, y_train, y_test, KNN_model , Model_name)

############################ PR ##############################

pr_auc=pr_curve(x_train, x_test, y_train, y_test, KNN_model , Model_name)

############################# Gain Chart #####################################

import scikitplot as skplt

y_probas = KNN_model.predict_proba(x_test)
skplt.metrics.plot_cumulative_gain(y_test, y_probas)
plt.show()

############################ Lift chart ###################################

import numpy as np
skplt.metrics.plot_lift_curve(y_test, y_probas)
plt.show()

###################### Storing Accuracy and recall  #######################

idx=len(df1)
df1.loc[idx,'Model_name'] = Model_name
df1.loc[idx,'Accuracy'] = accuracy
df1.loc[idx,'Recall'] = recall
df1.loc[idx,'Roc_auc'] = roc_auc
df1.loc[idx,'Pr_auc'] = pr_auc


# ## For input data

# In[53]:


a = bld.loc[len(bld)-1:,:].drop(['default'], axis=1)
new_output=KNN_model.predict(a)
print("KNN Model")
print("Default value for input data is:",int(new_output[0]))


# 
# 
# ### Naive Bayes
# 
# 

# In[54]:


# GaussianNB since our target variable(responded) has only two classes; yes or no .If it has multiple classes then MultinomialNB. 
from sklearn.naive_bayes import GaussianNB

X = bld.values[:len(bld)-1,0:len(bld.columns)-1]
Y = bld.values[:len(bld)-1,len(bld.columns)-1]

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.65,random_state=0)

# Naive Bayes implementation.x_train=independent variable of training data, y_train= dependent variable of training data.
NB_model = GaussianNB(var_smoothing=1e-11).fit(x_train,y_train)

# Predict test cases
NB_Predictions = NB_model.predict(x_test)
NB_Predictions

# Build confusion matrix
CM = pd.crosstab(y_test,NB_Predictions)

# Let us save  TP, TN, FP, FN.
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]
print(CM)

# Check accuracy of model.
accuracy = ((TP + TN)*100) / (TP + TN + FN + FP)
print("Naive Bayes", accuracy)
# recall
recall = (TP*100)/(FN + TP)
print("Recall",recall)

Model_name = "NB Model"


######################### ROC #############################

roc_auc=roc_curve(x_train, x_test, y_train, y_test,NB_model , Model_name)

############################ PR ##############################

pr_auc=pr_curve(x_train, x_test, y_train, y_test, NB_model , Model_name)

############################# Gain Chart #####################################

import scikitplot as skplt

y_probas = NB_model.predict_proba(x_test)
skplt.metrics.plot_cumulative_gain(y_test, y_probas)
plt.show()

############################ Lift chart ###################################

import numpy as np
skplt.metrics.plot_lift_curve(y_test, y_probas)
plt.show()

###################### Storing Accuracy and recall  #######################

idx=len(df1)
df1.loc[idx,'Model_name'] = Model_name
df1.loc[idx,'Accuracy'] = accuracy
df1.loc[idx,'Recall'] = recall
df1.loc[idx,'Roc_auc'] = roc_auc
df1.loc[idx,'Pr_auc'] = pr_auc


# ## For input data

# In[55]:


a = bld.loc[len(bld)-1:,:].drop(['default'], axis=1)
new_output=NB_model.predict(a)
print("NB Model")
print("Default value for input data is:",int(new_output[0]))


# 
# ### Gradient Boosting Classifier

# In[56]:



from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
# x, y = make_classification(random_state=0)


X = bld.values[:len(bld)-1,0:len(bld.columns)-1]
Y = bld.values[:len(bld)-1,len(bld.columns)-1]

x_train,x_test,y_train,y_test=train_test_split(X,Y, random_state=0,test_size=0.2)

GB_model = GradientBoostingClassifier(random_state=0, n_estimators=105, learning_rate=1.46, max_depth=1)
GB_model.fit(x_train, y_train)
# GradientBoostingClassifier(random_state=0)
GB_Predictions = GB_model.predict(x_test)
# array([1, 0])
GB_model.score(x_test, y_test)

# Build confusion matrix
CM = pd.crosstab(y_test,GB_Predictions)

# Let us save  TP, TN, FP, FN.
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]
print(CM)

# Check accuracy of model.
accuracy = round(((TP + TN)*100) / (TP + TN + FN + FP))
print("Gradient Boosting Algorithm", accuracy)
# recall
recall = (TP*100)/(FN + TP)
print("Recall ", recall)
Model_name = "GB Model"

############################ ROC #############################

roc_auc=roc_curve(x_train, x_test, y_train, y_test, GB_model, Model_name)

############################ PR ##############################

pr_auc=pr_curve(x_train, x_test, y_train, y_test, GB_model , Model_name)

############################# Gain Chart #####################################

import scikitplot as skplt

y_probas = GB_model.predict_proba(x_test)
skplt.metrics.plot_cumulative_gain(y_test, y_probas)
plt.show()

############################ Lift chart ###################################

import numpy as np
skplt.metrics.plot_lift_curve(y_test, y_probas)
plt.show()

###################### Storing Accuracy and recall  #######################

idx=len(df1)
df1.loc[idx,'Model_name'] = Model_name
df1.loc[idx,'Accuracy'] = accuracy
df1.loc[idx,'Recall'] = recall
df1.loc[idx,'Roc_auc'] = roc_auc
df1.loc[idx,'Pr_auc'] = pr_auc


# ## For input data

# In[57]:


a = bld.loc[len(bld)-1:,:].drop(['default'], axis=1)
new_output=GB_model.predict(a)
print("GB Model")
print("Default value for input data is:",int(new_output[0]))


# ### XG Boosting
# 

# In[58]:


from xgboost import XGBClassifier

# Dividing data into train and test.
X = bld.values[:len(bld)-1,0:len(bld.columns)-1]
Y = bld.values[:len(bld)-1,len(bld.columns)-1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.37,random_state=0)

XGB_model = XGBClassifier(random_state=0, n_estimators=53, learning_rate=1.32, max_depth=6).fit(x_train,y_train)
# model = XGBClassifier( random_state=0, n_estimators=140, learning_rate=0.14, max_depth=6)



# predict the target on the train dataset
XGB_Predictions = XGB_model.predict(x_test)

# a=XGB_model.score(x_test, y_test)
# a
# Build confusion matrix
CM = pd.crosstab(y_test, XGB_Predictions)

# Let us save  TP, TN, FP, FN.
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]
print(CM)

# Check accuracy of model.
accuracy=round(((TP + TN)*100) / (TP + TN + FN + FP))
print("Accuracy of XGB Model",accuracy)
#Recall
recall=(TP*100)/(FN + TP)
print("Recall: ",recall)

Model_name="XGB Model"

############################ ROC ##############################

roc_auc=roc_curve(x_train, x_test, y_train, y_test, XGB_model , Model_name)

############################ PR ###############################

pr_auc=pr_curve(x_train, x_test, y_train, y_test, XGB_model , Model_name)

############################# Gain Chart #####################################

import scikitplot as skplt

y_probas = XGB_model.predict_proba(x_test)
skplt.metrics.plot_cumulative_gain(y_test, y_probas)
plt.show()

############################ Lift chart ###################################

import numpy as np
skplt.metrics.plot_lift_curve(y_test, y_probas)
plt.show()


#########################################

idx=len(df1)
df1.loc[idx,'Accuracy'] = accuracy
df1.loc[idx,'Recall'] = recall
df1.loc[idx,'Model_name'] = Model_name
df1.loc[idx,'Roc_auc'] = roc_auc
df1.loc[idx,'Pr_auc'] = pr_auc


# ## For input data

# In[59]:


a = bld.loc[len(bld)-1:,:].drop(['default'], axis=1)
new_output=XGB_model.predict(a.values)
print("XGB Model")
print("Default value for input data is:",int(new_output[0]))


# ## Support Vector Machine Model

# In[60]:


#Data Pre-processing Step  
# importing libraries  
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  
 
   
#Extracting Independent and dependent Variable  
X= bld.iloc[:, 0:len(bld.columns)-1].values 
Y= bld.iloc[:, len(bld.columns)-1].values  
 
# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(X, Y, test_size= 0.20, random_state=0)  

#feature Scaling
# from sklearn.preprocessing import StandardScaler    
# st_x= StandardScaler()    
# x_train= st_x.fit_transform(x_train)    
# x_test= st_x.transform(x_test)

from sklearn.svm import SVC # "Support vector classifier" rbf 
# SVM_model = SVC(kernel='rbf', random_state=0, C=2.0, gamma=1.0)  
# SVM_model = SVC(kernel='rbf', random_state=0, C=3.0, gamma=0.5)
SVM_model = SVC(kernel='rbf', random_state=0, C=2.79, gamma=0.5 )

SVM_model.fit(x_train, y_train) 
print(SVM_model)
#Predicting the test set result  
SVM_Predictions= SVM_model.predict(x_test)  

# Build confusion matrix
CM = pd.crosstab(y_test, SVM_Predictions)

# Let us save  TP, TN, FP, FN.
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]
print(CM)

# Check accuracy of model.
accuracy=round(((TP + TN)*100) / (TP + TN + FN + FP))
print("Accuracy of SVM Model",accuracy)
#Recall
recall=(TP*100)/(FN + TP)
print("Recall: ",recall)

Model_name="SVM Model"


############################ ROC ##############################

roc_auc=roc_curve(x_train, x_test, y_train, y_test, XGB_model , Model_name)

############################ PR ###############################

pr_auc=pr_curve(x_train, x_test, y_train, y_test, XGB_model , Model_name)

############################# Gain Chart #####################################

import scikitplot as skplt

y_probas = XGB_model.predict_proba(x_test)
skplt.metrics.plot_cumulative_gain(y_test, y_probas)
plt.show()

############################ Lift chart ###################################

import numpy as np
skplt.metrics.plot_lift_curve(y_test, y_probas)
plt.show()


#########################################

idx=len(df1)
df1.loc[idx,'Accuracy'] = accuracy
df1.loc[idx,'Recall'] = recall
df1.loc[idx,'Model_name'] = Model_name
df1.loc[idx,'Roc_auc'] = roc_auc
df1.loc[idx,'Pr_auc'] = pr_auc


# ## For input data

# In[61]:


a = bld.loc[len(bld)-1:,:].drop(['default'], axis=1)
new_output=SVM_model.predict(a.values)
print("SVM Model")
print("Default value for input data is:", (new_output[0]))


# ## Dataframe for Accuracy and Recall of Models 

# In[62]:


df1


# In[ ]:




