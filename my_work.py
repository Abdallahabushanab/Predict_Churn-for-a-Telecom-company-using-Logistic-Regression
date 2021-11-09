#  librires 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_theme(style="darkgrid")

# import our data here and save it in orginal file

data = pd.read_csv('Desktop\ProjectPro\project_1_classifiaer_churn\Telecom_Train.csv')
data_test = pd.read_csv('Desktop\ProjectPro\project_1_classifiaer_churn\Telecom_Test.csv')
train = data.copy()
test = data_test.copy()

# understanding our data by small steps and visual tools
def understand_our_data(data):
    print('Our Data Info')
    print(data.info())
    print('Describe our data')
    print(data.describe())
    print('Objects columns')
    print(data.dtypes == 'object')
    print('Sorted type of columns')
    print(data.dtypes.sort_values())
    print('Number of null values')
    print(data.isna().sum().sort_values())
    print('Shape of our Data')
    print(data.shape)
    print('Percnt of test data comparing of train data')
    print(data.shape[0]/test.shape[0])
    print('Number of unique vales')
    print(data.nunique().sort_values())
    
understand_our_data(data)
understand_our_data(test)


unique_values = ['number_customer_service_calls','state','area_code', 
                 'international_plan','voice_mail_plan','churn' ]

for i in unique_values:
    print('for column {} this is unique values'.format(i))
    print('unique values number:{}'.format(len(data[i].value_counts())))
    print(data[i].value_counts(),"\n")
    plt.hist(data[i])
    plt.title(i)
    plt.show()


''' 
Notes for our data:
1) All our data not null values
2) state,area_code,international_plan,voice_mail_plan, churn all these data object  
3) we have 21 columns and 3333 rows
4) out test data is half our data (1.99)
5) account_length how many days customer was with company    
6) we study in 51 state
customer chun the perecantge of customer who stop doing business with any entity

'''
train = train.drop('Unnamed: 0', axis =1)
test = test.drop('Unnamed: 0', axis =1)

def cat_to_binary(df, colname):
    df[colname + '_num'] = df[colname].apply(lambda x : 1
                                             if x =='yes' else 0 )
    print('checking')
    print(df.groupby([colname + '_num', colname]).size())
    return df

convert_list = ['churn', 'international_plan','voice_mail_plan']
for col in convert_list:
    train = cat_to_binary(train , col)
    test = cat_to_binary(test , col)
    

plt.hist(list(train['area_code']))

plt.figure(figsize= (20,10))
plt.hist(list(train['state']), bins = 100)

train.mean()

# visulization pie plot 
topie = train['churn'].value_counts(sort =True)
colors = ['darkgreen', 'red']
plt.pie(topie, colors = colors, explode = [0,0.2], autopct='%1.1f%%',
        shadow =True, startangle=90)
plt.title('% churn in Training Data')
plt.show()

# find numeric variable 

continous_var  = train.select_dtypes([np.number]).columns.tolist()
continous_var = [x for x in continous_var if '_num' not in x]
calls_var = [x for x in continous_var if 'calls' in x]
train.boxplot(column= calls_var,figsize=(20,10))


# All same type varaible in one grapgh  
type_of_vars = ['intl', 'customer', 'minutes', 'call', 'charge']
remaning_list = train.columns
for var in type_of_vars:
    temp_list = [x for x in remaning_list if var in x]
    remaning_list = list(set(remaning_list).difference(set(temp_list)))
    train.boxplot(column= temp_list, figsize=(20,10))
    plt.title('Box Plot for ' + var +'variables')
    plt.show()

'''
We can say 0.15 of people will leave the company
 at the end of this year.

'''
# let's study the corrleation 
x = train.drop('churn_num', axis = 1)
corr = x.corr().unstack().reset_index()

corr_table = corr[corr['level_0'] != corr['level_1']]
corr_table.columns = ['var1', 'var2', 'corr_value']
corr_table['corr_abs'] = corr_table['corr_value'].abs()
corr_table = corr_table.sort_values(by=['corr_abs'], ascending=False)

sns.heatmap(x.corr())
sns.pairplot(x)


# After we got correlation we found mintues and charges have high correlation
# So we will calculate the percent of all charges per all mintus in one column
charges_var = [x for x in train.columns if 'charge' in x]
minutes_var = [x for x in train.columns if 'minutes' in x]

print(charges_var)
print(minutes_var)

def creat_cpm(df):
    df['total_charges'] = 0
    df['total_minutes'] = 0
    for y in range(0, len(charges_var)):
        df['total_charges'] += df[charges_var[y]]
        df['total_minutes'] += df[minutes_var[y]]
    df['charge_per_mintue'] = np.where(df['total_minutes'] >0,
                                       df['total_charges']/df['total_minutes'],0)
    df.drop(['total_charges','total_minutes'], axis = 1 , inplace = True)
    print(df['charge_per_mintue'].describe())
    return df 

train = creat_cpm(train)
test = creat_cpm(test)
train.boxplot(column = 'charge_per_mintue', figsize=(20,10))

# Normal distrubtion for each column
def create_pdf (df, varname):
    plt.figure(figsize=(25,5))
    plt.hist(list(df[df['churn_num'] ==0][varname]), bins=50, label='non-churend',
             density=True, color='g', alpha=0.8)
    plt.hist(list(df[df['churn_num'] ==1][varname]), bins=50, label='churend',
             density=True, color='r', alpha=0.8)
    plt.legend(loc='upper right')
    plt.xlabel(varname)
    plt.ylabel('pdf')
    plt.show()
    
for varname in train.columns:
    create_pdf(train, varname)
    

# prepare our data for modeling
drop_column = ['total_day_charge', 'total_eve_charge', 'total_night_charge',
               'total_intl_charge','churn']
train_1 = train.drop(drop_column, axis =1)
test_1 = test.drop(drop_column, axis = 1)
print(train_1.shape)

cat_columns = ['state', 'area_code']
train_1 = pd.concat([train_1, pd.get_dummies(train_1[cat_columns],
                                              drop_first=True)], axis=1)
test_1 = pd.concat([test_1, pd.get_dummies(test_1[cat_columns],
                                              drop_first=True)], axis=1)

train_1 =train_1.drop(['international_plan','voice_mail_plan','state',
                       'area_code'], axis=1)
test_1 =test_1.drop(['international_plan','voice_mail_plan','state',
                     'area_code'], axis=1)
train_1 =train_1.drop(['voice_mail_plan_num'], axis=1)
test_1 =test_1.drop(['voice_mail_plan_num'], axis=1)

x_train = train_1.drop('churn_num', axis=1)
x_test = test_1.drop('churn_num', axis=1)
y_train = train_1['churn_num']
y_test = test_1['churn_num']

# Great our data now is ready for working
# now lets apply some models

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

#Logistic Regression 
lr = LogisticRegression(random_state=42, solver='liblinear')
params = {'penalty': ['l1','l2'], 'C': [0.1,1,2,3,5]}
cv_lr = GridSearchCV(estimator=lr, param_grid = params, cv=5)
cv_lr.fit(x_train, y_train)
lr_best = cv_lr.best_estimator_
print(lr_best)
test_score_lr = lr_best.predict_proba(x_test)[:,1]
pd.Series(test_score_lr).describe()

# Gradient boosting classifier 
gbr = GradientBoostingClassifier(random_state=42)
params_gbr = {'n_estimators': [50,100,500], 'max_features': ['auto'],
          'learning_rate':[0.01,0.05,0.1,0.2]}
cv_gbr = GridSearchCV(estimator=gbr, param_grid = params_gbr, cv=5)
cv_gbr.fit(x_train, y_train)
gbr_best = cv_gbr.best_estimator_
print(gbr_best)

test_score_gbr = gbr_best.predict_proba(x_test)[:,1]
pd.Series(test_score_gbr).describe()

from sklearn.metrics import roc_auc_score, confusion_matrix,accuracy_score, precision_recall_curve
from sklearn.metrics import average_precision_score, roc_curve,precision_score

#  roc auc score test
ROC_lr = roc_auc_score(y_test, test_score_lr)
ROC_gbr = roc_auc_score(y_test, test_score_gbr)
print(ROC_gbr)

# precision score test
PRE_lr = average_precision_score(y_test, test_score_lr)
PRE_gbrr = average_precision_score(y_test, test_score_gbr)

# roc curve test
fpr_gbr, tpr_gbr, _ = roc_curve(y_test, test_score_gbr)
plt.plot(fpr_gbr, tpr_gbr, label='GBR')
fpr_lr, tpr_lr, _ = roc_curve(y_test, test_score_lr)
plt.plot(fpr_lr, tpr_lr, label='LR')
plt.title('ROC curve')
plt.xlabel('False Postive Rate');plt.ylabel('Ture Postive Rate')
plt.legend()

# precision_recall_curve test
precision_gbr, recall_gbr, _ = precision_recall_curve(y_test, test_score_gbr)
plt.plot(recall_gbr, precision_gbr, label='GBR')
precision_lr, recall_lr, _ = precision_recall_curve(y_test, test_score_lr)
plt.plot(recall_lr, precision_lr, label='LR')
plt.title('Precision Recall Curve')
plt.xlabel('Recall');plt.ylabel('precision')
plt.legend()

# Confusion Matrix
cm = confusion_matrix(y_test, (test_score_gbr>=0.5))
ax = plt.subplot()
sns.heatmap(cm, ax = ax, annot=True, fmt='g')
ax.set_xlabel('Predection Labels'); ax.set_ylabel('True Labels')
ax.xaxis.set_ticklabels(['retained', 'churned']) 
ax.yaxis.set_ticklabels(['retained', 'churned'])
ax.set_title('Confusion Matrix')

# accurcy score
accuracy_score(y_test, (test_score_gbr>=0.5))

# find the number of people who will leave the company
get_top10 =pd.concat([pd.Series(test_score_gbr, name='model_score'), y_test], axis =1)
get_top10 = get_top10.sort_values(by=['model_score'], ascending=False)
get_top10.head()  
get_top10['rownum'] = np.arange(len(get_top10))
get_top10[get_top10['rownum']<= y_test.shape[0]/10]['churn_num'].value_counts()

# Elimination our feutares 

def get_fi (gbr_best, x_train):
    feature_importance = pd.DataFrame([x_train.columns.tolist(), gbr_best.feature_importances_]).T
    feature_importance.columns = ['varname', 'importance']
    feature_importance = feature_importance.sort_values(by=['importance'], ascending= False)
    feature_importance['cm_importance'] = feature_importance['importance'].cumsum()
    return feature_importance

area_var = [x for x in x_train.columns if 'area' in x]
state_var = [x for x in x_train.columns if 'state' in x]
total_var = area_var + state_var

x_train_rfe = x_train.drop(total_var, axis=1)
x_test_rfe = x_test.drop(total_var, axis=1)

# Gradient boosting classifier for new  
gbr = GradientBoostingClassifier(random_state=42)
params_gbr = {'n_estimators': [50,100,500], 'max_features': ['auto'],
          'learning_rate':[0.01,0.05,0.1,0.2]}
cv_gbr = GridSearchCV(estimator=gbr, param_grid = params_gbr, cv=5)
cv_gbr.fit(x_train_rfe, y_train)
gbr_best_rfe = cv_gbr.best_estimator_
print(gbr_best_rfe)

# test our new 

test_score_gbr_rfe = gbr_best_rfe.predict_proba(x_test_rfe)[:,1]
ROC_gbr_rfe = roc_auc_score(y_test, test_score_gbr_rfe)
PRE_gbrr_rfe = average_precision_score(y_test, test_score_gbr_rfe)

fi = get_fi(gbr_best_rfe,x_test_rfe)
vals= list(fi['importance'])
plt.barh(fi['varname'], fi['importance'])
plt.title('Importance Of Differnet Variables')
plt.gca().xaxis.grid(linestyle=':')



*******************************************************************************************

# correlaction
import scipy 

a = train['total_day_calls'].to_numpy()
b =train['total_day_calls'].to_numpy()
scipy.stats.pearsonr(a, b)

