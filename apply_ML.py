
# Models
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
print(ROC_gbr,ROC_lr)

# precision score test
PRE_lr = average_precision_score(y_test, test_score_lr)
PRE_gbrr = average_precision_score(y_test, test_score_gbr)
print(PRE_gbrr,PRE_lr)

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


# Handle our dependent varaible we could say it is not important
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


# test our data 

test_score_gbr_rfe = gbr_best_rfe.predict_proba(x_test_rfe)[:,1]
ROC_gbr_rfe = roc_auc_score(y_test, test_score_gbr_rfe)
PRE_gbrr_rfe = average_precision_score(y_test, test_score_gbr_rfe)


# Elimination our feutares 
def get_fi (gbr_best, x_train):
    feature_importance = pd.DataFrame([x_train.columns.tolist(), gbr_best.feature_importances_]).T
    feature_importance.columns = ['varname', 'importance']
    feature_importance = feature_importance.sort_values(by=['importance'], ascending= False)
    feature_importance['cm_importance'] = feature_importance['importance'].cumsum()
    return feature_importance

fi = get_fi(gbr_best_rfe,x_test_rfe)
vals= list(fi['importance'])
plt.barh(fi['varname'], fi['importance'])
plt.title('Importance Of Differnet Variables')
plt.gca().xaxis.grid(linestyle=':')











