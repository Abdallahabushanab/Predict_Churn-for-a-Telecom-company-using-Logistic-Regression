# find numeric variable 
continous_var  = train.select_dtypes([np.number]).columns.tolist()
continous_var = [x for x in continous_var if '_num' not in x]

# calling columns
calls_var = [x for x in continous_var if 'calls' in x]
train.boxplot(column= calls_var,figsize=(20,10))

# All same type varaible in one grapgh  
type_of_vars = ['intl', 'customer', 'minutes', 'call', 'charge']
remaning_list = train.columns

# apply box plot for all others
for var in type_of_vars:
    temp_list = [x for x in remaning_list if var in x]
    remaning_list = list(set(remaning_list).difference(set(temp_list)))
    train.boxplot(column= temp_list, figsize=(20,10))
    plt.title('Box Plot for ' + var +'variables')
    plt.show()
    
'''
The remaning list 

['voice_mail_plan_num',
 'international_plan_num',
 'state',
 'voice_mail_plan',
 'international_plan',
 'churn_num',
 'churn',
 'account_length',
 'number_vmail_messages',
 'area_code']

'''
# let's study the corrleation 
x = train.drop('churn_num', axis = 1)
corr = x.corr().unstack().reset_index()

corr_table = corr[corr['level_0'] != corr['level_1']]
corr_table.columns = ['var1', 'var2', 'corr_value']
corr_table['corr_abs'] = corr_table['corr_value'].abs()
corr_table = corr_table.sort_values(by=['corr_abs'], ascending=False)


# Heatmap 
sns.heatmap(x.corr())
sns.pairplot(x)

# After we got correlation we found mintues and charges have high correlation
# So we will calculate the percent of all charges per all mintus in one column
charges_var = [x for x in train.columns if 'charge' in x]
minutes_var = [x for x in train.columns if 'minutes' in x]

print(charges_var)
print(minutes_var)

# Handle charges and minutes 
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

# handle category columns 
cat_columns = ['state', 'area_code']
train_1 = pd.concat([train_1, pd.get_dummies(train_1[cat_columns],
                                              drop_first=True)], axis=1)
test_1 = pd.concat([test_1, pd.get_dummies(test_1[cat_columns],
                                              drop_first=True)], axis=1)


# Deop unessary column 
train_1 =train_1.drop(['international_plan','voice_mail_plan','state',
                       'area_code', 'voice_mail_plan_num'], axis=1)
test_1 =test_1.drop(['international_plan','voice_mail_plan','state',
                     'area_code', 'voice_mail_plan_num'], axis=1)

x_train = train_1.drop('churn_num', axis=1)
x_test = test_1.drop('churn_num', axis=1)
y_train = train_1['churn_num']
y_test = test_1['churn_num']











