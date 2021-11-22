#  librires 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_theme(style="darkgrid")

# import our data here and save it in orginal file

data = pd.read_csv('Desktop\ProjectPro\project_1_classifiaer_churn\Churn_project\Telecom_Train.csv')
data_test = pd.read_csv('Desktop\ProjectPro\project_1_classifiaer_churn\Churn_project\Telecom_Test.csv')
train = data.copy()
test = data_test.copy()

# understanding our data by small steps and visual tools
def understand_our_data(data):
    print('Our Data Info','\n')
    print(data.info(),'\n')
    print('Describe our data','\n')
    print(data.describe(),'\n')
    print('Objects columns','\n')
    print(data.dtypes == 'object','\n')
    print('Sorted type of columns','\n')
    print(data.dtypes.sort_values(),'\n')
    print('Number of null values','\n')
    print(data.isna().sum().sort_values(),'\n')
    print('Shape of our Data','\n')
    print(data.shape,'\n')
    print('Percnt of test data comparing of train data','\n')
    print(data.shape[0]/test.shape[0],'\n')
    print('Number of unique vales','\n')
    print(data.nunique().sort_values(),'\n')
    
understand_our_data(train)
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
2) state,area_code,international_plan,voice_mail_plan, churn all these
    data object  
3) we have 21 columns and 3333 rows
4) our test data is half train data (1.99)
5) account_length how many days customer was with company    
6) we are studying in 51 state
customer chun the perecantge of customer who stop doing business with any

'''

# Clean The Data
train = train.drop('Unnamed: 0', axis =1)
test = test.drop('Unnamed: 0', axis =1)

# convert category to binary 
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

# Check othe catogries data
plt.hist(list(train['area_code']))

plt.figure(figsize= (20,10))
plt.hist(list(train['state']), bins = 100)

# how it become
train.mean()

# Our Target 
# visulization pie plot 
topie = train['churn'].value_counts(sort =True)
colors = ['darkgreen', 'red']
plt.pie(topie, colors = colors, explode = [0,0.2], autopct='%1.1f%%',
        shadow =True, startangle=90)
plt.title('% churn in Training Data')
plt.show()

'''
We can say 0.15 of people will leave the company
 at the end of this year.

'''













