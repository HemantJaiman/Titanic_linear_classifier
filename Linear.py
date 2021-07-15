

#Data preprocessing

import numpy as np
import pandas as pd


alpha=0.0000012
sol_list=[]
t = np.zeros(6)
cost_before=1

data = pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

testing_f = test.drop(['PassengerId','Name','Ticket','Fare','Cabin','Embarked'],axis=1)
testing_l=[]
final_list=test.iloc[:,0]



training_f=data.drop(['PassengerId','Survived','Name','Ticket','Fare','Cabin','Embarked'],axis=1)
training_l=data['Survived']

new_col = np.ones(len(training_f))
training_f.insert(loc=0,column='t0',value=new_col)


training_f['Age']=training_f['Age'].fillna(training_f['Age'].mean())
testing_f['Age']=testing_f['Age'].fillna(testing_f['Age'].mean())

training_f.Sex = pd.Categorical.from_array(training_f.Sex).codes
testing_f.Sex = pd.Categorical.from_array(testing_f.Sex).codes

#Linear Regression       
for itr in range(0,5000):

    
    features = training_f.as_matrix()
    hypo = np.matmul(t.T,features.T)
    
    sum_cost=0
    cost=0
    for i in range(0,891):
        sum_cost=sum_cost+( hypo[i] - training_l[i])
    cost=(sum_cost)/(2*len(training_f))
    cost=cost**2
    
    
    # Gradient Descend
    
    for i in range(0,891):
        for j in range(0,len(t)):
            t[j]=t[j]-alpha*((hypo[i]-training_l[i])*features[i,j])
            
    print('the value of iteration no '+ str(itr) +'is  ' +str(cost))

 

    cost_before=cost

prediction = np.matmul(t.T,features.T)    

for i in range(0,418):
    
    if (prediction[i] >= 0.5):
        sol_list.append(1)
        
    else:
        sol_list.append(0)
        
print(sol_list)     


final_out=list(zip(final_list,sol_list))
sub=pd.DataFrame(final_out)


sub.to_csv("Linear_out.csv",index=False)
