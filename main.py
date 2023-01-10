import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


data = pd.read_excel('athletes.xlsx', sheet_name='athletes')
#print(data.isna().sum())
data=data.dropna()

score=data['gold']*7 + data['silver']*5 + data['bronze']*4
data['score']=score

data['man']=LabelEncoder().fit_transform(data['sex'])

model=LogisticRegression()

X=data[['nationality','score','height','weight','sport']]
Y=data[['man']]
X=pd.get_dummies(X, columns=['nationality','sport'])

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state = 0)

model.fit(X_train, Y_train)

predictions = model.predict_proba(X_test)

print(model.score(X_train,Y_train))
print(model.score(X_test,Y_test))

tp, fn, fp, tn = 0, 0, 0, 0

def fp_tp_count(bound):
    global tp, fn, fp, tn
    tp, fn, fp, tn = 0, 0, 0, 0
    for predicted_prob, actual in zip(predictions[:,1], Y_test['man']):
        if predicted_prob > bound:
            if actual==1:
                tp+=1
            else:
                fp+=1

        else:
            if actual==1:
                fn+=1
            else:
                tn+=1

fp_tp_count(0.5)
print('tp=',tp,' ','fp=',fp,' ', 'tn=', tn, 'fn=', fn)

fprR, tprR, thresR = roc_curve(Y_test['man'],predictions[:,1])

plt.plot(fprR,tprR)
plt.title('ROC-кривая с помощью функции roc_curve')
plt.plot([0,1],[0,1])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.grid()
plt.show()

print('roc_auc_score=', roc_auc_score(Y_test['man'], predictions[:,1]))

FPR, TPR= 0, 0
def FPR_TPR_count():
    global FPR, TPR
    FPR, TPR = 0, 0
    FPR=fp/(fp+tn)
    TPR=tp/(tp+fn)

FPR_L=[]
TPR_L=[]
Precision=0
Recall=0

def precision_recall_count():
    global Precision, Recall
    Precision, Recall = 0, 0
    Precision=tp/(tp+fp)
    Recall=tp/(tp+fn)

Precision_list=[]
Recall_list=[]

for i in range(0,105,5):
    k=i/100
    fp_tp_count(k)
    FPR_TPR_count()
    FPR_L.append(FPR)
    TPR_L.append(TPR)

plt.plot(FPR_L,TPR_L)
plt.title('ROC-кривая при помощи подсчета True positive rate и False positive rate')
plt.plot([0,1],[0,1])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.grid()
plt.show()


plt.plot(FPR_L,TPR_L)
plt.plot(fprR,tprR)
plt.title('Сравнение двух ROC-кривых')
plt.legend(['Самостоятельная реализация', 'При помощи готовой функции'],loc ="lower right")
plt.plot([0,1],[0,1])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.grid()
plt.show()

for i in range (5,100,5):
    k = i / 100
    fp_tp_count(k)
    precision_recall_count()
    Precision_list.append(Precision)
    Recall_list.append(Recall)

plt.plot(Recall_list,Precision_list)
plt.title('Precision-Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid()
plt.show()