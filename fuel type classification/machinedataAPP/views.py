from django.shortcuts import render
from django.shortcuts import render
from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import login ,logout,authenticate
from django.shortcuts import get_object_or_404
import json
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
# Create your views here.
def home(request):
    return render(request,'home.html')
def register(request):
    if request.method == 'POST':
        First_Name = request.POST['name']
        Email = request.POST['email']
        username = request.POST['username']
        password = request.POST['password']
        confirmation_password = request.POST['cnfm_password']
        if password == confirmation_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists, please choose a different one.')
                return redirect('register')
            else:
                if User.objects.filter(email=Email).exists():
                    messages.error(request, 'Email already exists, please choose a different one.')
                    return redirect('register')
                else:
                    user = User.objects.create_user(
                        username=username,
                        password=password,
                        email=Email,
                        first_name=First_Name
                    )
                    user.save()
                    return redirect('login')
        else:
            messages.error(request, 'Passwords do not match.')
        return render(request, 'register.html')
    return render(request, 'register.html')

def login_view(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        if User.objects.filter(username=username).exists():
            user=User.objects.get(username=username)
            if user.check_password(password):
                user = authenticate(username=username,password=password)
                if user is not None:
                    login(request,user)
                    messages.success(request,'login successfull')
                    return redirect('/')
                else:
                   messages.error(request,'please check the Password Properly')
                   return redirect('login')
            else:
                messages.error(request,"please check the Password Properly")  
                return redirect('login') 
        else:
            messages.error(request,"Username does not exist.")
            return redirect('login')
    return render(request,'login.html')
# Load and preprocess the dataset
def logout_view(request):
    logout(request)
    return redirect('login')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import warnings
warnings.filterwarnings('ignore')

from django.core.files.storage import default_storage
labels=['PETROL', 'DIESEL', 'BATTERY', 'CNG PETROL', 'PETROL LPG',
       'CNG', 'PETROL ELECTRIC', 'LPG', 'PETROL CNG']
global label_encoder
dataset=pd.read_csv(r'static/Dataset.csv')
categorical_columns = ['modelDesc', 'fuel', 'colour', 'vehicleClass', 'secondVehicle', 'category', 'makerName','tempRegistrationNumber','OfficeCd']

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to each categorical column
for col in categorical_columns:
    if col in dataset.columns:
        dataset[col] = label_encoder.fit_transform(dataset[col])

def Upload_data(request):
    load=True
    global df_resampled,label_encoder
    global df_train,df_test 
    global  X_train,X_test,y_train,y_test
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        dataset=pd.read_csv(default_storage.path(file_path))
        categorical_columns = ['modelDesc', 'fuel', 'colour', 'vehicleClass', 'secondVehicle', 'category', 'makerName','tempRegistrationNumber','OfficeCd']

        # Initialize LabelEncoder
        label_encoder = LabelEncoder()

        # Apply label encoding to each categorical column
        for col in categorical_columns:
            if col in dataset.columns:
                dataset[col] = label_encoder.fit_transform(dataset[col])
        dataset.dropna(inplace=True)
        date_columns = ['makeYear', 'fromdate', 'todate']
        # Convert specified date columns to datetime format, replacing out-of-bounds dates with NaT
        dataset[date_columns] = dataset[date_columns].apply(lambda x: pd.to_datetime(x, errors='coerce'))
        # Convert specified date columns to datetime format, replacing out-of-bounds dates with NaT
        dataset[date_columns] = dataset[date_columns].apply(lambda x: pd.to_datetime(x, errors='coerce'))
        # Convert datetime columns to integer representation
        dataset[date_columns] = dataset[date_columns].apply(lambda x: pd.to_numeric(x.dt.strftime('%Y%m%d'), errors='coerce'))
        X=dataset.drop('fuel',axis=1)
        y = dataset['fuel']
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
        html_table = f"""
        <table border="1">
            <tr>
                <th>Dataset</th>
                <th>Shape</th>
            </tr>
            <tr>
                <td>X_train</td>
                <td>{X_train.shape}</td>
            </tr>
            <tr>
                <td>y_train</td>
                <td>{y_train.shape}</td>
            </tr>
            <tr>
                <td>X_test</td>
                <td>{X_test.shape}</td>
            </tr>
            <tr>
                <td>y_test</td>
                <td>{y_test.shape}</td>
            </tr>
        </table>
            """
        default_storage.delete(file_path)
        return render(request,'prediction.html',{'predict':html_table })
    return render(request,'prediction.html',{'upload':load})
precision = []
recall = []
fscore = []
accuracy = []
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

#function to calculate various metrics such as accuracy, precision etc
def calculateMetrics(image,algorithm, testY,predict):
    testY = testY.astype('int')
    predict = predict.astype('int')
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100 

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    print(algorithm+' Accuracy    : '+str(a))
    print(algorithm+' Precision   : '+str(p))
    print(algorithm+' Recall      : '+str(r))
    print(algorithm+' FSCORE      : '+str(f))
    report=classification_report(predict, testY,target_names=labels)
    print('\n',algorithm+" classification report\n",report)
    conf_matrix = confusion_matrix(testY, predict) 
    if os.path.exists(image):
        return image
    else:
        plt.figure(figsize =(5, 5)) 
        ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="Blues" ,fmt ="g");
        ax.set_ylim([0,len(labels)])
        plt.title(algorithm+" Confusion matrix") 
        plt.ylabel('True class') 
        plt.xlabel('Predicted class') 
        plt.savefig(image)
        return image
        
def KNeighbors_model(request):
    # Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train the classifier
    knn.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred1 = knn.predict(X_test)
    image='static/images/knn.png'
    calculateMetrics(image,"KNeighborsClassifier", y_pred1, y_test)

    return render(request,'prediction.html',
                  {'algorithm':'KNeighborsClassifier',
                   'image':image,
                   'accuracy':accuracy[-1],
                   'precision':precision[-1],
                   'recall':recall[-1],
                   'fscore':fscore[-1]})
import os
import joblib
model_path = 'static/random_forest_model.pkl'

def RandomForest_model(request):
    global X_test, X_train, y_train, y_test
    # Create a Random Forest classifier
    if os.path.exists(model_path):
        # Load the existing model
        rf = joblib.load(model_path)
        print("Loaded existing RandomForest model.")
    else:
        # Train a new model
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        joblib.dump(rf, model_path)
        print("Trained and saved new RandomForest model.")
   # y_pred2 = rf.predict(X_test)
    y_pred2 = rf.predict(X_test)
    image='static/images/rfc.png'
    calculateMetrics(image,"RandomForestClassifier", y_pred2, y_test)

    return render(request, 'prediction.html', {
        'algorithm': 'RandomForestClassifier',
        'image':image,
        'accuracy': accuracy[-1],
        'precision': precision[-1],
        'recall': recall[-1],
        'fscore': fscore[-1]
    })
    
def prediction(request):
    Test = True
    rf=joblib.load(model_path)
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        test = pd.read_csv(default_storage.path(file_path))
        categorical_columns = ['modelDesc', 'fuel', 'colour', 'vehicleClass', 'secondVehicle', 'category', 'makerName','tempRegistrationNumber','OfficeCd']

        # Apply label encoding to each categorical column
        for col in categorical_columns:
            if col in test.columns:
                test[col] = label_encoder.fit_transform(test[col])
        date_columns = ['makeYear', 'fromdate', 'todate']
        # Convert specified date columns to datetime format, replacing out-of-bounds dates with NaT
        test[date_columns] = test[date_columns].apply(lambda x: pd.to_datetime(x, errors='coerce'))
        # Convert specified date columns to datetime format, replacing out-of-bounds dates with NaT
        test[date_columns] = test[date_columns].apply(lambda x: pd.to_datetime(x, errors='coerce'))
        # Convert datetime columns to integer representation
        test[date_columns] = test[date_columns].apply(lambda x: pd.to_numeric(x.dt.strftime('%Y%m%d'), errors='coerce'))
        test
        rf=joblib.load('static/random_forest_model.pkl')
        predict=rf.predict(test)
        default_storage.delete(file_path)
        data=[]
        for i,p in enumerate(predict):
            data.append({
                'row': f'Row {i+1}: { X_test.iloc[i].to_string(index=False)}',
                'message': f"Row {i + 1} classified as {labels[p]}"
            })
        return render(request, 'prediction.html', {'output': data})
    return render(request, 'prediction.html', {'test': Test})