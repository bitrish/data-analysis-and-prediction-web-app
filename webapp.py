import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import model_selection

st.title('The Data Analysis and Prediction Web App')

def main():
	activities=['EDA','Visualisation','Model','About us']
	option=st.sidebar.selectbox('Select Option:',activities)

	if option=='EDA':
		st.subheader("Exploratory Data Analysis")
		data=st.file_uploader("Upload Your Dataset",type=['csv','xlsx','txt','json'])
		if data is not None:
			st.success("Data has been loaded suceesfully")
			df=pd.read_csv(data)
			st.dataframe(df.head(10))
			if st.checkbox("Display Shape"):
				st.write(df.shape)
			if st.checkbox("Display Columns"):
				st.write(df.columns)
			if st.checkbox("Select multiple columns"):
				selected_columns=st.multiselect('Select prefered columns',df.columns)
				df1=df[selected_columns]
				st.dataframe(df1)
			if st.checkbox("Display Summery"):
				st.write(df.describe().T)

			if st.checkbox("Display Null values"):
				st.write(df.isnull().sum())
			if st.checkbox("Display thier data types"):
				st.write(df.dtypes)
			if st.checkbox("Display correation of various columns"):
				st.write(df.corr())


	elif option=='Visualisation':
		st.subheader("Data Visualisation")
		data=st.file_uploader("Upload Your Dataset",type=['csv','xlsx','txt','json'])
		if data is not None:
			st.success("Data has been loaded suceesfully")
			df=pd.read_csv(data)
			st.dataframe(df.head(10))

			if st.checkbox('Select Multiple columns to plot'):
				selected_columns=st.multiselect('Select your preffered coulmns',df.columns)
				df1=df[selected_columns]
				st.dataframe(df1)
			if st.checkbox('Display Heatmap'):
				st.write(sns.heatmap(df1.corr(),vmax=1,square=True,annot=True,cmap='viridis'))
				st.pyplot()
			if st.checkbox('Display Pairplot'):
				st.write(sns.pairplot(df1,diag_kind='kde'))
				st.pyplot()
			if st.checkbox('Display Pie Chart'):
				all_columns=df.columns.to_list()
				pie_columns=st.selectbox("Select Columns to Display",all_columns)
				pieChart=df[pie_columns].value_counts().plot.pie(autopct='%1.1f%%')
				st.write(pieChart)
				st.pyplot()


	elif option=='Model':
		st.subheader("Build your own model with different algorithms")
		data=st.file_uploader("Upload Your Dataset",type=['csv','xlsx','txt','json'])
		if data is not None:
			st.success("Data has been loaded suceesfully")
			df=pd.read_csv(data)
			st.dataframe(df.head(10))
			if st.checkbox('Select multiple columns '):
				new_data=st.multiselect("Select yout preffered columns,Please select target column as the last columns",df.columns)
				df1=df[new_data]
				st.dataframe(df1)

				X=df1.iloc[:,0:-1]
				y=df1.iloc[:,-1]

			seed=st.sidebar.slider('Seed',1,200)
			classifier_name=st.sidebar.selectbox('Select your preffered classifier',('KNN','SVM','LR','Naive_bayes','Decision trees'))

			def add_parameter(name_of_clf):
				param=dict()
				if name_of_clf=='SVM':
					C=st.sidebar.slider('C',1,15)
					param['C']=C;
				if name_of_clf=='KNN':
					K=st.sidebar.slider('K',1,15)
					param['K']=K;
				return param

			param=add_parameter(classifier_name)


			def get_classifier(name_of_clf,param):
				clf=None
				if name_of_clf=='SVM':
					clf=SVC(C=param['C'])
				elif name_of_clf=='KNN':
					clf=KNeighborsClassifier(n_neighbors=param['K'])
				elif name_of_clf=='LR':
					clf=LogisticRegression()
				elif name_of_clf=='naive_bayes':
					clf=GaussianNB()
				elif name_of_clf=='Decision trees':
					clf=DecisionTreeClassifier()
				else:
					St.warning("Select your choice of algorithm")

				return clf


			clf=get_classifier(classifier_name,param)

			X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=seed)
			clf.fit(X_train,y_train)

			y_pred=clf.predict(X_test)
			st.write("Predictions:",y_pred)

			accuracy=accuracy_score(y_test,y_pred)
			st.write("Name of classifier:", classifier_name)
			st.write("Accuracy:",accuracy)












	elif option=='About us':
		st.write("dhhhgh")

if __name__ == '__main__':
	main()
