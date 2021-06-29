#importing all neccessary libraries

import streamlit as st#for building the webapp
import numpy as np
import pandas as pd#for loading datasets
import seaborn as sns#for plotting 
import matplotlib.pyplot as plt#for plotting



import time



from sklearn.model_selection import train_test_split#for splitting the data sets ito training and test
from sklearn import model_selection
from sklearn import datasets



#for training the model on differnt algortihms
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression



#for calculating accuracy and making confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score




st.title('Data Analysis and Prediction ML Webapp')


#progress bar
progress=st.progress(0)
for i in range(100):
	time.sleep(0.01)
	progress.progress(i+1)


st.markdown("""
:sunglasses:
""")

#main function-streamlit structure design
def main():
	activities=['EDA📈','Visualisation 📊','Model🛠','About App📱','Contact Us 📞']
	option=st.sidebar.selectbox('Select Option:',activities)





#EDA part
	if option=='EDA📈':
		st.markdown("""
		## Exploratory Data Analysis
        """)
		st.warning("Only CSV formats datasets are allowed for now")
		data=st.file_uploader("Upload Your Dataset",type=['csv'])
		if data is not None:
			st.success("Data has been loaded suceesfully")
			df=pd.read_csv(data)
			st.dataframe(df.head(10))
			df1=df

			if st.checkbox("Display Shape"):
				st.write(df.shape)

			if st.checkbox("Display Columns Names"):
				st.write(df.columns)

			if st.checkbox("Select multiple columns"):
				selected_columns=st.multiselect('Select prefered columns',df.columns)
				df1=df[selected_columns]
				st.dataframe(df1)


			if st.checkbox("Display Count of Null values in column"):
				st.write(df1.isnull().sum())
				
                

			if st.checkbox("Display columns data types"):
				if df1.empty:
					st.write(df.dtypes)
				else:
					st.write(df1.dtypes)


			if st.checkbox("Display Summery"):
				if df1.empty:
					st.write(df.describe().T)
				else:
					st.write(df1.describe().T)
				

			if st.checkbox("Display correation between columns"):
				st.write(df1.corr())






#Viualisation part
	elif option=='Visualisation 📊':
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






#model building part
	elif option=='Model🛠':
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
			if st.checkbox("Build Confusion matrix"):
				acc=confusion_matrix(y_test,y_pred)
				acc






#About app part
	elif option=='About App📱':
		st.write("gfgwrfr")
		

		

            




#Contact Us part
	else:
		st.write("dhhhgh")


hide_streamlit_style = """
            <style>
            
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)






if __name__ == '__main__':
	main()
