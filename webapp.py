#importing all neccessary libraries

import streamlit as st#for building the webapp
import streamlit.components as stc

import numpy as np
import pandas as pd#for loading datasets
import seaborn as sns#for plotting 
import matplotlib.pyplot as plt#for plotting




from sklearn.model_selection import train_test_split#for splitting the data sets ito training and test
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn import datasets
#from pandas_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report


#for file dowloads feature 
import base64
import time
timestr=time.strftime("%Y%m%d-%H%M%S")

#for training the model on differnt algortihms
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression



#for calculating accuracy and making confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#for computing missing values
from sklearn.impute import SimpleImputer



#html_temp="""
#<div style="background-color:grey">st.title('Data Analysis and Prediction ML Webapp')</div>
#"""
#st.markdown(html_temp,unsafe_allow_html=True)
st.title('Data Analysis and Prediction Web Application')


#progress bar
progress=st.progress(0)
for i in range(100):
	time.sleep(0.01)
	progress.progress(i+1)


st.markdown("""
:sunglasses:
""")
st.set_option('deprecation.showPyplotGlobalUse', False)



#main function-streamlit structure design
def main():
	activities=['EDA📈','Visualisation 📊','Feature Engineering⛏','Model🛠','Contact Us 📞']
	option=st.sidebar.selectbox('Select Option:',activities)



#EDA part
	if option=='EDA📈':
		st.markdown("""
		## Exploratory Data Analysis
        """)
		st.info("Only CSV formats datasets are supported for now")
		data=st.file_uploader("Upload Your Dataset",type=['csv'])
		if data is not None:
			st.success("Data has been loaded suceesfully")
			df=pd.read_csv(data)
			df1=df


            #showing data
			if st.checkbox("Show Dataset"):
				number=st.number_input("No of rows to view",1,100000)
				st.dataframe(df1.head(number))


            #showing shape of the data
			if st.checkbox("Display Shape"):
				st.write(df1.shape)



            #show column names
			if st.checkbox("Display Columns Names"):
				st.write(df1.columns)

			#show columns data types
			if st.checkbox("Display columns data types"):
				st.write(df1.dtypes)


			#show value counts of target column
			if st.checkbox("Show values counts of target column"):
				st.warning("Make sure that your target/output variable is in the last column")
				st.text("Counts")
				st.write(df1.iloc[:,-1].value_counts())


			#show the correlation of data columns
			if st.checkbox("Display correation between columns"):
				st.write(df1.corr())




            #select multipme columns
			if st.checkbox("Select multiple columns"):
				selected_columns=st.multiselect('Select prefered columns',df.columns)
				if len(selected_columns)!=0:
					df1=df[selected_columns]
				st.dataframe(df1)


            #null values count and plot
			if st.checkbox("Display Count of Null values in column"):
				st.write(df1.isnull().sum())
				if st.checkbox("Display percentage of miising values in columns"):
					missing_percentage=df1.isna().sum().sort_values(ascending=False)/len(df1)
					st.write(missing_percentage)
				if st.checkbox("Visualise null values in columns"):
					st.write(sns.heatmap(df1.isnull(),yticklabels=False,cbar=False,cmap='viridis'))
					st.pyplot()


			
                       
   
            #show the summery of dataset
			if st.checkbox("Display Summery"):
				if df1.empty:
					st.write(df.describe().T)
				else:
					st.write(df1.describe().T)
				


			#if st.checkbox("Create profile report"):
				#pr=ProfileReport(df1,explorative=True)
				#st.header("**Pandas profiling report**")
				#st_profile_report(pr)
			








#Viualisation part
	elif option=='Visualisation 📊':
		st.subheader("Data Visualisation")
		st.info("Only CSV formats datasets are supported for now")
		data=st.file_uploader("Upload Your Dataset",type=['csv'])
		if data is not None:
			st.success("Data has been loaded suceesfully")
			df=pd.read_csv(data)
			df1=df
			

            #show dataset
			if st.checkbox("Show Dataset"):
				number=st.number_input("No of rows to view",1,100000)
				st.dataframe(df1.head(number))


            #selecting columns
			if st.checkbox('Select Multiple columns to plot'):
				selected_columns=st.multiselect('Select your preffered coulmns',df.columns)
				if len(selected_columns)!=0:
					df1=df[selected_columns]
				st.dataframe(df1)


			#display heatmap
			if st.checkbox('Display Heatmap'):
				st.write(sns.heatmap(df1.corr(),vmax=1,square=True,annot=True,cmap='viridis'))
				st.pyplot()


			#display pair plot
			if st.checkbox('Display Pairplot'):
				st.write(sns.pairplot(df1,diag_kind='kde'))
				st.pyplot()


			#display pie chart
			if st.checkbox('Display Pie Chart'):
				all_columns=df.columns.to_list()
				pie_columns=st.selectbox("Select Column to Display",all_columns)
				if st.button("Generate chart"):
					st.success("Generating pie chart") 
					pieChart=df1[pie_columns].value_counts().plot.pie(autopct='%1.1f%%')
					st.write(pieChart)
					st.pyplot()
				
				
			#display customisable plot
			if st.checkbox('Display Customisable plots of your choice'):
				all_column_names=df1.columns.tolist()
				type_of_plot=st.selectbox("Select type of plot",["area","bar","line","hist","box","kde"])
				selected_columns_names=st.multiselect("Select columns to plot",all_column_names)
				if st.button("Generate plot"):
					st.success("Generating customisable plot of type {} for {}".format(type_of_plot,selected_columns_names))


					#plot by streamlit
					if type_of_plot=="area":
						cust_data=df1[selected_columns_names]
						st.area_chart(cust_data)
					elif type_of_plot=="bar":
						cust_data=df1[selected_columns_names]
						st.bar_chart(cust_data)
					elif type_of_plot=="line":
						cust_data=df1[selected_columns_names]
						st.line_chart(cust_data)


					#plot by seaborn/matplotlib
					elif type_of_plot:
						cust_plot=df1[selected_columns_names].plot(kind=type_of_plot)
						st.write(cust_plot)
						st.pyplot()



   #Feature Engineering
	elif option=='Feature Engineering⛏':
		st.markdown("""
		## Feature Engineering
        """)
		st.info("Only CSV formats datasets are supported for now")
		data=st.file_uploader("Upload Your Dataset",type=['csv'])
		if data is not None:
			st.success("Data has been loaded suceesfully")
			df=pd.read_csv(data)
			df1=df


            #showing data
			if st.checkbox("Show Dataset"):
				number=st.number_input("No of rows to view",1,100000)
				st.dataframe(df1.head(number))


			if st.checkbox("take care of missing values"):
				X=df1.iloc[:,:-1].values
				y=df1.iloc[:,-1].values

				
				strategy=['mean', 'median', 'most_frequent']
				selected_strategy=st.selectbox('Select On what strategy you want to fill your missing values',strategy)
				st.success("Congracts your missing values are filled with {}".format(selected_strategy))
				new=np.column_stack((X,y))
				dff=pd.DataFrame.from_records(new)
				
				



                
                
				if st.checkbox("Download the csv file with filled missing values"):
					csvfile= df1.to_csv()
					b64= base64.b64encode(csvfile.encode()).decode()
					new_filename= "new_csv_file_{}_.CSV". format(timestr)
					st.markdown ("#### Download File ###")
					href =f'<a href="data: file/csv;base64, {b64}" download="{new_filename}">Click Here!!</a>'
					st.markdown (href, unsafe_allow_html=True)
                    



    #Model building
	elif option=='Model🛠':
		st.subheader("Build your own model with different classifiers")
		st.info("Only CSV formats datasets are supported for now")
		data=st.file_uploader("Upload Your Dataset",type=['csv'])
		if data is not None:
			st.success("Data has been loaded suceesfully")
			df=pd.read_csv(data)
			df1=df
			#show dataset
			if st.checkbox("Show Dataset"):
				number=st.number_input("No of rows to view",1,100000)
				st.dataframe(df1.head(number))


			if st.checkbox('Select multiple columns '):
				new_data=st.multiselect("Select yout prefered columns,Please select target column as the last columns",df.columns)
				if len(new_data)!=0:
					df1=df[new_data]
				st.dataframe(df1)


			X=df1.iloc[:,0:-1]
			y=df1.iloc[:,-1]
			
			#imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
			#imputer.fit(X[:,0:-1])
			#X[:,0:-1]=imputer.transform(X[:,0:-1])
            #used to fix the set of train test split  every time we perfrom train test split function

			seed=st.sidebar.slider('Random state Seed',1,200)
			classifier_name=st.sidebar.selectbox('Select your preffered classifier',('KNN','SVM','LR','Naive bayes','Decision trees'))
			def add_parameter(name_of_clf):
				param=dict()
				if name_of_clf=='SVM':
					C=st.sidebar.slider('C',1,100)
					param['C']=C;
				if name_of_clf=='KNN':
					K=st.sidebar.slider('K',1,100)
					param['K']=K;
				if name_of_clf=='Decision trees':
					max_depth=st.sidebar.slider('max_depth',1,100)
					param['max_depth']=max_depth;
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
				elif name_of_clf=='Naive bayes':
					clf=GaussianNB()
				elif name_of_clf=='Decision trees':
					clf=DecisionTreeClassifier(max_depth=param['max_depth'])
				else:
					st.warning("Select your choice of algorithm")

				return clf


			clf=get_classifier(classifier_name,param)

			X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=seed)
			clf.fit(X_train,y_train)
			y_pred=clf.predict(X_test)
			if st.checkbox("Make prediction on X_test data"):
				st.write(y_pred)
			accuracy=accuracy_score(y_test,y_pred)
			st.write("Name of classifier:", classifier_name)
			if st.checkbox("Calculate acuuracy of the model"):
				st.write("Accuracy of the Model:",accuracy)
			if classifier_name=='Decision trees':
				if st.checkbox("Apply grid search CV to find the best parameter values for the model"):
					param_dict={"max_depth":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
					grid=GridSearchCV(model,param_grid=param_dict,n_jobs=-1)
					grid.fit(X_train,y_train)
					st.write(grid.best_score_)
					if st.write("Get the best parameter value"):
						st.write(grid.best_params_)
					

			if st.checkbox("Build Confusion matrix"):
				acc=confusion_matrix(y_test,y_pred)
				acc
			









           


hide_streamlit_style = """
            <style>
            
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)






if __name__ == '__main__':
	main()
