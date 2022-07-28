
# Imports needed for requirements.txt
import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import sweetviz as sv
import streamlit.components.v1 as components
import codecs
from streamlit_plotly_events import plotly_events
from scipy.optimize import curve_fit
from matplotlib import pyplot
from numpy import arange
from bokeh.models.widgets import Div

st.markdown('''
# **Customer Segmentation**
**Customer Segmentation** is the process of division of a customer base into several groups.
''')

st.image("audience-segmentation-concept-man-near-a-large-circular-chart-with-images-of-people-illustration-flat-vector.jpg")

st.write("📊  These groups share similarities that are relevant to marketing such as gender, age, annual income and spending habits.")
st.write("📊  Once your company understands the characteristics of these 'clusters' of clients you can divert your ad budget away from those who are unlikely to purchase your product or service towards your most valuable customers")
st.write("📊  This customer segmentation will be completed on our **Mall Dataset**")

if st.button('Press me for Customer Segmentation'):
        @st.cache(allow_output_mutation=True)
        def load_data():
            a = pd.read_csv("Mall_Customers.csv")
            return a
        df = load_data() 

        #-- Preparing KMeans Elbow
        X1 = df[['Age' , 'Spending Score (1-100)']].iloc[: , :].values
        inertia = []
        for n in range(1 , 11):
            algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                            tol=0.0001,  random_state= 111  , algorithm='elkan') )
            algorithm.fit(X1)
            inertia.append(algorithm.inertia_)

        #-- Plotting Kmeans Elbow
        st.subheader("KMeans Elbow")
        fig, ax = plt.subplots()
        plt.plot(np.arange(1 , 11) , inertia , 'o')
        plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
        plt.xlabel("Optimal Number of Clusters")
        plt.ylabel("Distortion Score")
        st.pyplot(fig)

        st.write("📊 The first step in any customer segmentation is to work out the optimal number of groups of customers or 'clusters'") 
        st.write("📊 Following the implementation of a KMeans algorithm, the above graph shows us that in this case the optimal number of clusters is **SIX**")

        st.write("---")
        #----Creating Segemenation Illustration
        algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 
                            tol=0.0001,  random_state= 111  , algorithm='elkan') )
        algorithm.fit(X1)
        labels1 = algorithm.labels_
        centroids1 = algorithm.cluster_centers_

        h = 0.02
        x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
        y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 

        st.header("Customer Segmentation: Age and Spending Score")
        fig, ax = plt.subplots()
        plt.clf()
        Z = Z.reshape(xx.shape)
        plt.imshow(Z , interpolation='nearest', 
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

        plt.scatter( x = 'Age' ,y = 'Spending Score (1-100)' , data = df , c = labels1 , 
                s = 100 )
        plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 200 , c = 'red' , alpha = 0.7)
        plt.xlabel("Age")
        plt.ylabel("Spending Score (1-100)")
        st.pyplot(fig)

        st.write(" 📊  Machine learning models are powerful decision-making tools. They can precisely identify customer segments, which is much harder to do manually or with conventional analytical methods.")
        st.write("📊  Above ⬆︎ we can see a visual representation of a customer segmentation on our **Mall Dataset**")
        st.write("📊  In this case the clusters have been segmented based on Age and Spending Score into 6 seperate groups")

        ##### 3D Vis
        X3 = df[['Age' , 'Annual Income (k$)' ,'Spending Score (1-100)']].iloc[: , :].values
        inertia = []
        for n in range(1 , 11):
            algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                                tol=0.0001,  random_state= 111  , algorithm='elkan') )
            algorithm.fit(X3)
            inertia.append(algorithm.inertia_)

        algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 
                                tol=0.0001,  random_state= 111  , algorithm='elkan') )
        algorithm.fit(X3)
        labels3 = algorithm.labels_
        centroids3 = algorithm.cluster_centers_

        st.subheader("3D Customer Segmentation")
        st.write("📊  Feel free to play around with our 3D segmentation. If its a little confusing dont worry we provide further insights below!")
        df['label3'] =  labels3
        trace1 = go.Scatter3d(
                    x= df['Age'],
                    y= df['Spending Score (1-100)'],
                    z= df['Annual Income (k$)'],
                    mode='markers',
                    marker=dict(
                        color = df["label3"], 
                        size= 15,
                        line=dict(
                            color= df['label3'],
                            width= 12,
                        ),
                        opacity=0.7
                    )
                )
        data = [trace1]
        layout = go.Layout(
                    margin=dict(
                    l=1,
                    r=1,
                    b=1,
                    t=1
                    ),
                    scene = dict(
                            xaxis = dict(title  = 'Age'),
                            yaxis = dict(title  = 'Spending Score'),
                            zaxis = dict(title  = 'Annual Income')
                        )
                        )
        fig = go.Figure(data=data, layout=layout)
        plotly_events(fig, click_event=False, hover_event=False)
        #New 3D Customer Segmentation

        st.write("---")

        ####---- Extra work to make the below work!
        st.subheader("Customer Segmentation Insights")
        st.write("📊  For the following graphics please press the ⤡ button for a better view!")
        X3 = df[['Age' , 'Annual Income (k$)' ,'Spending Score (1-100)']].iloc[: , :].values
        inertia = []
        for n in range(1 , 11):
            algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                            tol=0.0001,  random_state= 111  , algorithm='elkan') )
            algorithm.fit(X3)
            inertia.append(algorithm.inertia_)

        algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 
                            tol=0.0001,  random_state= 111  , algorithm='elkan') )
        algorithm.fit(X3)
        labels3 = algorithm.labels_
        centroids3 = algorithm.cluster_centers_

        df['label3'] =  labels3


        df4 = df.copy()
        df4.rename(columns ={"label3":"Cluster"}, inplace = True)

        grouped_km = df4.groupby(['Cluster']).mean().round(1)
        grouped_km2 = df4.groupby(['Cluster']).mean().round(1).reset_index()
        grouped_km2['Cluster'] = grouped_km2['Cluster'].map(str)
        grouped_km2.drop(columns =["CustomerID"], inplace = True)

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(grouped_km2["Spending Score (1-100)"], grouped_km2["Annual Income (k$)"], grouped_km2["Age"],color=['yellow','red','green','orange','blue','purple'],alpha=0.5,s=500)

        # add annotations one by one with a loop
        for line in range(0,grouped_km.shape[0]):
            ax.text(grouped_km2['Spending Score (1-100)'][line], grouped_km2['Annual Income (k$)'][line],grouped_km2['Age'][line], s=('Cluster \n'+grouped_km2['Cluster'][line]), horizontalalignment='center', fontsize=12, fontweight='light', fontfamily='serif')

        ax.set_xlabel("Spending Score (1-100)", fontsize = 12)
        ax.set_ylabel("Annual Income (k$)",fontsize = 12)
        ax.set_zlabel("Age", fontsize = 12)

        fig.text(0.15, .95, '3D Plot: Clusters Visualized', fontsize=20, fontweight='bold', fontfamily='sans-serif')
        fig.text(0.15, .9, 'Clusters by averages in 3D.', fontsize=15, fontweight='light', fontfamily='sans-serif')

        fig.text(1.172, 0.95, 'Insight', fontsize=20, fontweight='bold', fontfamily='sans-serif')

        fig.text(1.172, 0.3, '''
        We observe a clear distinction between clusters. 

        As a business, we might want to rename our clusters
        so that they have a clear & obvious meaning; right now
        the cluster labels mean nothing. 

        Let's change that:

        Cluster 0 - Middle spending score, Middle income, High age - Valuable

        Cluster 1 - High spending score, High income, Young age - Most Valuable

        Cluster 2 - Lowest spending score, High income, High age - Less Valuable

        Cluster 3 - High spending score, Low income, Young age - Very Valuable.

        Cluster 4 - Low spending score, Low income, High age - Least Valuable

        Cluster 5 - Middle spending score, Middle income, Young age - Targets.
        '''
                , fontsize=20, fontweight='light', fontfamily='sans-serif')

        import matplotlib.lines as lines
        l1 = lines.Line2D([1, 1], [0, 1], transform=fig.transFigure, figure=fig,color='black',lw=0.2)
        fig.lines.extend([l1])
        st.pyplot(fig)


        st.markdown(
           """
        | Rank | Order of Importance (Customer Segmentation) | Recommendations |
        | --- | --- | --- |
        | 1 | **Targets** |  Untapped potential. Customers with massive upside if targeted correctly |
        | 2 | **Most Valuable** | High priority customers. "Whales" |
        | 3 | **Very Valuable** | Priority customers |
        | 4 | **Valuable** | Customers that should be maintained |
        | 5 | **Less Valuable** | Customers to pay less attention to |
        | 6 | **Least Valuable** | Customers to direct marketing away from |

        """)

        st.write("📊  It is important to remember that these rankings and recommendations are our thanks to **HAB LABS** expertise and experience. Customer segmentation is subjective by nature.")

        st.write("---")

        #Percentages BarPlot by Gender

        df4['Cluster_Label'] = df4['Cluster'].apply(lambda x: 'Less Valuable' if x == 0 else 
                                               'Targets' if x == 1 else
                                               'Valuable' if x == 2 else
                                               'Most Valuable' if x == 3 else
                                               'Least Valuable' if x == 4 else 'Very Valuable')

        # New column for radar plots a bit later on 

        df4['Sex (100=Male)'] = df4['Gender'].apply(lambda x: 100 if x == 'Male' else 0)

        df4['Cluster'] = df4['Cluster'].map(str)
        # Order for plotting categorical vars
        Cluster_ord = ['0','1','2','3','4','5']
        clus_label_order = ['Targets','Most Valuable','Very Valuable','Valuable','Less Valuable','Least Valuable']


        clus_ord = df4['Cluster_Label'].value_counts().index

        clu_data = df4['Cluster_Label'].value_counts()[clus_label_order]
        ##

        data_cg = df4.groupby('Cluster_Label')['Gender'].value_counts().unstack().loc[clus_label_order]
        data_cg['sum'] = data_cg.sum(axis=1)

        ##
        data_cg_ratio = (data_cg.T / data_cg['sum']).T[['Male', 'Female']][::-1]


        ### Plotting
        fig, ax = plt.subplots(1,1,figsize=(18, 10))

        ax.barh(data_cg_ratio.index, data_cg_ratio['Male'], 
                color='cadetblue', alpha=0.7, label='Male')
        ax.barh(data_cg_ratio.index, data_cg_ratio['Female'], left=data_cg_ratio['Male'], 
                color='coral', alpha=0.7, label='Female')


        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_yticklabels((data_cg_ratio.index), fontfamily='sans-serif', fontsize=14)


        # male percentage
        for i in data_cg_ratio.index:
            ax.annotate(f"{data_cg_ratio['Male'][i]*100:.3}%", 
                        xy=(data_cg_ratio['Male'][i]/2, i),
                        va = 'center', ha='center',fontsize=14, fontweight='light', fontfamily='sans-serif',
                        color='white')

        for i in data_cg_ratio.index:
            ax.annotate(f"{data_cg_ratio['Female'][i]*100:.3}%", 
                        xy=(data_cg_ratio['Male'][i]+data_cg_ratio['Female'][i]/2, i),
                        va = 'center', ha='center',fontsize=14, fontweight='light', fontfamily='sans-serif',
                        color='#244247')


        fig.text(0.129, 0.98, 'Gender Distribution by Cluster', fontsize=20, fontweight='bold', fontfamily='serif')   
        fig.text(0.129, 0.88, 
                '''
        We see that females dominate most of our categories; except our Targets cluster.
        How might we encourage more male customers?
        Incentive programs for females in the Targets cluster?''' , fontsize=14,fontfamily='serif')   

        for s in ['top', 'left', 'right', 'bottom']:
            ax.spines[s].set_visible(False)

        ax.legend().set_visible(False)

        fig.text(0.777,0.98,"Male", fontweight="bold", fontfamily='serif', fontsize=18, color='cadetblue')
        fig.text(0.819,0.98,"|", fontweight="bold", fontfamily='serif', fontsize=18, color='black')
        fig.text(0.827,0.98,"Female", fontweight="bold", fontfamily='serif', fontsize=18, color='coral')
        st.pyplot(fig)

        st.write("---")

        st.markdown('''
        # **FREE CONSULTATION**
        If you like what you see what else we can do at HAB LABS visit:
        ''')
        link = '[Free Consultation](https://hablabs.tech)'
        st.markdown(link, unsafe_allow_html=True)
        
         
            
        

  


        

            
                






        
