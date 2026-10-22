"""Unified Streamlit GUI for the two supplied ML notebooks.
Run: streamlit run two_project_ml_gui.py
Install: pip install streamlit pandas numpy matplotlib seaborn scikit-learn scipy
"""
import io, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             silhouette_score, davies_bouldin_score, calinski_harabasz_score)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram

st.set_page_config(page_title='ML Project Explorer', page_icon='🧠', layout='wide')
WATER_COLUMNS = ["Date", "Q-E", "ZN-E", "PH-E", "DBO-E", "DQO-E", "SS-E", "SSV-E", "SED-E", "COND-E", "PH-P", "DBO-P", "SS-P", "SSV-P", "SED-P", "COND-P", "PH-D", "DBO-D", "DQO-D", "SS-D", "SSV-D", "SED-D", "COND-D", "PH-S", "DBO-S", "DQO-S", "SS-S", "SSV-S", "SED-S", "COND-S", "RD-DBO-P", "RD-SS-P", "RD-SED-P", "RD-DBO-S", "RD-DQO-S", "RD-DBO-G", "RD-DQO-G", "RD-SS-G", "RD-SED-G"]

def plot_show(fig):
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

def read_water(file):
    raw = file.getvalue()
    if file.name.lower().endswith('.csv'):
        df = pd.read_csv(io.BytesIO(raw), na_values=['?', ' ?', ''])
        if len(df.columns) == len(WATER_COLUMNS) and (str(df.columns[0]).lower() == 'date' or all(str(c).strip().replace('.', '', 1).isdigit() for c in df.columns[:3])):
            df = pd.read_csv(io.BytesIO(raw), header=None, names=WATER_COLUMNS, na_values='?', skipinitialspace=True)
        return df
    return pd.read_csv(io.BytesIO(raw), header=None, names=WATER_COLUMNS, na_values='?', skipinitialspace=True)

def water_app():
    st.title('💧 Water Treatment Plant Clustering Explorer')
    st.caption('Unsupervised learning • EDA • outlier analysis • PCA/t-SNE • cluster profiles')
    with st.sidebar:
        st.header('Water dataset')
        up = st.file_uploader('Upload water-treatment.data or CSV', type=['data','csv','txt'], key='water_file')
        st.divider(); st.header('Clustering settings')
        algorithm = st.selectbox('Selected algorithm', ['K-Means','Agglomerative','Gaussian Mixture (GMM)','DBSCAN'])
        k = st.slider('Number of clusters (K)', 2, 10, 3)
        eps = st.slider('DBSCAN eps', 0.1, 5.0, 1.5, 0.1)
        min_samples = st.slider('DBSCAN min_samples', 2, 30, 5)
        cap = st.checkbox('Cap outliers using IQR', value=True)
        run = st.button('Run clustering', type='primary', use_container_width=True)
    if up is None:
        st.info('Upload your Water Treatment Plant dataset from the sidebar. The original UCI file is headerless and uses commas.')
        st.markdown('**Included sections:** dataset audit, missing-value handling, feature distributions, correlation heatmap, IQR outliers, four clustering algorithms, internal metrics, PCA/t-SNE projections, cluster profiles and CSV export.')
        return
    try: raw = read_water(up)
    except Exception as e: st.error(f'Could not read dataset: {e}'); return
    if raw.empty: st.error('The uploaded file is empty.'); return
    original_dupes = int(raw.duplicated().sum())
    df = raw.drop_duplicates().copy()
    date_cols = [c for c in df.columns if str(c).strip().lower() in ('date','timestamp')]
    df = df.drop(columns=date_cols, errors='ignore')
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    if df.shape[1] < 2 or len(df) < 3: st.error('At least two numeric features and three rows are required. Check the uploaded file format.'); return
    X = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(df), columns=df.columns, index=df.index)
    X = X.drop(columns=X.columns[X.nunique() <= 1], errors='ignore')
    if X.shape[1] < 2: st.error('Fewer than two usable features remain after cleaning.'); return
    X_before = X.copy(); X_work = X.copy()
    if cap:
        for c in X_work:
            q1,q3=X_work[c].quantile(.25),X_work[c].quantile(.75); iqr=q3-q1
            X_work[c]=X_work[c].clip(q1-1.5*iqr,q3+1.5*iqr)
    if {'DQO-E','DBO-E'}.issubset(X_work.columns): X_work['DQO_DBO_ratio_E']=X_work['DQO-E']/(X_work['DBO-E']+1)
    if {'SSV-E','SS-E'}.issubset(X_work.columns): X_work['VolatileFraction_E']=X_work['SSV-E']/(X_work['SS-E']+1)
    scaler=StandardScaler(); xs=scaler.fit_transform(X_work)
    effective_k=min(k, len(X_work)-1)
    labels_all={
        'K-Means': KMeans(n_clusters=effective_k,n_init=10,random_state=42).fit_predict(xs),
        'Agglomerative': AgglomerativeClustering(n_clusters=effective_k,linkage='ward').fit_predict(xs),
        'GMM': GaussianMixture(n_components=effective_k,random_state=42).fit_predict(xs),
        'DBSCAN': DBSCAN(eps=eps,min_samples=min_samples).fit_predict(xs)
    }
    selected={'Gaussian Mixture (GMM)':'GMM'}.get(algorithm,algorithm); labels=labels_all[selected]
    def eval_cluster(labs):
        labs=np.asarray(labs); mask=labs!=-1; unique=np.unique(labs[mask]); n=len(unique); noise=int((~mask).sum())
        out={'Clusters':n,'Silhouette':np.nan,'Davies-Bouldin':np.nan,'Calinski-Harabasz':np.nan,'Noise points':noise}
        if n>=2 and mask.sum()>n:
            out['Silhouette']=silhouette_score(xs[mask],labs[mask]); out['Davies-Bouldin']=davies_bouldin_score(xs[mask],labs[mask]); out['Calinski-Harabasz']=calinski_harabasz_score(xs[mask],labs[mask])
        return out
    st.subheader('Dataset overview')
    a,b,c,d=st.columns(4); a.metric('Rows after cleaning',f'{len(X):,}'); b.metric('Numeric features',f'{X.shape[1]:,}'); c.metric('Duplicate rows removed',f'{original_dupes:,}'); d.metric('Missing cells before imputation',f'{int(raw.isna().sum().sum()):,}')
    with st.expander('Preview data'): st.dataframe(raw.head(20),use_container_width=True)
    with st.expander('Missing values by column'):
        miss=pd.DataFrame({'Missing count':raw.isna().sum(),'Missing %':(raw.isna().mean()*100).round(2)}); st.dataframe(miss[miss['Missing count']>0],use_container_width=True)
    tabs=st.tabs(['Clustering Results','EDA & Correlations','Outliers','PCA & t-SNE','Algorithm Comparison','Cluster Profiles','Export'])
    with tabs[0]:
        score=eval_cluster(labels); cols=st.columns(4)
        for col,key in zip(cols,['Clusters','Silhouette','Davies-Bouldin','Noise points']): col.metric(key,'N/A' if pd.isna(score[key]) else (f'{score[key]:.4f}' if isinstance(score[key],(float,np.floating)) else str(score[key])))
        pca=PCA(n_components=2,random_state=42).fit_transform(xs); fig,ax=plt.subplots(figsize=(8,5)); sns.scatterplot(x=pca[:,0],y=pca[:,1],hue=labels,palette='tab10',s=35,ax=ax); ax.set_title(f'{selected} — PCA projection'); plot_show(fig)
        st.write('Cluster sizes'); st.dataframe(pd.Series(labels,name='Cluster').value_counts().sort_index().rename_axis('Cluster').reset_index(name='Rows'),use_container_width=True,hide_index=True)
        st.caption('Silhouette and Calinski–Harabasz: higher is generally better. Davies–Bouldin: lower is generally better. DBSCAN noise points are labelled -1.')
    with tabs[1]:
        chosen=st.multiselect('Features to plot',list(X.columns),default=list(X.columns[:min(6,len(X.columns))]),key='water_hist')
        for c in chosen:
            fig,ax=plt.subplots(figsize=(8,3)); sns.histplot(X[c],bins=25,ax=ax); ax.set_title(f'Distribution: {c}'); plot_show(fig)
        fig,ax=plt.subplots(figsize=(12,7)); sns.heatmap(X.corr(),cmap='coolwarm',center=0,ax=ax); ax.set_title('Feature correlation heatmap'); plot_show(fig)
        if {'PH-E','COND-E'}.issubset(X.columns):
            fig,ax=plt.subplots(figsize=(7,4)); sns.scatterplot(data=X,x='PH-E',y='COND-E',alpha=.6,ax=ax); ax.set_title('pH vs Conductivity at Entry'); plot_show(fig)
        if {'DBO-E','DBO-S'}.issubset(X.columns):
            fig,ax=plt.subplots(figsize=(7,4)); sns.scatterplot(data=X,x='DBO-E',y='DBO-S',alpha=.6,ax=ax); ax.set_title('Incoming vs Outgoing BOD'); plot_show(fig)
    with tabs[2]:
        rows=[]
        for c in X_before:
            q1,q3=X_before[c].quantile(.25),X_before[c].quantile(.75); iq=q3-q1; count=int(((X_before[c]<q1-1.5*iq)|(X_before[c]>q3+1.5*iq)).sum()); rows.append({'Feature':c,'Potential outliers':count})
        st.dataframe(pd.DataFrame(rows).sort_values('Potential outliers',ascending=False),use_container_width=True,hide_index=True)
        chosen=st.multiselect('Features for boxplots',list(X_before.columns),default=list(X_before.columns[:min(10,len(X_before.columns))]),key='water_box')
        if chosen:
            fig,ax=plt.subplots(figsize=(12,5)); X_before[chosen].boxplot(rot=70,ax=ax); ax.set_title('Boxplots before IQR capping'); plot_show(fig)
        st.caption('IQR capping limits extreme values; it does not delete observations.')
    with tabs[3]:
        pca=PCA(n_components=2,random_state=42).fit_transform(xs); fig,axes=plt.subplots(2,2,figsize=(12,9))
        for ax,(name,labs) in zip(axes.flat,labels_all.items()): sns.scatterplot(x=pca[:,0],y=pca[:,1],hue=labs,palette='tab10',s=22,legend=False,ax=ax); ax.set_title(name+' — PCA')
        plot_show(fig)
        if len(xs)>5:
            perplexity=min(30,max(2,(len(xs)-1)//3)); ts=TSNE(n_components=2,perplexity=perplexity,init='pca',learning_rate='auto',random_state=42).fit_transform(xs)
            fig,ax=plt.subplots(figsize=(8,5)); sns.scatterplot(x=ts[:,0],y=ts[:,1],hue=labels_all['K-Means'],palette='tab10',s=25,ax=ax); ax.set_title('K-Means — t-SNE'); plot_show(fig)
        pv=PCA(random_state=42).fit(xs); fig,ax=plt.subplots(figsize=(8,4)); ax.plot(np.arange(1,len(pv.explained_variance_ratio_)+1),np.cumsum(pv.explained_variance_ratio_),marker='o'); ax.axhline(.9,linestyle='--'); ax.set_xlabel('Number of components'); ax.set_ylabel('Cumulative explained variance'); plot_show(fig)
        with st.expander('Hierarchical dendrogram'):
            z=linkage(xs[:min(150,len(xs))],method='ward'); fig,ax=plt.subplots(figsize=(12,5)); dendrogram(z,truncate_mode='lastp',p=30,leaf_rotation=90,ax=ax); ax.set_title('Ward linkage dendrogram'); plot_show(fig)
    with tabs[4]:
        comp=pd.DataFrame([{'Algorithm':name,**eval_cluster(labs)} for name,labs in labels_all.items()]); st.dataframe(comp.round(4),use_container_width=True,hide_index=True)
        valid=comp.dropna(subset=['Silhouette'])
        if not valid.empty:
            fig,ax=plt.subplots(figsize=(8,4)); sns.barplot(data=valid,x='Algorithm',y='Silhouette',ax=ax); ax.set_title('Silhouette comparison (higher is better)'); plot_show(fig)
        with st.expander('K-Means elbow and silhouette curves'):
            ks=range(2,min(10,len(xs)-1)+1); inertias=[]; sil=[]
            for kk in ks:
                labs=KMeans(n_clusters=kk,n_init=10,random_state=42).fit_predict(xs); inertias.append(KMeans(n_clusters=kk,n_init=10,random_state=42).fit(xs).inertia_); sil.append(silhouette_score(xs,labs))
            fig,axes=plt.subplots(1,2,figsize=(11,4)); axes[0].plot(list(ks),inertias,marker='o'); axes[0].set_title('K-Means elbow'); axes[1].plot(list(ks),sil,marker='o'); axes[1].set_title('Silhouette vs K'); plot_show(fig)
    with tabs[5]:
        profile=X_work.assign(Cluster=labels).groupby('Cluster').mean().round(2); st.write('Mean feature values (original units)'); st.dataframe(profile.T,use_container_width=True)
        zdf=pd.DataFrame(xs,columns=X_work.columns,index=X_work.index); zp=zdf.assign(Cluster=labels).groupby('Cluster').mean()
        fig,ax=plt.subplots(figsize=(max(8,len(zp.columns)*.45),max(4,len(zp.index)*.8))); sns.heatmap(zp.T,cmap='coolwarm',center=0,ax=ax); ax.set_title('Standardised cluster profile'); plot_show(fig)
        for cid in zp.index: st.write(f'Cluster {cid}: '+', '.join(zp.loc[cid].abs().sort_values(ascending=False).head(5).index))
    with tabs[6]:
        out=X_work.copy(); out['Cluster']=labels; st.download_button('Download dataset with cluster labels (CSV)',out.to_csv(index=False).encode(),file_name=f'water_treatment_{selected.lower()}_clusters.csv',mime='text/csv')

def churn_app():
    st.title('📞 Telco Customer Churn Classification Explorer')
    st.caption('Customer churn prediction • data quality • EDA • ten classifiers • evaluation metrics • feature importance')
    with st.sidebar:
        st.header('Churn dataset'); up=st.file_uploader('Upload telco_churn.csv',type=['csv'],key='churn_file')
        st.caption('Expected target column: Churn (Yes/No).')
        st.divider(); st.header('Training settings')
        model_names=['Logistic Regression','KNN','Gaussian Naive Bayes','Decision Tree','SVM','Random Forest','Extra Trees','AdaBoost','Gradient Boosting','Bagging (Decision Tree)','MLP Neural Network']
        selected=st.multiselect('Models to train',model_names,default=model_names)
        test_pct=st.slider('Test set (%)',10,35,20,5); row_limit=st.selectbox('Rows for faster training',['Use all rows','5000','3000','1500'],index=0)
        seed=st.number_input('Random state',min_value=0,max_value=9999,value=42); train=st.button('Train / refresh models',type='primary',use_container_width=True)
    if up is None:
        st.info('Upload your Telco Customer Churn CSV in the sidebar. The notebook uses Churn as the target and removes customerID as an identifier.')
        st.markdown('**Included sections:** dataset audit, TotalCharges cleaning, ServiceCount feature engineering, target distribution, numerical/categorical EDA, correlation heatmap, IQR outlier review, model comparison, confusion matrix, classification report and CSV export.')
        return
    try: raw=pd.read_csv(up,low_memory=False); raw.columns=[str(c).strip() for c in raw.columns]
    except Exception as e: st.error(f'Could not read CSV: {e}'); return
    if 'Churn' not in raw.columns: st.error("The CSV must contain a 'Churn' target column."); return
    df=raw.copy(); dup=int(df.duplicated().sum()); df=df.drop_duplicates()
    if 'TotalCharges' in df:
        df['TotalCharges']=pd.to_numeric(df['TotalCharges'].astype(str).str.strip().replace('',np.nan),errors='coerce')
        if 'tenure' in df:
            zero_tenure = pd.to_numeric(df['tenure'], errors='coerce').fillna(-1).eq(0)
            df.loc[zero_tenure & df['TotalCharges'].isna(), 'TotalCharges'] = 0
        df['TotalCharges']=df['TotalCharges'].fillna(df['TotalCharges'].median())
    df=df.dropna(subset=['Churn']); df['Churn']=df['Churn'].astype(str).str.strip()
    if 'customerID' in df: df=df.drop(columns=['customerID'])
    service_cols=[c for c in ['PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'] if c in df]
    if service_cols: df['ServiceCount']=(df[service_cols].astype(str)=='Yes').sum(axis=1)
    X=df.drop(columns=['Churn']); y=df['Churn']
    # Remove columns that carry no information and keep target labels unchanged.
    X=X.drop(columns=[c for c in X if X[c].nunique(dropna=True)<=1],errors='ignore')
    if len(X)<10 or X.shape[1]==0: st.error('The dataset needs at least 10 rows and one predictor.'); return
    if row_limit!='Use all rows' and len(X)>int(row_limit):
        try:
            chosen_idx, _ = train_test_split(X.index, train_size=int(row_limit), random_state=int(seed), stratify=y)
            X = X.loc[chosen_idx].copy(); y = y.loc[chosen_idx].copy()
        except ValueError:
            X = X.sample(n=int(row_limit), random_state=int(seed)); y = y.loc[X.index]
    ncols=X.select_dtypes(include=np.number).columns.tolist(); ccols=[c for c in X.columns if c not in ncols]
    st.subheader('Dataset overview'); a,b,c,d=st.columns(4); a.metric('Rows loaded',f'{len(raw):,}'); b.metric('Rows used',f'{len(X):,}'); c.metric('Predictor features',f'{X.shape[1]:,}'); d.metric('Churn classes',str(y.nunique()))
    with st.expander('Preview cleaned data'): st.dataframe(df.head(20),use_container_width=True)
    with st.expander('Missing values by column'):
        miss=pd.DataFrame({'Missing count':raw.isna().sum(),'Missing %':(raw.isna().mean()*100).round(2)}); st.dataframe(miss[miss['Missing count']>0],use_container_width=True)
    tabs=st.tabs(['Overview & EDA','Outliers','Train & Evaluate','Model Comparison','Confusion Matrix','Feature Insights','Prediction & Export'])
    with tabs[0]:
        counts=y.value_counts().rename_axis('Churn').reset_index(name='Customers'); fig,ax=plt.subplots(figsize=(7,4)); sns.barplot(data=counts,x='Churn',y='Customers',ax=ax); ax.set_title('Customer Churn Distribution'); plot_show(fig)
        if ncols:
            chosen=st.multiselect('Numerical features',ncols,default=ncols[:min(4,len(ncols))],key='churn_nums')
            for c in chosen:
                fig,ax=plt.subplots(figsize=(7,3)); sns.histplot(pd.to_numeric(X[c],errors='coerce'),bins=30,ax=ax); ax.set_title(f'Distribution: {c}'); plot_show(fig)
            if len(ncols)>=2:
                corr_cols=st.multiselect('Features for correlation heatmap',ncols,default=ncols[:min(10,len(ncols))],key='churn_corr')
                if len(corr_cols)>=2:
                    corr=X[corr_cols].corr(); fig,ax=plt.subplots(figsize=(9,6)); sns.heatmap(corr,cmap='coolwarm',center=0,annot=len(corr_cols)<=8,fmt='.2f',ax=ax); ax.set_title('Numerical Feature Correlation'); plot_show(fig)
        if ccols:
            cat=st.selectbox('Categorical feature',ccols); vc=X[cat].astype(str).value_counts().head(20).rename_axis(cat).reset_index(name='Count'); fig,ax=plt.subplots(figsize=(8,4)); sns.barplot(data=vc,y=cat,x='Count',ax=ax); ax.set_title(f'Top categories: {cat}'); plot_show(fig)
    with tabs[1]:
        if not ncols: st.info('No numeric predictor columns found.')
        else:
            rows=[]
            for c in ncols:
                s=pd.to_numeric(X[c],errors='coerce').dropna(); q1,q3=s.quantile(.25),s.quantile(.75); iq=q3-q1; lo,hi=q1-1.5*iq,q3+1.5*iq; count=int(((s<lo)|(s>hi)).sum()); rows.append({'Feature':c,'Lower bound':lo,'Upper bound':hi,'Potential outliers':count,'Outlier %':round(count/max(len(s),1)*100,2)})
            st.dataframe(pd.DataFrame(rows).sort_values('Potential outliers',ascending=False),use_container_width=True,hide_index=True)
            chosen=st.multiselect('Features for boxplots',ncols,default=ncols[:min(8,len(ncols))],key='churn_box')
            if chosen:
                fig,ax=plt.subplots(figsize=(11,5)); X[chosen].boxplot(rot=60,ax=ax); ax.set_title('Numeric predictor boxplots'); plot_show(fig)
        st.caption('Outliers are flagged using the 1.5 × IQR rule. For churn data, unusual values should be reviewed rather than automatically removed.')
    if 'churn_results' not in st.session_state: st.session_state.churn_results=None
    if train:
        if not selected: st.warning('Select at least one model.'); return
        try:
            strat=y if y.value_counts().min()>=2 else None
            Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_pct/100,random_state=int(seed),stratify=strat)
            num=Xtr.select_dtypes(include=np.number).columns.tolist(); cat=[c for c in Xtr.columns if c not in num]
            scaled=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
            unscaled=ColumnTransformer([('num',SimpleImputer(strategy='median'),num),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
            constructors={'Logistic Regression':(LogisticRegression(max_iter=1500,random_state=int(seed)),True),'KNN':(KNeighborsClassifier(n_neighbors=5),True),'Gaussian Naive Bayes':(GaussianNB(),True),'Decision Tree':(DecisionTreeClassifier(random_state=int(seed),class_weight='balanced'),False),'SVM':(SVC(probability=True,random_state=int(seed),class_weight='balanced'),True),'Random Forest':(RandomForestClassifier(n_estimators=120,random_state=int(seed),n_jobs=-1,class_weight='balanced'),False),'Extra Trees':(ExtraTreesClassifier(n_estimators=120,random_state=int(seed),n_jobs=-1,class_weight='balanced'),False),'AdaBoost':(AdaBoostClassifier(n_estimators=80,random_state=int(seed)),False),'Gradient Boosting':(GradientBoostingClassifier(random_state=int(seed)),False),'Bagging (Decision Tree)':(BaggingClassifier(estimator=DecisionTreeClassifier(random_state=int(seed)),n_estimators=50,random_state=int(seed),n_jobs=-1),False),'MLP Neural Network':(MLPClassifier(hidden_layer_sizes=(64,32),max_iter=250,early_stopping=True,random_state=int(seed)),True)}
            results=[]; fitted={}; preds={}; probs={}; progress=st.progress(0)
            for i,name in enumerate(selected):
                try:
                    clf,uses_scale=constructors[name]; pipe=Pipeline([('prep',scaled if uses_scale else unscaled),('clf',clf)])
                    if name=='Gaussian Naive Bayes':
                        prep=scaled; xt=prep.fit_transform(Xtr,ytr); xv=prep.transform(Xte)
                        if hasattr(xt,'toarray'): xt=xt.toarray(); xv=xv.toarray()
                        clf.fit(xt,ytr); pred=clf.predict(xv); prob=clf.predict_proba(xv) if hasattr(clf,'predict_proba') else None; fitted[name]=('dense',prep,clf)
                    else:
                        pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); prob=pipe.predict_proba(Xte) if hasattr(pipe,'predict_proba') else None; fitted[name]=('pipeline',pipe)
                    positive='Yes' if 'Yes' in list(pd.unique(y)) else sorted(pd.unique(y))[0]
                    try: auc=roc_auc_score((yte==positive).astype(int),prob[:,list(pipe.classes_ if name!='Gaussian Naive Bayes' else clf.classes_).index(positive)]) if prob is not None and len(pd.unique(y))==2 else np.nan
                    except Exception: auc=np.nan
                    results.append({'Model':name,'Accuracy':accuracy_score(yte,pred),'Precision':precision_score(yte,pred,pos_label=positive,average='binary',zero_division=0) if len(pd.unique(y))==2 else precision_score(yte,pred,average='weighted',zero_division=0),'Recall':recall_score(yte,pred,pos_label=positive,average='binary',zero_division=0) if len(pd.unique(y))==2 else recall_score(yte,pred,average='weighted',zero_division=0),'Weighted F1-score':f1_score(yte,pred,average='weighted',zero_division=0),'ROC-AUC':auc})
                    preds[name]=pred; probs[name]=prob
                except Exception as e: st.warning(f'{name} could not be trained: {e}')
                progress.progress((i+1)/len(selected))
            progress.empty()
            if not results: st.error('No models completed.'); return
            st.session_state.churn_results={'results':pd.DataFrame(results).sort_values('Weighted F1-score',ascending=False),'models':fitted,'preds':preds,'probs':probs,'X_test':Xte,'y_test':yte,'X_train':Xtr,'features':list(X.columns),'positive':positive}
        except Exception as e: st.error(f'Training failed: {e}')
    result=st.session_state.churn_results
    if result is None:
        with tabs[2]: st.info('Choose models and settings in the sidebar, then click Train / refresh models.')
        return
    res=result['results'];
    with tabs[2]:
        a,b,c=st.columns(3); a.metric('Models completed',len(res)); b.metric('Training rows',len(result['X_train'])); c.metric('Test rows',len(result['y_test']))
        st.dataframe(res.round(4),use_container_width=True,hide_index=True)
        best=res.iloc[0]; st.success(f"Highest weighted F1-score: {best['Model']} ({best['Weighted F1-score']:.4f})")
        choice=st.selectbox('Detailed classification report',res['Model'].tolist(),key='report_model'); st.dataframe(pd.DataFrame(classification_report(result['y_test'],result['preds'][choice],output_dict=True,zero_division=0)).T.round(4),use_container_width=True)
    with tabs[3]:
        metric=st.radio('Compare by',['Accuracy','Precision','Recall','Weighted F1-score','ROC-AUC'],horizontal=True)
        valid=res.dropna(subset=[metric]).sort_values(metric,ascending=False)
        if not valid.empty:
            fig,ax=plt.subplots(figsize=(10,max(4,.45*len(valid)))); sns.barplot(data=valid,x=metric,y='Model',ax=ax); ax.set_xlim(0,1); ax.set_title(f'Model comparison — {metric}'); plot_show(fig)
        st.dataframe(res.round(4),use_container_width=True,hide_index=True)
    with tabs[4]:
        cm_model=st.selectbox('Choose model for confusion matrix',res['Model'].tolist(),key='cm_churn'); order=sorted(pd.unique(result['y_test'])); cm=confusion_matrix(result['y_test'],result['preds'][cm_model],labels=order)
        fig,ax=plt.subplots(figsize=(6,5)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=order,yticklabels=order,ax=ax); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(f'{cm_model} — confusion matrix'); plot_show(fig)
        st.caption('Rows are actual classes and columns are predicted classes. Diagonal cells are correct predictions.')
    with tabs[5]:
        tree_names=[n for n in res['Model'] if n in ['Decision Tree','Random Forest','Extra Trees']]
        if tree_names:
            fi=st.selectbox('Tree model',tree_names,key='churn_fi'); item=result['models'][fi]; pipe=item[1] if item[0]=='pipeline' else None
            if pipe is not None and hasattr(pipe.named_steps['clf'],'feature_importances_'):
                try:
                    names=pipe.named_steps['prep'].get_feature_names_out(); imp=pd.DataFrame({'Feature':names,'Importance':pipe.named_steps['clf'].feature_importances_}).sort_values('Importance',ascending=False).head(20)
                    fig,ax=plt.subplots(figsize=(9,6)); sns.barplot(data=imp,x='Importance',y='Feature',ax=ax); ax.set_title(f'Top feature importances — {fi}'); plot_show(fig); st.dataframe(imp,use_container_width=True,hide_index=True)
                except Exception as e: st.info(f'Could not extract feature names: {e}')
        else: st.info('Train a tree-based model to inspect feature importance.')
    with tabs[6]:
        choice=st.selectbox('Model for prediction/export',res['Model'].tolist(),key='churn_predict')
        st.write('Download held-out test predictions and the model comparison table below.')
        out=result['X_test'].copy(); out['Actual_Churn']=result['y_test'].values; out['Predicted_Churn']=result['preds'][choice]
        st.download_button('Download test predictions (CSV)',out.to_csv(index=False).encode(),file_name='telco_churn_predictions.csv',mime='text/csv')
        st.download_button('Download model comparison (CSV)',res.to_csv(index=False).encode(),file_name='telco_churn_model_comparison.csv',mime='text/csv')
        st.subheader('Predict one customer')
        with st.form('single_customer'):
            values={}; cols=st.columns(3)
            for i,col in enumerate(result['features']):
                if col in ncols:
                    ser=pd.to_numeric(X[col],errors='coerce').dropna(); default=float(ser.median()) if not ser.empty else 0.0; lo=float(ser.min()) if not ser.empty else 0.0; hi=float(ser.max()) if not ser.empty else 1.0
                    if hi<=lo: hi=lo+1
                    values[col]=cols[i%3].number_input(col,min_value=lo,max_value=hi,value=float(np.clip(default,lo,hi)))
                else:
                    opts=X[col].dropna().astype(str).value_counts().index.tolist(); opts=opts[:100] or ['Unknown']; values[col]=cols[i%3].selectbox(col,opts)
            submitted=st.form_submit_button('Predict churn')
        if submitted:
            sample=pd.DataFrame([values],columns=result['features']); item=result['models'][choice]
            try:
                if item[0]=='dense':
                    _,prep,clf=item; transformed=prep.transform(sample)
                    if hasattr(transformed,'toarray'): transformed=transformed.toarray()
                    pred=clf.predict(transformed)[0]
                else: pred=item[1].predict(sample)[0]
                st.success(f'Predicted churn: **{pred}**')
                st.caption('This is a model prediction, not a guarantee of customer behaviour.')
            except Exception as e: st.error(f'Prediction failed: {e}')

def main():
    st.sidebar.title('🧠 ML Project Explorer')
    st.sidebar.caption('Choose one of the two notebook projects')
    choice=st.sidebar.radio('Select project',['Telco Customer Churn Classification','Water Treatment Plant Clustering'])
    if choice=='Telco Customer Churn Classification': churn_app()
    else: water_app()
    st.divider(); st.caption('Local Streamlit application • Upload your own CSV/data file • Results are generated from the uploaded dataset.')

if __name__=='__main__': main()
