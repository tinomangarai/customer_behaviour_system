import os
import joblib
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

def generate_customer_report(data_path, model_path='models/', refit_models=False):
    # Load and preprocess data
    data = pd.read_csv(data_path)
    os.makedirs(model_path, exist_ok=True)
    
    # Convert and filter timestamps
    data['TimeStamp'] = pd.to_datetime(data['TimeStamp'], errors='coerce')
    data = data[data['TimeStamp'] <= pd.Timestamp.now()]
    
    # Identify customers
    customers = data[data['FromPhone'].str.contains('263', na=False)]['FromPhone'].unique()
    data = data[data['FromPhone'].isin(customers)]
    
    # Feature engineering
    cutoff_date = data['TimeStamp'].max() - pd.DateOffset(months=3)
    
    # Interaction frequency
    interaction_freq = data[data['TimeStamp'] > cutoff_date].groupby('FromPhone').size()
    interaction_freq = interaction_freq.reset_index(name='InteractionCount')
    
    # Last activity
    last_activity = data.groupby('FromPhone')['TimeStamp'].max().reset_index(name='LastContact')
    last_activity['DaysSinceLast'] = (pd.Timestamp.now() - last_activity['LastContact']).dt.days
    
    # Sentiment analysis
    text_data = data[(data['ContentType'] == 'text') & (data['MsgContent'].notna())].copy()
    text_data['Sentiment'] = text_data['MsgContent'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )
    text_data['SentimentType'] = text_data['Sentiment'].apply(
        lambda x: 'Positive' if x > 0.2 else 'Negative' if x < -0.2 else 'Neutral'
    )
    
    sentiment = text_data.groupby('FromPhone').agg({
        'Sentiment': 'mean',
        'SentimentType': lambda x: x.mode()[0] if len(x) > 0 else 'Neutral',
        'MsgContent': 'count'
    }).rename(columns={
        'Sentiment': 'AvgSentiment',
        'MsgContent': 'TextInteractions',
        'SentimentType': 'DominantSentiment'
    })
    
    # UX components calculation (updated to your method)
    ux_data = data.groupby('FromPhone').agg({
        'TimeStamp': ['min', 'max', 'count'],
        'ContentType': pd.Series.nunique,
        'MsgContent': lambda x: x.str.len().mean()
    })
    
    # Flatten the MultiIndex columns
    ux_data.columns = ['_'.join(col) for col in ux_data.columns]
    
    # Calculate interaction span and the UX score
    ux_data['interaction_span'] = (ux_data['TimeStamp_max'] - ux_data['TimeStamp_min']).dt.days
    ux_data['UX_score'] = (ux_data['MsgContent_<lambda>'] + ux_data['ContentType_nunique']) / ux_data['TimeStamp_count']
    
    # Merge features
    features = interaction_freq.merge(last_activity, on='FromPhone', how='outer')
    features = features.merge(sentiment, on='FromPhone', how='left')
    features = features.merge(ux_data[['UX_score']], on='FromPhone', how='left')
    
    # Handle missing data
    features = features.fillna({
        'InteractionCount': 0,
        'DaysSinceLast': 365,
        'AvgSentiment': 0,
        'TextInteractions': 0,
        'UX_score': 0,
        'DominantSentiment': 'Neutral'
    })
    
    # NPS Calculation
    features['NPS_Group'] = np.select(
        [features['AvgSentiment'] > 0.5, features['AvgSentiment'] < -0.3],
        ['Promoter', 'Detractor'],
        default='Passive'
    )
    features['NPS_Score'] = features['NPS_Group'].map({'Promoter':1, 'Passive':0, 'Detractor':-1})
    
    # Advanced Clustering for Personas
    features['DominantSentiment_Score'] = features['DominantSentiment'].map({'Positive':1, 'Neutral':0, 'Negative':-1})
    
    cluster_features = features[['DaysSinceLast', 'DominantSentiment_Score', 'InteractionCount']].copy()
    
    # Apply power transformation to handle skewness
    pt = PowerTransformer()
    cluster_features['InteractionCount'] = pt.fit_transform(cluster_features[['InteractionCount']])
    
    # Scale features
    scaler_cluster = StandardScaler()
    scaled_features = scaler_cluster.fit_transform(cluster_features)
    
    # Optimized clustering
    best_score = -1
    best_labels = None
    
    for name, algorithm in [('KMeans', KMeans(n_clusters=5, random_state=42, n_init=20)),
                           ('Spectral', SpectralClustering(n_clusters=5, random_state=42))]:
        
        labels = algorithm.fit_predict(scaled_features)
        silhouette = silhouette_score(scaled_features, labels)
        
        if silhouette > best_score:
            best_score = silhouette
            best_labels = labels
    
    features['Cluster'] = best_labels
    
    # Dynamic Persona Naming
    def assign_persona(row):
        if row['DaysSinceLast'] > features['DaysSinceLast'].quantile(0.75):
            if row['DominantSentiment_Score'] > 0:
                return 'At-Risk Positive'
            return 'Lapsed Customers'
        elif row['InteractionCount'] > features['InteractionCount'].quantile(0.75):
            if row['DominantSentiment_Score'] > 0:
                return 'Active Promoters'
            return 'High-Usage Concern'
        else:
            if row['DominantSentiment_Score'] > 0:
                return 'Positive But Inactive'
            return 'Neutral Observers'
    
    features['Persona'] = features.apply(assign_persona, axis=1)
    
    # Model training
    if refit_models:
        features['Churned'] = (features['DaysSinceLast'] > 90).astype(int)
        
        X = features[['InteractionCount', 'DaysSinceLast', 'UX_score', 'NPS_Score', 'DominantSentiment_Score']]
        y = features['Churned']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Scale features
        scaler_model = StandardScaler()
        X_train_scaled = scaler_model.fit_transform(X_train)
        X_test_scaled = scaler_model.transform(X_test)
        
        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
        
        # Hyperparameter tuning
        param_dist = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(class_weight='balanced', random_state=42)
        random_search = RandomizedSearchCV(
            rf, param_distributions=param_dist,
            n_iter=50, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1
        )
        random_search.fit(X_resampled, y_resampled)
        
        # Save best model and scalers
        best_model = random_search.best_estimator_
        joblib.dump(best_model, f'{model_path}churn_model.pkl')
        joblib.dump(scaler_model, f'{model_path}model_scaler.pkl')
        joblib.dump(scaler_cluster, f'{model_path}cluster_scaler.pkl')
        joblib.dump(pt, f'{model_path}power_transformer.pkl')
        
        # Evaluate
        y_pred = best_model.predict(X_test_scaled)
        y_proba = best_model.predict_proba(X_test_scaled)[:,1]
        
        print("\nModel Evaluation:")
        print(f"Best Params: {random_search.best_params_}")
        print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
    else:
        best_model = joblib.load(f'{model_path}churn_model.pkl')
        scaler_model = joblib.load(f'{model_path}model_scaler.pkl')
        scaler_cluster = joblib.load(f'{model_path}cluster_scaler.pkl')
        pt = joblib.load(f'{model_path}power_transformer.pkl')
    
    # Generate predictions
    X_full = features[['InteractionCount', 'DaysSinceLast', 'UX_score', 'NPS_Score', 'DominantSentiment_Score']]
    features['ChurnProbability'] = best_model.predict_proba(scaler_model.transform(X_full))[:,1]
    features['ChurnRisk'] = np.where(features['ChurnProbability'] > 0.5, 'High', 'Low')
    
    # Final report
    report = features[[
        'FromPhone', 'InteractionCount', 'DaysSinceLast',
        'AvgSentiment', 'DominantSentiment', 'NPS_Group',
        'UX_score', 'Persona', 'ChurnProbability', 'ChurnRisk'
    ]].rename(columns={
        'FromPhone': 'CustomerID',
        'InteractionCount': '3MonthInteractions',
        'DaysSinceLast': 'DaysSinceLastContact',
        'DominantSentiment': 'PrimarySentiment'
    })
    
    # Cluster analysis
    cluster_profile = features.groupby('Persona').agg({
        'NPS_Score': 'mean',
        'AvgSentiment': 'mean',
        'InteractionCount': ['mean', 'count'],
        'ChurnProbability': 'mean',
        'UX_score': 'mean'
    })
    
    print("\nCluster Profiles:")
    print(cluster_profile)
    
    # Visualize clusters
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaler_cluster.transform(cluster_features))
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=pca_features[:,0], y=pca_features[:,1], 
                    hue=features['Persona'], palette='viridis', s=100)
    plt.title('Customer Personas Projection')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('persona_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return report.sort_values('ChurnProbability', ascending=False)

if __name__ == "__main__":
    report = generate_customer_report("dandemutande_dump.csv", refit_models=True)
    report.to_csv("complete_customer_report_final.csv", index=False)
    print("\nReport generated successfully!")
    print(report.head())
