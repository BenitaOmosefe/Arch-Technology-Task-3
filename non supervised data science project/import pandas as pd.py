import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load data
data = pd.read_csv('synthetic_customer_data.csv')

# 2. Preprocessing & Feature Selection
# Task 3: Group customers into different segments based on purchasing behavior 
# We'll use 'Income_Level' and 'Total_Spending' as primary behavioral indicators.
features = ['Income_Level', 'Total_Spending', 'Avg_Order_Value', 'Total_Purchases']
data_for_clustering = data[features]

# Scaling is crucial for KMeans
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_for_clustering)

# 3. Find optimal clusters (Elbow Method)
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.savefig('elbow_method.png')
plt.close()

# 4. Perform Clustering with k=4 (typical for this type of data)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_transform(scaled_data).argmax(axis=1) # Using labels_ is cleaner
data['Cluster'] = kmeans.labels_

# 5. Visualize Results
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Income_Level', y='Total_Spending', hue='Cluster', palette='viridis')
plt.title('Customer Segments: Income vs Spending')
plt.savefig('customer_segments.png')
plt.close()

# 6. Save segmented data
data.to_csv('segmented_customer_data.csv', index=False)

# Display cluster characteristics
cluster_summary = data.groupby('Cluster')[features].mean()
print(cluster_summary)
