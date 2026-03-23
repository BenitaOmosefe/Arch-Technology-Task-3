INTERNSHIP PROJECT REPORT: TASK 3

Organization: Arch Technologies

Project: Data Science Internship (Month 2)

Task Name: Customer Segmentation Analysis

Date: March 22, 2026

1.Introduction

The goal of this project was to analyze customer purchasing behavior to identify distinct groups within a dataset. By applying unsupervised machine learning techniques, specifically KMeans Clustering, the project aims to provide actionable insights for targeted marketing and customer relationship management.

2.Dataset Analysis

The analysis was performed on a dataset containing 1,000 customer records. The primary features used for segmentation included:

•	Income_Level: The annual earnings of the customer.

•	Total_Spending: The cumulative amount spent by the customer.

•	Total_Purchases: The frequency of transactions.

•	Avg_Order_Value: The average amount spent per transaction.

3.Methodology

3.1.Data Preprocessing

To ensure the model's accuracy, the data underwent the following steps:

•	Library Integration: Utilized pandas for data manipulation and scikit-learn for machine learning.

•	Feature Scaling: Applied StandardScaler to normalize the data. This step is critical because KMeans relies on Euclidean distance; without scaling, features with larger numerical ranges (like Income) would dominate the model.

3.2.The Elbow Method

To determine the optimal number of clusters ($k$), I implemented the Elbow Method. By calculating the Within-Cluster Sum of Squares (WCSS) for $k$ values 1 through 10, I identified the "elbow" point where adding more clusters no longer significantly improved the model.

4.Implementation & Results

The model was executed with $k=4$ clusters. The resulting segments are characterized as follows:

Cluster ID	Segment Name	Characteristics	Strategic Recommendation

Cluster 0	Budget Seekers	Low Income, Low Spending	Focus on essential goods and discounts.

Cluster 1	VIP/High Value	High Income, High Spending	Target with luxury offers and loyalty rewards.

Cluster 2	Potential Growth	High Income, Low Spending	Use personalized ads to increase engagement.

Cluster 3	Loyal Frequent	Moderate Income, High Purchases	Reward frequency with "buy-X-get-Y" deals.

5.Visualization

The final segmentation was visualized using a scatter plot comparing Income_Level and Total_Spending. This visualization confirms that the KMeans algorithm successfully separated the customers into non-overlapping, distinct groups.

6.Conclusion

The implementation of KMeans clustering on the provided dataset successfully identified four primary customer archetypes. These insights allow for more efficient resource allocation and personalized customer experiences, fulfilling the requirements for Task 3 of the Arch Technologies internship program.

Submitted by: Nikoro Omosefe Benita

Technical Stack: Python, Pandas, Scikit-learn, Matplotlib, Seaborn.
