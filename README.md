# Classifying-Shooting-Incident-Fatality-Classification
# Overview
Public safety resources are often limited, and "predictive policing" requires not just data, but ethical and actionable intelligence. In this project, I analyzed over 27,000 records from the NYPD Shooting Incident Dataset to build a classification model that predicts whether a shooting incident is likely to be fatal.

The goal wasn't just to predict an outcome, but to identify the risk factors (time, location, demographics) that drive fatalities, enabling law enforcement and community leaders to allocate resources proactively rather than reactively.

# Technologies Used
- Core Stack: Python, Pandas, NumPy
- Machine Learning: Scikit-Learn (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, Naive Bayes)
- Data Handling: SMOTE (for class imbalance), TargetEncoder, StandardScaler
- Visualization: Seaborn, Matplotlib (for heatmaps and geo-clustering)
- Business Tools: PowerPoint (for stakeholder presentation)

# Features
- Imbalanced Data Handling: Successfully treated a skewed dataset (fewer fatal incidents than non-fatal) using SMOTE (Synthetic Minority Over-sampling Technique) to prevent model bias.
- Geo-Spatial Clustering: Used Region_ID clustering to identify urban "hotspots" where fatalities are statistically more likely, moving beyond simple borough-level analysis.
- Bias & Demographics Analysis: Rigorously tested demographic features (victim/perpetrator age, race, gender) using Chi-Square tests to understand statistical significance versus potential bias.
- Model vs. Recall Trade-off: Prioritized Recall over simple Accuracy. In public safety, a "False Negative" (failing to predict a fatal risk) is much costlier than a "False Positive."

# The Process
1. Data Cleaning & Ethics:
- Handled missing values carefully to avoid discarding valuable incident reports.
- Encoded high-cardinality categorical data (like specific precincts) using Target Encoding to capture their relationship with the target variable without exploding dimensionality.

2. Exploratory Data Analysis (EDA):

- Discovery: "Hour of Incident" was a critical predictor—incidents in the late night/early morning were significantly more likely to be fatal than daytime events.
- Discovery: Seasonality plays a role, with spikes in fatalities during summer months.

3. Modeling Strategy:

- Tested 7+ algorithms.
- Logistic Regression offered the best "quick flagging" capability (high recall).
- Decision Trees offered the best "interpretability" (explaining why a risk flag was raised).

4. Final Selection:

- The Tuned Decision Tree was chosen for its balance of F1-score and its ability to be explained to non-technical stakeholders (police captains, city officials).

# What I Learned
- The "Accuracy Paradox": A model can have 80% accuracy but be useless if it misses all the fatal cases. I learned to optimize for F1-Score and Recall to ensure we catch the minority class (fatalities).
- Feature Engineering: Creating "Time Buckets" (Morning, Afternoon, Late Night) was more predictive than raw timestamps.
- Stakeholder Communication: Presenting this data requires sensitivity. It’s not just about "predicting crime"; it’s about "resource allocation" to save lives.

# Overall Growth
- This project matured my understanding of Classification Problems in high-stakes environments.
- Technical Growth: Mastered the use of GridSearchCV for hyperparameter tuning and SMOTE for balancing datasets.
- Strategic Growth: I learned to frame data science results as Policy Recommendations (e.g., "Increase patrols between 2 AM - 5 AM in Region X") rather than just code outputs.

# How can it be improved?
- External Factors: Integrating hospital proximity data (time-to-trauma-center) could be a massive predictor of fatality that is currently missing.
- Real-Time Dashboard: Building a Streamlit app for dispatchers to see "Risk Probability" in real-time based on incoming 911 call details.
- Bias Mitigation: Further auditing the model to ensure it doesn't over-police specific neighborhoods based on historical bias in the training data.
