#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso, lasso_path, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,confusion_matrix, classification_report
get_ipython().system('pip install -q xgboost')
import xgboost as xgb
from sklearn.model_selection import cross_val_predict
get_ipython().system('pip install stargazer')




# For a CSV file
df = pd.read_csv('AnonUnrelatedDonors.csv')

print(df)
living_donors = df
mean_transplants = living_donors['Transplants'].mean()
std_transplants = living_donors['Transplants'].std()



#df = df[df['State'] != 'District of Columbia']
print(mean_transplants, std_transplants)

print(df)

#think about weighting of DC vs other locations

popn = pd.read_csv('Annual US State Populations 3.csv')






pop_long = popn.melt(id_vars='observation_date', var_name='State', value_name='Population')

# Rename observation_date to Year
pop_long.rename(columns={'observation_date': 'Year'}, inplace=True)




txrate = pd.merge(df, pop_long, on=['State', 'Year'], how='left')

# Calculate Transplant Rate (per 100,000 people)
txrate['Transplant Rate'] = (txrate['Transplants'] / txrate['Population']) * 100

txrate = pd.merge(df, pop_long, on=['State', 'Year'], how='left')

df['Year'] = df['Year'].astype(int)
pop_long['Year'] = pop_long['Year'].astype(int)

txrate['Transplant Rate'] = (txrate['Transplants'] / txrate['Population']) * 100

print(txrate[['State', 'Year', 'Transplants', 'Population', 'Transplant Rate']].head())





income = pd.read_csv("State Income Per Year.csv")

income_long = income.melt(id_vars='Unnamed: 0', var_name='State', value_name='Income')

# Rename to match other data
income_long.rename(columns={'Unnamed: 0': 'Year'}, inplace=True)
income_long['Year'] = income_long['Year'].astype(int)
income_long['State'] = income_long['State'].str.strip()

law = pd.read_csv("Organ Legislation by State.csv")




law_long = law.melt(id_vars='observation_date', var_name='State', value_name='Legislation')

law_long.rename(columns={'observation_date': 'Year'}, inplace=True)
law_long['Year'] = law_long['Year'].astype(int)
law_long['State'] = law_long['State'].str.strip()




poverty = pd.read_csv("poverty_rate_by_state_corrected.csv")

unemp = pd.read_csv("Annual Unemployment Rate by State.csv")




# Merge law data
merged = pd.merge(txrate, law_long, on=['State', 'Year'], how='left')

# Merge income data
merged = pd.merge(merged, income_long, on=['State', 'Year'], how='left')



unemp_long = unemp.melt(id_vars='observation_date', var_name='State', value_name='Unemployment Rate')
unemp_long.rename(columns={'observation_date': 'Year'}, inplace=True)





# Merge poverty
merged = pd.merge(merged, poverty, on=['State', 'Year'], how='left')

# Merge unemployment
merged = pd.merge(merged, unemp_long, on=['State', 'Year'], how='left')




missing_states = set(txrate['State'].unique()) - set(income_long['State'].unique())
print("States missing from income file:", missing_states)

model_cols = ['Transplant Rate', 'Legislation', 'Income', 'Unemployment Rate']

# Drop rows where any of these are missing
cleaned = merged
cleaned = cleaned.dropna(subset=model_cols)

print("Before dropping NaNs:", merged.shape)
print("After dropping NaNs:", cleaned.shape)

merged.to_csv("final_transplant_dataset_full_cleaned.csv", index=False)

print("FILE")

print(merged)







cleaned = merged
cleaned = cleaned.dropna(subset=['Transplant Rate', 'Legislation', 'Income', 'Unemployment Rate'])


# Add dummies for fixed effects
cleaned_dummies = pd.get_dummies(cleaned, columns=['State', 'Year'], drop_first=True)


X = cleaned_dummies.drop(['Transplant Rate', 'Transplants', 'Population', 'Unemployment Rate'], axis=1, errors='ignore')
y = cleaned_dummies['Transplant Rate']



# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge with fixed effects
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
ridge.fit(X_train, y_train)
ridge_preds = ridge.predict(X_test)

# LASSO with fixed effects
lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.1, max_iter=10000))
lasso.fit(X_train, y_train)
lasso_preds = lasso.predict(X_test)

# Evaluation
print("Ridge R²:", r2_score(y_test, ridge_preds))
print("Ridge MAE:", mean_absolute_error(y_test, ridge_preds))
print("LASSO R²:", r2_score(y_test, lasso_preds))
print("LASSO MAE:", mean_absolute_error(y_test, lasso_preds))


plt.figure(figsize=(10, 6))
plt.scatter(y_test, ridge_preds, alpha=0.7, label='Ridge', color='blue')
plt.scatter(y_test, lasso_preds, alpha=0.7, label='LASSO', color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

plt.xlabel('Actual Transplant Rate')
plt.ylabel('Predicted Transplant Rate')
plt.title('Actual vs Predicted Transplant Rates (w/ Fixed Effects)')
plt.legend()
plt.show()



# Select features and response
df_fe = cleaned.dropna(subset=['Transplant Rate', 'Legislation', 'Income', 'State', 'Year','Unemployment Rate'])



df_fe['State'] = df_fe['State'].astype('category')
df_fe['Year'] = df_fe['Year'].astype('category')

# Fixed effects as dummies
X = pd.get_dummies(df_fe[['Legislation', 'Income', 'State', 'Year', 'Unemployment Rate']], drop_first=True)
y = df_fe['Transplant Rate']



year_dummies = X.filter(like='Year_')



#OLS MODEL

X_ols = X.copy()
y_ols = y.copy()

#Drop rows with any missing values in X or y
valid_idx = X_ols.dropna().index.intersection(y_ols.dropna().index)

X_ols = X_ols.loc[valid_idx]
y_ols = y_ols.loc[valid_idx]
X_ols = X_ols.apply(pd.to_numeric, errors='coerce').astype(float)

# Add constant and fit model
X_ols = sm.add_constant(X_ols)


ols_model = sm.OLS(y_ols, X_ols)

fitted_model = ols_model.fit(cov_type='cluster', cov_kwds={'groups': df_fe['State']})

print(fitted_model.summary())




#pca = PCA(n_components=1)
#year_pc = pca.fit_transform(year_dummies)

# Add back to X
#X['Year_PC1'] = year_pc
#X = X.drop(columns=year_dummies.columns)

# Standardize X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Get LASSO path
alphas, coefs, _ = lasso_path(X_scaled, y, alphas=None)

# Plot
plt.figure(figsize=(10, 8))



for i, feature in enumerate(X.columns):
    plt.plot(-np.log10(alphas), coefs[i], label=feature)

plt.xlabel('-log(λ)', fontsize=14)
plt.ylabel('Standardized coefficients', fontsize=14)
plt.title('LASSO Path (Fixed Effects Included)', fontsize=16)
plt.legend(loc='best', fontsize='small', ncol=2)
plt.tight_layout()
plt.show()



# Compute max absolute coefficient for each variable over the LASSO path
max_coefs = np.max(np.abs(coefs), axis=1)

# Get indices of top 20
top_20_indices = np.argsort(max_coefs)[-20:][::-1]  # reverse to get descending

# Plot only top 20
plt.figure(figsize=(12, 8))

for i in top_20_indices:
    plt.plot(-np.log10(alphas), coefs[i], label=X.columns[i])

plt.xlabel('-log(λ)', fontsize=14)
plt.ylabel('Standardized Coefficients', fontsize=14)
plt.title('Top 20 LASSO Paths by Max Coefficient Magnitude', fontsize=16)
plt.legend(loc='best', fontsize='small')
plt.tight_layout()
plt.show()

#SUMMARY STATISTICS

print(cleaned_dummies.describe())

pd.plotting.scatter_matrix(cleaned_dummies[['Transplant Rate', 'Income', 'Unemployment Rate']], figsize=(8, 8), diagonal='hist')

plt.show()

print(cleaned_dummies[['Transplant Rate', 'Income', 'Unemployment Rate', 'Legislation']]
      .groupby('Legislation').describe())

latex_table = fitted_model.summary().as_latex()

with open("ols_results_table.tex", "w") as f:
    f.write(latex_table)

print(latex_table)

print(fitted_model.summary(xname=list(X_ols.columns)))

import statsmodels.api as sm


fe_model = sm.OLS(y_ols, X_ols).fit(cov_type='cluster', cov_kwds={'groups': df_fe['State']})


# ✅ Save LaTeX summary
latex_table = fe_model.summary().as_latex()
print(latex_table)

with open("ols_results.tex", "w") as f:
    f.write(latex_table)


# In[3]:


print(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# ========== Ridge Regression ==========
print("\n----- Ridge Regression -----")
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
ridge.fit(X_train, y_train)
ridge_preds = ridge.predict(X_test)
print("Ridge R^2:", r2_score(y_test, ridge_preds))
print("Ridge MAE:", mean_absolute_error(y_test, ridge_preds))


# ========== LASSO Regression ==========
print("\n----- LASSO Regression -----")
lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.1, max_iter=10000))
lasso.fit(X_train, y_train)
lasso_preds = lasso.predict(X_test)
print("LASSO R^2:", r2_score(y_test, lasso_preds))
print("LASSO MAE:", mean_absolute_error(y_test, lasso_preds))


# ========== Plot Predictions ==========
plt.figure(figsize=(10, 6))
plt.scatter(y_test, ridge_preds, alpha=0.6, label='Ridge', color='blue')
plt.scatter(y_test, lasso_preds, alpha=0.6, label='LASSO', color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Transplant Rate')
plt.ylabel('Predicted Transplant Rate')
plt.title('Actual vs Predicted Transplant Rate')
plt.legend()
plt.tight_layout()
plt.show()


# ========== LASSO Path Plot ==========
print("\n----- LASSO Path Plot -----")
X_std = StandardScaler().fit_transform(X)
alphas, coefs, _ = lasso_path(X_std, y, alphas=None)
max_coefs = np.max(np.abs(coefs), axis=1)
top_20_indices = np.argsort(max_coefs)[-20:]

plt.figure(figsize=(10, 6))
for i in top_20_indices:
    plt.plot(-np.log10(alphas), coefs[i], label=X.columns[i])
    plt.xlabel('-log10(lambda)')
    plt.ylabel('Standardized Coefficients')
    plt.title('LASSO Path (Top 20 Features)')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.tight_layout()
plt.show()


# ========== Cross-Validation Curves ==========
print("\n----- Cross-Validation Plots -----")


from sklearn.model_selection import cross_val_score
import numpy as np


lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000).fit(X_scaled, y)


ridge_errors = []
alphas = np.logspace(-3, 3, 100)

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    # Use negative MSE so we negate later to get positive MSE
    scores = cross_val_score(ridge, X_scaled, y, scoring='neg_mean_squared_error', cv=5)
    ridge_errors.append(-scores.mean())

plt.figure(figsize=(10, 6))

# LASSO plot
plt.plot(lasso_cv.alphas_, np.mean(lasso_cv.mse_path_, axis=1), label='LASSO CV Error', color='orange')
plt.axvline(lasso_cv.alpha_, linestyle='--', color='orange', label=f'LASSO Best Alpha: {lasso_cv.alpha_:.4f}')


plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Mean Squared Error (CV)')
plt.title('Cross-Validation Error vs Alpha')
plt.legend()
plt.tight_layout()
plt.style.use('default')
plt.show()

# Ridge plot
plt.plot(alphas, ridge_errors, label='Ridge CV Error', color='blue')
best_ridge_alpha = alphas[np.argmin(ridge_errors)]
plt.axvline(best_ridge_alpha, linestyle='--', color='blue', label=f'Ridge Best Alpha: {best_ridge_alpha:.4f}')

plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Mean Squared Error (CV)')
plt.title('Cross-Validation Error vs Alpha')
plt.legend()
plt.tight_layout()
plt.style.use('default')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Reuse X and y from your fixed effects model
alphas = np.logspace(-3, 6, 200)

coefs = []
for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_scaled, y)
    coefs.append(ridge.coef_)

plt.figure(figsize=(10, 6))
for i in top_20_indices:
    plt.plot(np.log10(alphas), [coef[i] for coef in coefs], label=X.columns[i])

plt.xlabel('log10(lambda)', fontsize=12)

plt.ylabel('Standardized Coefficients', fontsize=12)

plt.title('Ridge Coefficient Path (Top 20 Features)', fontsize=14)
plt.legend(fontsize='x-small', ncol=2, loc='best')
plt.tight_layout()
plt.style.use('default')
plt.show()

K = 5  # number of CV folds
alphas = np.logspace(-3, 3, 50)

# ==== Ridge ====
ridge_pipeline = make_pipeline(StandardScaler(), Ridge())
ridge_grid = GridSearchCV(ridge_pipeline,
                          param_grid={'ridge__alpha': alphas},
                          scoring='neg_mean_squared_error',
                          cv=K,
                          return_train_score=True)
ridge_grid.fit(X_scaled, y)

# ==== LASSO ====
lasso_pipeline = make_pipeline(StandardScaler(), Lasso(max_iter=10000))
lasso_grid = GridSearchCV(lasso_pipeline,
                          param_grid={'lasso__alpha': alphas},
                          scoring='neg_mean_squared_error',
                          cv=K,
                          return_train_score=True)
lasso_grid.fit(X, y)

# ==== PLOT RIDGE ====
ridge_fig, ax = plt.subplots(figsize=(8, 6))

ridge_means = -ridge_grid.cv_results_['mean_test_score']
ridge_stds = ridge_grid.cv_results_['std_test_score']
ridge_se = ridge_stds / np.sqrt(K)
plt.style.use('default')
ax.errorbar(-np.log(alphas), ridge_means, yerr=ridge_se, label='Ridge CV Error', color='blue', capsize=3)
ax.set_xlabel('$-\\log(\\lambda)$', fontsize=14)
ax.set_ylabel('Cross-validated MSE', fontsize=14)
ax.set_title('Ridge CV Error with Standard Error Bars', fontsize=16)
ax.set_ylim([ridge_means.min() * 0.9, ridge_means.max() * 1.1])
ax.legend()
plt.tight_layout()
plt.show()

# ==== PLOT LASSO ====
lasso_fig, ax = plt.subplots(figsize=(8, 6))

lasso_means = -lasso_grid.cv_results_['mean_test_score']
lasso_stds = lasso_grid.cv_results_['std_test_score']
lasso_se = lasso_stds / np.sqrt(K)
plt.style.use('default')
ax.errorbar(-np.log(alphas), lasso_means, yerr=lasso_se, label='LASSO CV Error', color='orange', capsize=3)
ax.set_xlabel('$-\\log(\\lambda)$', fontsize=14)
ax.set_ylabel('Cross-validated MSE', fontsize=14)
ax.set_title('LASSO CV Error with Standard Error Bars', fontsize=16)
ax.set_ylim([lasso_means.min() * 0.9, lasso_means.max() * 1.1])
ax.legend()
plt.tight_layout()
plt.show()


from sklearn import tree
sqft_tree = tree.DecisionTreeRegressor(max_depth=5).fit(X,y)

# use the fitted tree to predict
y_pred_tree = sqft_tree.predict(X)

# find the error of prediction (MSE)
from sklearn import metrics
print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred_tree))

plt.figure(figsize=(25,20))
tree.plot_tree(sqft_tree, feature_names=X.columns, filled=True)


plt.show()


# In[4]:


import pandas as pd
import statsmodels.api as sm

# START FROM CLEANED SUBSET
df = merged.copy()

# Use only living donors, and drop rows missing any key info
df = df[df['Donor Type'] == 'Living Donor']
df = df.dropna(subset=['Transplant Rate', 'Legislation', 'Income'])

# Convert State and Year to category (for fixed effects)
df['State'] = df['State'].astype('category')
df['Year'] = df['Year'].astype('category')

# Set up regression formula with fixed effects
formula = 'Q("Transplant Rate") ~ Legislation + Income + C(State) + C(Year)'

# Fit model using patsy + statsmodels
model = sm.OLS.from_formula(formula, data=df).fit()

# Print regression output
print(model.summary())

from statsmodels.iolib.summary2 import summary_col

# One model case
reg_summary = summary_col([model], stars=True, model_names=['OLS'])
print(reg_summary)


with open("regression_table.tex", "w") as f:
    f.write(reg_summary.as_latex())


# In[ ]:


from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,confusion_matrix, classification_report


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


regr_bagg = RandomForestRegressor(max_features= 18, random_state=1) 
regr_bagg.fit(X_train, y_train)


pred = regr_bagg.predict(X)

plt.scatter(pred, y, label='log price')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y')
plt.show()

# define and fit
regr_RF = RandomForestRegressor(max_features=5, random_state=1).fit(X, y)

#predict
pred = regr_RF.predict(X_test)

#calculate MSE
mean_squared_error(y_test, pred)
print(mean_squared_error(y_test, pred))

Importance = pd.DataFrame({'Importance':regr_RF.feature_importances_*100}, index=X.columns)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r', )
plt.xlabel('Variable Importance')
plt.gca().legend_ = None



# In[ ]:


Importance = pd.DataFrame({'Importance':regr_RF.feature_importances_*100}, index=X.columns)
top_n = 25
top_importance = Importance.sort_values('Importance', ascending=False).head(top_n).sort_values('Importance')

# Plot top N
top_importance.plot(kind='barh', color='r')
plt.xlabel('Variable Importance')
plt.title(f'Top {top_n} Features')
plt.gca().legend_ = None
plt.tight_layout()
plt.show()


# In[ ]:


from sklearn import tree
sqft_tree = tree.DecisionTreeRegressor(max_depth=3).fit(X_train,y_train)

# use the fitted tree to predict
y_pred_tree = sqft_tree.predict(X_test)

# find the error of prediction (MSE)
from sklearn import metrics
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_tree))

plt.figure(figsize=(25,20))
tree.plot_tree(sqft_tree, feature_names=X_test.columns, filled=True)


plt.show()


# In[ ]:


#BAGGING


regr_bagg = RandomForestRegressor(max_features= 18, random_state=1) 
regr_bagg.fit(X_train, y_train)


pred = regr_bagg.predict(X_test)

plt.scatter(pred, y_test, label='log price')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y')


mean_squared_error(y_test, pred)

print(mean_squared_error(y_test, pred))


# In[ ]:


# Convert the data into XGBoost's DMatrix format




dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define the parameters for the XGBoost model
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'eta': 0.1,
    'max_depth': 10,
    'subsample': 0.5,
    'colsample_bytree': 0.5,
    'seed': 42
}
watchlist = [(dtrain, 'train'), (dtest, 'eval')]



# Train the XGBoost model with the optimal number of boosting rounds
model = xgb.train(params, dtrain, num_boost_round=1000,          # set high
    evals=watchlist,
    early_stopping_rounds=50,      # stop if no improvement for 50 rounds
    verbose_eval=True
)

# Make predictions 
y_pred = model.predict(dtest)

# Calculate and print the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Best RMSE:", model.best_score)
print("Best boosting round:", model.best_iteration)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c='grey', alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('True Y')
plt.ylabel('Predicted Y')
plt.title('Y vs Predicted Y (Y hat)')
plt.show()


# In[ ]:


import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import graphviz

COLORS = [
    '#00B0F0',
    '#FF0000'
]

nodes = ['A', 'B', 'C', 'D']

# Define graphs
graphs = {

    'DAG': {
        'graph': graphviz.Digraph(format='png'),
        'edges': ['AB', 'BC', 'AD', 'DC']
    },

    'DCG': {
        'graph': graphviz.Digraph(format='png'),
        'edges': ['AB', 'AD', 'BB', 'BC', 'DC', 'CA']
    },

    'Undirected': {
        'graph': graphviz.Graph(format='png'),
        'edges': ['AB', 'BC', 'AD', 'DC']
    }, 

    'Fully connected': {
        'graph': graphviz.Graph(format='png'),
        'edges': ['AB', 'AC', 'AD', 'BC', 'BD', 'CD']
    },

    'Partially connected': {
        'graph': graphviz.Graph(format='png'),
        'edges': ['AB', 'AC', 'BC']
    }
}

graphs


# Iterate over graphs and render
for name, graph in graphs.items():

    [graph['graph'].node(n) for n in nodes]
    graph['graph'].edges(graph['edges'])

    graph['graph'].render(f'img/ch_04_graph_{name}')

    graph = graphviz.Digraph(format='png')

nodes = ['0', '1', '2', '3']

edges = ['02', '13', '32', '30']

[graph.node(n) for n in nodes]
graph.edges(edges)

graph.render(f'img/ch_04_graph_adj_02')

graph


import networkx as nx

'''
# Define the graph
sample_gml = graph [directed 1, node [id 0 label "0"]

node [
    id 1
    label "1"
    ]

node [
    id 2
    label "2"
    ]


edge [
    source 0
    target 1
    ]

edge [
    source 2
    target 1
    ]]

'''

import networkx as nx
import matplotlib.pyplot as plt

nodes = ['0', '1', '2', '3']
edges = [('0','2'), ('1','3'), ('3','2'), ('3','0')]

# Create a NetworkX DiGraph
G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Draw it
plt.figure(figsize=(6,6))
nx.draw(G, with_labels=True, node_size=2500, node_color='#00B0F0', font_color='white')
plt.show()
# Get the graph
#graph = nx.parse_gml(sample_gml)

# Plot
nx.draw(
    G=graph, 
    with_labels=True,
    node_size=2500,
    node_color=COLORS[0],
    font_color='white'
)

# Define the matrix
adj_matrix = np.array([
    [0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0],
])

# Get the graph
graph = nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph)

# Plot
nx.draw(
    G=graph, 
    with_labels=True,
    node_size=2000,
    node_color=COLORS[0],
    font_color='white',
    pos=nx.planar_layout(graph)
)


# In[ ]:


     State  1988  1989  1990  1991  1992  1993  1994  1995  \
0                Alabama    8002    800    800    800    800    800    800    800     
2                Arizona    18008008    800    800    800    800    800    800    800   
3               Arkansas    800    800    800    800    800    800    800    800   


# In[7]:


# Sort by state and year to ensure chronological order
cleaned = cleaned.sort_values(['State', 'Year'])

# Create 1-year lag of legislation within each state
cleaned['Legislation_lag2'] = cleaned.groupby('State')['Legislation'].shift(2)

# Drop the first observation of each state (lag is NaN)
cleaned = cleaned.dropna(subset=['Legislation_lag2'])

cleaned_dummies = pd.get_dummies(cleaned, columns=['State', 'Year'], drop_first=True)


X = cleaned_dummies.drop(['Transplant Rate', 'Transplants', 'Poverty Rate', 'Population', 'Legislation'], axis=1, errors='ignore')
y = cleaned_dummies['Transplant Rate']

# Build design matrix
X_ols = X.copy()
y_ols = y

valid_idx = X_ols.dropna().index.intersection(y_ols.dropna().index)

X_ols = X_ols.loc[valid_idx]
y_ols = y_ols.loc[valid_idx]
X_ols = X_ols.apply(pd.to_numeric, errors='coerce').astype(float)


# Add constant for OLS intercept
X_ols = sm.add_constant(X_ols)

# Fit OLS
# Align the State column with the cleaned (lagged) dataset used for regression
groups = cleaned.loc[X_ols.index, 'State']

# Confirm everything lines up
print(len(X_ols), len(y_ols), len(groups))  # should all be identical

# Fit OLS with cluster-robust SEs
ols_model = sm.OLS(y_ols, X_ols).fit(
    cov_type='cluster',
    cov_kwds={'groups': groups}
)

print(ols_model.summary())



from stargazer.stargazer import Stargazer

# Create LaTeX regression table
stargazer = Stargazer([ols_model])
latex_table = stargazer.render_latex()

# Save to file
with open("ols_stargazer.tex", "w") as f:
    f.write(latex_table)

print(latex_table)


# In[6]:


import seaborn as sns
sns.lineplot(
    data=df,
    x='Year', y='Transplant Rate',
    hue='Legislation',
    estimator='mean'
)
plt.title('Average Transplant Rate by Legislation Status (Pre-trend Check)')
plt.ylabel('Average Transplant Rate per 100,000')
plt.show()


# In[7]:


df.groupby(['Year', 'Legislation'])['Transplant Rate'].mean().unstack().plot()
plt.title("Pre-trend Check: Transplant Rate by Legislation Status")
plt.ylabel("Average Transplant Rate")
plt.show()


# In[8]:


from scipy.stats import ttest_ind

# 1️⃣ Clean state names first
cleaned['State'] = cleaned['State'].str.strip()

# 2️⃣ Compute first policy year per state (treatment year)
policy_years = (
    cleaned[cleaned['Legislation'] == 1]
    .groupby('State')['Year']
    .min()
    .reset_index()
    .rename(columns={'Year': 'Policy_Year'})
)
policy_years['State'] = policy_years['State'].str.strip()

# 3️⃣ Drop duplicates and merge once
cleaned = cleaned.drop(columns=['Policy_Year'], errors='ignore')
cleaned = cleaned.merge(policy_years, on='State', how='left')

# 🧠 Check that merge worked
print("\nUnique policy years found:")
print(cleaned[['State', 'Policy_Year']].drop_duplicates().sort_values('Policy_Year').head(15))

# 4️⃣ Identify treated states
treated_states = policy_years['State'].unique()
print(f"\nNumber of treated states: {len(treated_states)}")

# 5️⃣ Define pre-treatment samples
# Pre-treatment years for treated states (before their policy year)
treated_pre = cleaned[
    (cleaned['State'].isin(treated_states)) &
    (cleaned['Year'] < cleaned['Policy_Year'])
]

# Control group: states that never adopted legislation (Policy_Year is NaN)
control_pre = cleaned[
    (~cleaned['State'].isin(treated_states)) &
    (cleaned['Year'] < cleaned['Year'].max())
]

print(f"\nTreated pre rows: {len(treated_pre)}, Control pre rows: {len(control_pre)}")

# 6️⃣ Run t-test (Income example)
t_stat, p_value = ttest_ind(
    treated_pre['Income'].dropna(),
    control_pre['Income'].dropna(),
    equal_var=False
)
print(f"\nIncome balance test — T-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")

# 7️⃣ (Optional) Shorter equivalent test printout
print("\nFull t-test output:")
print(ttest_ind(treated_pre['Income'].dropna(),
                control_pre['Income'].dropna(),
                equal_var=False))


# In[9]:


from sklearn.preprocessing import StandardScaler

covariates = ['Income', 'Unemployment Rate']
scaled = cleaned[covariates].apply(lambda x: (x - x.mean()) / x.std())
scaled['Legislation'] = cleaned['Legislation']

means = scaled.groupby('Legislation')[covariates].mean().T
means.plot(kind='bar', color=['#3498db', '#e74c3c'])
plt.title('Standardized Covariate Balance: Treated vs Control')
plt.ylabel('Mean (Standardized Units)')
plt.xlabel('Covariate')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# In[10]:


vars_to_describe = [
    'Transplant Rate', 'Income', 'Unemployment Rate',
    'Population', 'Poverty Rate', 'Legislation'
]

desc = cleaned[vars_to_describe].describe().T  # Transpose for readability
desc = desc[['mean', 'std', 'min', 'max']]     # Keep key summary stats
desc.rename(columns={
    'mean': 'Mean', 'std': 'Std. Dev.', 'min': 'Min', 'max': 'Max'
}, inplace=True)

# Round for clean presentation
desc_rounded = desc.round(3)

# Export to LaTeX table
latex_table = desc_rounded.to_latex(
    index=True,
    caption="Descriptive Statistics of Key Variables",
    label="tab:descstats",
    escape=False,
    column_format='lcccc'
)

with open("descriptive_stats.tex", "w") as f:
    f.write(latex_table)

print(latex_table)


# In[ ]:





# In[ ]:





# In[ ]:




