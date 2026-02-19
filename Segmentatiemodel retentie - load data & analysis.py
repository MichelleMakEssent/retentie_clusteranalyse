# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Inladen benodigde packages

# COMMAND ----------

# DBTITLE 1,import
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import numpy as np
import plotly.express as px
import json
from scipy.stats import norm
from itertools import combinations
from scipy.stats import chi2_contingency
from datetime import datetime


# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Data Load 

# COMMAND ----------

# DBTITLE 1,read basetable from snowflake
df = spark.sql('''

SELECT * FROM `snowflake_prd_prototype_data_analysts`.`playground`.BASETABLE_RETENTION_SEGMENTATION_V2_28022026 
--SELECT * FROM `snowflake_prd_prototype_data_analysts`.`playground`.BASETABLE_RETENTION_SEGMENTATION_V2_160126 
--SELECT * FROM `snowflake_prd_prototype_data_analysts`.`playground`.`BASETABLE_RETENTION_SEGMENTATION_V1_281025`
''').toPandas()

## dataset geupdate op 16-01-2026
## dataset geupdate op 26-01-2026 door Roel met toevoeging van EDM data

# COMMAND ----------

current_date = datetime.now().strftime('%d%m%y')
df.to_parquet(
    f'/Workspace/data_analysts/MM/retentie_clusteranalyse/data/base_df_{current_date}.parquet',
    index=False
)

# COMMAND ----------

# Check for duplicate rows
duplicate_rows_count = df.duplicated().sum()
print("Duplicate rows count:", duplicate_rows_count)

# Display duplicate rows if any
if duplicate_rows_count > 0:
    display(df[df.duplicated()])

# Drop duplicate rows
df = df.drop_duplicates()

# COMMAND ----------

# DBTITLE 1,head of df
df.head()

# COMMAND ----------

len(df)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Data Cleaning

# COMMAND ----------

# DBTITLE 1,column list
columns_list = df.columns.tolist()
print(columns_list)

# COMMAND ----------

null_age_count = df['AGE'].isnull().sum()
print("Number of rows with AGE null:", null_age_count)

# COMMAND ----------

# DBTITLE 1,preprocessing
# Impute 'AGE' with mean if it is null
df['AGE'] = df['AGE'].astype(float).fillna(df['AGE'].astype(float).mean())

# Keep only relevant columns for analysis
#columns_to_keep = ['CUSTOMERNUMBER', 'BRANDID', 'AANTAL_MAANDEN_KLANT', 'KANAAL', 'HOEVAAK_KLANT_GEWEEST', 'AGE', 
                #    'TERUGLEVERING_JA', 'LIVE_CONTACTS', 'ENERGIEPROFIEL_INGEVULD', 'ONLINE_VISITS', 'CONSENT_GEGEVEN', 'LOYALTY_KLANT', 'ORIENTATIE_SCORE', 'DAYS_RETENTIE_CONTRACT_GETEKEND']

df_segm = df.copy() #[columns_to_keep]

def preprocess_categorical(df, categorical_columns):
    """
    Optimized version: Faster preprocessing of categorical data, including Y/N conversion and One-Hot Encoding.
    Additionally, sets specified columns to 0 where they have NULLs.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - categorical_columns (list): List of categorical columns to process.
    - null_to_zero_columns (list): List of columns to set to 0 where they have NULLs.
    
    Returns:
    - pd.DataFrame: Transformed DataFrame.
    """
    df = df.copy()  # Avoid modifying the original DataFrame
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=False, drop_first=False)
    return df

# Apply function
categorical_cols = [
    'KANAAL'
    ]

processed_df = preprocess_categorical(df_segm, categorical_cols)
processed_df.head()

# COMMAND ----------

#Check of alle klantnrs in originele df ook in processed df zitten 
df["CUSTOMERNUMBER"].isin(processed_df["CUSTOMERNUMBER"]).all()


# COMMAND ----------

display(type(df), "-")

display(type(processed_df))

# COMMAND ----------

# DBTITLE 1,describe df
processed_df.describe()

# COMMAND ----------

# DBTITLE 1,to numeric
# Ensure all relevant columns are numeric (float)
for col in ['CUSTOMERNUMBER', 'AANTAL_MAANDEN_KLANT', 'KANAAL', 'HOEVAAK_KLANT_GEWEEST', 'AGE', 'TERUGLEVERING_JA', 'LIVE_CONTACTS', 'ENERGIEPROFIEL_INGEVULD', 'ONLINE_VISITS', 'CONSENT_GEGEVEN', 'LOYALTY_KLANT', 'ORIENTATIE_SCORE', 'DAYS_RETENTIE_CONTRACT_GETEKEND']:

    if col in processed_df.columns:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

# COMMAND ----------

# DBTITLE 1,preprocessed df to parquet
current_date = datetime.now().strftime('%d%m%y')
processed_df.to_parquet(
    f'/Workspace/data_analysts/MM/retentie_clusteranalyse/data/processed_df_{current_date}.parquet',
    index=False
)

# COMMAND ----------

# DBTITLE 1,dtypes
print(processed_df.dtypes)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Data Visualisation

# COMMAND ----------

#ik moet hier alsnog handmatig de datum inzetten, checken of het mogelijk is om die ook te automatiseren? 
df_processed = pd.read_parquet('/Workspace/data_analysts/MM/retentie_clusteranalyse/data/processed_df_190226.parquet',)
df_processed.head(15)

# COMMAND ----------

essent_bordeaux = '#B8255F'
essent_white = '#FFFFFF'
ed_main = '#5FB623'
blue_lighten =  '#3375BB'
grey_90 = '#BEBEBE'
light_grey = '#d3d3d3'
axis_grey = '#949494'

color_map={'Essent':essent_bordeaux,
           'ES':essent_bordeaux,
           '040':essent_bordeaux,
           'EnergieDirect.nl':ed_main,
           'ED':ed_main,
           '050':ed_main}

# COMMAND ----------

numeric_cols = [
    'AANTAL_MAANDEN_KLANT',
    'HOEVAAK_KLANT_GEWEEST',
    'LIVE_CONTACTS',
    'ONLINE_VISITS',
    'ENERGIEPROFIEL_INGEVULD',
    'TERUGLEVERING_JA',
    'CONSENT_GEGEVEN',
    'LOYALTY_KLANT',
    'DAYS_RETENTIE_CONTRACT_GETEKEND'
]

for col in numeric_cols:
    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

display(df_processed[numeric_cols].dtypes)

# COMMAND ----------

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

essent_bordeaux = '#B8255F'
essent_white = '#FFFFFF'
ed_main = '#5FB623'
blue_lighten =  '#3375BB'
grey_90 = '#BEBEBE'
light_grey = '#d3d3d3'
axis_grey = '#949494'

selected_cols = ['BRANDID', 'AANTAL_MAANDEN_KLANT', 'HOEVAAK_KLANT_GEWEEST', 'AGE', 'TERUGLEVERING_JA', 'LIVE_CONTACTS', 'ENERGIEPROFIEL_INGEVULD', 'ONLINE_VISITS', 'CONSENT_GEGEVEN', 'LOYALTY_KLANT', 'ORIENTATIE_SCORE', 'DAYS_RETENTIE_CONTRACT_GETEKEND']

# Selecteer alleen numerieke kolommen die bestaan
numeric_cols = [
    col for col in selected_cols
    if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col])
]

n = len(numeric_cols)
if n == 0:
    raise ValueError("Geen numerieke kolommen gevonden")

cols_per_row = 4
rows = math.ceil(n / cols_per_row)

plt.style.use('seaborn-whitegrid')
fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 5, rows * 4), facecolor=essent_white)
axes = axes.flatten()

for ax, col in zip(axes, numeric_cols):

    data = df_processed[col].dropna()
    unique_vals = sorted(data.unique())

    # --------- Histogram ----------
    if set(unique_vals).issubset({0, 1}) and len(unique_vals) > 0:
        counts = [(data == 0).sum(), (data == 1).sum()]
        ax.bar([0, 1], counts, color=essent_bordeaux, edgecolor=grey_90, width=0.6, align='center')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['0', '1'])
    else:
        df_processed[col].hist(
            bins=20,
            ax=ax,
            color=essent_bordeaux,
            edgecolor=grey_90,
            grid=False
        )
        if len(data) > 0:
            min_val, max_val = data.min(), data.max()
            ticks = np.linspace(min_val, max_val, num=7)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:.0f}" for t in ticks], rotation=45)

    ax.set_title(col, fontsize=10, color=essent_bordeaux)
    ax.set_xlabel(col, color=axis_grey)
    ax.set_ylabel("Frequency", color=axis_grey)
    ax.spines['top'].set_color(light_grey)
    ax.spines['right'].set_color(light_grey)
    ax.spines['left'].set_color(light_grey)
    ax.spines['bottom'].set_color(light_grey)
    ax.tick_params(axis='x', colors=axis_grey)
    ax.tick_params(axis='y', colors=axis_grey)
    ax.set_facecolor(essent_white)

# Verberg ongebruikte subplots
for ax in axes[n:]:
    ax.set_visible(False)

fig.suptitle('Histograms – alle brands samen', fontsize=14, color=essent_bordeaux, y=1.02)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Clean numerical data

# COMMAND ----------

# Selecteer numerieke kolommen die bestaan
numeric_cols = [
    col for col in selected_cols
    if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col])
]

# Bereken min en max
min_max_df = df_processed[numeric_cols].agg(['min', 'max', 'mean']).T

min_max_df

# COMMAND ----------

#hoeveel klanten hebben getekend voor retentieperiode 'officiele' start 
count_negative_days = (df_processed['DAYS_RETENTIE_CONTRACT_GETEKEND'] < 0).sum()
print("Number of rows with DAYS_RETENTIE_CONTRACT_GETEKEND < 0:", count_negative_days)

# COMMAND ----------

# Convert the column to float before quantile calculation
df_processed['AANTAL_MAANDEN_KLANT'] = df_processed['AANTAL_MAANDEN_KLANT'].astype(float)

low_percentile = df_processed['AANTAL_MAANDEN_KLANT'].quantile(0.25)
mid_percentile = df_processed['AANTAL_MAANDEN_KLANT'].quantile(0.50)
high_percentile = df_processed['AANTAL_MAANDEN_KLANT'].quantile(0.75)

def categorize_duur(row):
    if pd.isnull(row['AANTAL_MAANDEN_KLANT']):
        return 'Geen klant'
    elif row['AANTAL_MAANDEN_KLANT'] <= low_percentile:
        return 'Net klant'
    elif low_percentile < row['AANTAL_MAANDEN_KLANT'] <= mid_percentile:
        return 'Medium lang klant'
    else:
        return 'Heel lang klant'

df_processed['Klantduur'] = df_processed.apply(
    categorize_duur,
    axis=1
)

display(df_processed['Klantduur'].value_counts())
display(df_processed.groupby('Klantduur').mean())

# COMMAND ----------

# Convert the column to float before quantile calculation
df_processed['DAYS_RETENTIE_CONTRACT_GETEKEND'] = df_processed['DAYS_RETENTIE_CONTRACT_GETEKEND'].astype(float)

low_percentile = df_processed['DAYS_RETENTIE_CONTRACT_GETEKEND'].quantile(0.25)
mid_percentile = df_processed['DAYS_RETENTIE_CONTRACT_GETEKEND'].quantile(0.50)
high_percentile = df_processed['DAYS_RETENTIE_CONTRACT_GETEKEND'].quantile(0.75)

def categorize_duur(row):
    if pd.isnull(row['DAYS_RETENTIE_CONTRACT_GETEKEND']):
        return 'Niet getekend?'
    elif row['DAYS_RETENTIE_CONTRACT_GETEKEND'] <= low_percentile:
        return 'Vroeg getekend'
    elif low_percentile < row['DAYS_RETENTIE_CONTRACT_GETEKEND'] <= mid_percentile:
        return 'Gemiddeld getekend'
    else:
        return 'Laat getekend'

df_processed['Moment_van_tekenen'] = df_processed.apply(
    categorize_duur,
    axis=1
)

display(df_processed['Moment_van_tekenen'].value_counts())
display(df_processed.groupby('Moment_van_tekenen').mean())

# COMMAND ----------

# MAGIC %md
# MAGIC #6. Segmentation analysis

# COMMAND ----------

df_processed.columns.to_list()

# COMMAND ----------

#maak van moment van tekenen ook 3 categorische columns
def preprocess_categorical(df, categorical_columns):
    """
    Optimized version: Faster preprocessing of categorical data, including Y/N conversion and One-Hot Encoding.
    Additionally, sets specified columns to 0 where they have NULLs.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - categorical_columns (list): List of categorical columns to process.
    - null_to_zero_columns (list): List of columns to set to 0 where they have NULLs.
    
    Returns:
    - pd.DataFrame: Transformed DataFrame.
    """
    df = df.copy()  # Avoid modifying the original DataFrame
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=False, drop_first=False)
    return df

# Apply function
categorical_cols = [
    'Moment_van_tekenen'
    ]

df_processed = preprocess_categorical(df_processed, categorical_cols)
df_processed.head()

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

# Select relevant features
features = [
 'HOEVAAK_KLANT_GEWEEST',
 'AANTAL_MAANDEN_KLANT',
 #'AGE',
 'TERUGLEVERING_JA',
 'LIVE_CONTACTS',
 'ENERGIEPROFIEL_INGEVULD',
 'ONLINE_VISITS',
 'CONSENT_GEGEVEN',
 'LOYALTY_KLANT',
 'ORIENTATIE_SCORE',
 'DAYS_RETENTIE_CONTRACT_GETEKEND',
#  'Moment_van_tekenen_Gemiddeld getekend',
#  'Moment_van_tekenen_Laat getekend',
#  'Moment_van_tekenen_Vroeg getekend',
 'KANAAL_Face 2 Face',
 'KANAAL_Inbound',
 'KANAAL_Online',
 'KANAAL_Online Partners',
 'KANAAL_Outbound',
 'KANAAL_Retail',
 ]

# Normalize/scale the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_processed[features])

# COMMAND ----------

#checken of de scaling goed is gegaan 
df_scaled_check = pd.DataFrame(df_scaled, columns=features)

df_scaled_check.describe().T[['mean', 'std']]

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)
# Plotting the results onto a line graph to observe the 'Elbow'
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', markersize=5)
plt.title('Elbow Method For Optimal Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# COMMAND ----------

from sklearn.cluster import KMeans# Performing K-means clustering with 4 clusters

k = 4
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
cluster_labels = kmeans.fit_predict(df_scaled)

# Add the cluster labels to the original dataset, start cluster numbering from 1
df_processed['Cluster'] = cluster_labels + 1  

# Previewing the data with clusters
df_processed[features].head()

# Calculate the mean of each feature within each cluster to characterize the personas
cluster_characteristics = df_processed.groupby('Cluster')[features].mean()

# Calculate the mean age within each cluster
cluster_age_means = df_processed.groupby('Cluster')['AGE'].mean()

# Impute the NaN values in 'LEEFTIJD' with the mean age of the corresponding cluster
def impute_age(row):
    if pd.isna(row['AGE']):
        return cluster_age_means[row['Cluster']]
    else:
        return row['AGE']

df_processed['AGE'] = df_processed.apply(impute_age, axis=1)

# Calculate the mean age within each cluster again after imputation
cluster_age = df_processed.groupby('Cluster')['AGE'].mean()
#cluster_live_contact = df_processed.groupby('Cluster')['live_contact'].mean()

# Combine the two sets of characteristics
cluster_personas = pd.concat([cluster_characteristics, cluster_age], axis=1)
display(cluster_personas)

# COMMAND ----------

cluster_counts = df_processed['Cluster'].value_counts().sort_index()

numbers = cluster_counts.tolist()
total = sum(numbers)
percentages = [round(n / total * 100, 2) for n in numbers]

for idx, (n, p) in enumerate(zip(numbers, percentages), start=1):
    print(f"Cluster {idx}: {n} ({p}%)")

# COMMAND ----------

import seaborn as sns 
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
#components = pca.fit_transform(df_processed)
components = pca.fit_transform(df_scaled)
df_processed['PC1'], df_processed['PC2'] = components[:,0], components[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df_processed, 
    x='PC1', y='PC2', 
    hue='Cluster', 
    palette='Set2', 
    alpha=0.7
)
plt.title('Customer Clusters in PCA Space')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC !! CLUSTER 2 IS PAULIEN

# COMMAND ----------

import matplotlib.pyplot as plt

cluster_counts = df_processed['Cluster'].value_counts().sort_index()

plt.figure(figsize=(6,4))
cluster_counts.plot(kind='bar', color='#5FB623', edgecolor='black')
plt.title('Verdeling van de Clusters (aantallen)')
plt.xlabel('Cluster')
plt.ylabel('Aantal')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# COMMAND ----------

current_date = datetime.now().strftime('%d%m%y')

#naar parquet schrijven om makkelijker op te halen zonder alle code te hoeven runnen 
df_processed.to_parquet(
    f'/Workspace/data_analysts/MM/retentie_clusteranalyse/data/total_clusters_df_{current_date}.parquet',
    index=False
)

# COMMAND ----------

# ALLEEN RUNNEN ALS JE WILT EXPORTEREN NAAR CSV - DUURT LANG! 

#moet in aparte csv's want 1 is te groot 
current_date = datetime.now().strftime('%d%m%y')

for cluster_label, cluster_df in df_processed.groupby('Cluster'):
    cluster_df.to_csv(f'/Workspace/data_analysts/MM/retentie_clusteranalyse/data/df_clusters_{current_date}_cluster_{cluster_label}.csv', index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # 7. Cluster info

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load parquet dataset 
# MAGIC
# MAGIC Vanaf codeblok hieronder inladen als je opnieuw opstart. Alle regels hierboven zijn dan al gedaan en zitten in de parquet file.

# COMMAND ----------

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import numpy as np
import plotly.express as px
import json
from scipy.stats import norm
from itertools import combinations
from scipy.stats import chi2_contingency
from datetime import datetime

# COMMAND ----------

#inlezen van de cluster dataset
df_clusters = pd.read_parquet('/Workspace/data_analysts/MM/retentie_clusteranalyse/data/total_clusters_df_190226.parquet',)
df_clusters.head()

# COMMAND ----------

# DBTITLE 1,visualisation of characteristics
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import OneHotEncoder

# [010] Essent kleurdefinities
essent_bordeaux = '#B8255F'
essent_white    = '#FFFFFF'
ed_main         = '#5FB623'
blue_lighten    = '#3375BB'
grey_90         = '#BEBEBE'
light_grey      = '#d3d3d3'
axis_grey       = '#949494'

# [020] Input: features en clusterkolom
feature_cols = [
    # 'AANTAL_MAANDEN_KLANT',
    'TERUGLEVERING_JA',
    # 'LIVE_CONTACTS',
    'ENERGIEPROFIEL_INGEVULD',
    # 'ONLINE_VISITS',
    'CONSENT_GEGEVEN','LOYALTY_KLANT', 'EDM_HH_KOOPKR','EDM_HH_LVNSFS2','EDM_HH_SOCKLAS','EDM_HH_WONTYP2','EDM_HH_WON_EIG','EDM_HH_E_TYPE','EDM_HH_MBEWUST'
    # 'ORIENTATIE_SCORE','DAYS_RETENTIE_CONTRACT_GETEKEND',
    # 'EDM_PROV','EDM_HH_4GEOTYP','EDM_HH_LVNSFS2','EDM_HH_OPLEID2','EDM_HH_FUNCTIE',
    # 'EDM_HH_WONTYP2','EDM_HH_WON_EIG','EDM_HH_ZP_AANW','EDM_HH_E_TYPE','EDM_HH_BS_SRT',
    # 'EDM_HH_B_SPAAR','EDM_HH_RISAVRS','EDM_HH_FINTYPE',

]
cluster_col = 'Cluster'  # let op: exact zoals in je data

# [030] Data selecteren
df_features = df_clusters[feature_cols + [cluster_col]].copy()
df_features[cluster_col] = df_features[cluster_col].astype(str)  # prettige labels

# [040] Splits numeriek vs categorisch
numeric_cols = []
cat_cols = []
for c in feature_cols:
    if df_features[c].dtype.kind in ('i','u','f') and not c.startswith('EDM_'):
        numeric_cols.append(c)
    else:
        cat_cols.append(c)

# [050] Voorbewerking: missings
df_prep = df_features.copy()
for c in numeric_cols:
    df_prep[c] = pd.to_numeric(df_prep[c], errors='coerce')
    df_prep[c] = df_prep[c].fillna(df_prep[c].median())
for c in cat_cols:
    df_prep[c] = df_prep[c].astype('string').fillna('Onbekend')

# [060] One-hot encoding categorisch
# Let op: in nieuwere sklearn versies is 'sparse_output=False'; in veel Databricks omgevingen werkt 'sparse=False' breder.
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
ohe_matrix = ohe.fit_transform(df_prep[cat_cols])
ohe_cols = [f"{orig}={cat}" for orig, cats in zip(cat_cols, ohe.categories_) for cat in cats]
df_cat = pd.DataFrame(ohe_matrix, columns=ohe_cols, index=df_prep.index)

# [070] Combineer numeriek + categorisch, voeg cluster toe
df_num = df_prep[numeric_cols].copy()
X = pd.concat([df_num, df_cat], axis=1)
X[cluster_col] = df_prep[cluster_col].values

# [080] Totaal-statistieken
# Numeriek: z-score t.o.v. totaal
num_means = df_num.mean()
num_stds  = df_num.std(ddof=0).replace(0, np.nan)

# Categorisch (OHE): prevalentie totaal
cat_means = df_cat.mean()

# [090] Per-cluster aggregatie
grouped = X.groupby(cluster_col)

# Numeriek: gemiddelde -> z-score vs totaal
if numeric_cols:
    num_by_cluster = grouped[numeric_cols].mean()
    num_z_by_cluster = (num_by_cluster - num_means) / num_stds
else:
    num_z_by_cluster = pd.DataFrame(index=grouped.size().index)

# Categorisch: log-lift
cat_by_cluster = grouped[df_cat.columns].mean()  # p_cluster
eps = 1e-9
lift_by_cluster = cat_by_cluster / (cat_means + eps)
cat_score_by_cluster = np.log(lift_by_cluster + eps)  # 0 = gelijk aan totaal; >0 overrepr.

# [100] Combineer scorematrix
score_matrix = pd.concat([num_z_by_cluster, cat_score_by_cluster], axis=1)

# [110] Normaliseer per feature naar [0,1] voor de heatmap
def minmax01(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.5, index=s.index)  # vlak: neutrale middenkleur
    return (s - mn) / (mx - mn)

heat_matrix_full = score_matrix.apply(minmax01, axis=0)

# [120] Selectie: sorteer o.b.v. cluster '1' en pak top 15 kolommen
if "2" not in heat_matrix_full.index:
    raise ValueError("Cluster '1' niet gevonden. Controleer de waarden/typen in de kolom 'Cluster'.")

col_order_by_cl3 = heat_matrix_full.loc["2"].sort_values(ascending=False).index
top15_cols = col_order_by_cl3[:20]
heat_matrix = heat_matrix_full[top15_cols]

# [130] Essent colormap (blauw → wit → bordeaux)
essent_cmap = LinearSegmentedColormap.from_list(
    "essent_cmap",
    [blue_lighten, essent_white, essent_bordeaux]
)

# [140] Plot
plt.figure(figsize=(max(20, len(heat_matrix.columns)*0.6), 4 + 0.4*len(heat_matrix.index)))

ax = sns.heatmap(
    heat_matrix,
    cmap=essent_cmap,
    cbar_kws={'label': 'Score'},
    linewidths=0.2,
    linecolor=light_grey
)

ax.set_title("Cluster Paspoort: Top 15 kenmerken met hoogste score voor cluster 3", fontsize=12, pad=12)
ax.set_xlabel("Kenmerken", color=axis_grey)
ax.set_ylabel("Cluster", color=axis_grey)

# As-styling in Essent-grijs
ax.tick_params(axis='x', colors=axis_grey)
ax.tick_params(axis='y', colors=axis_grey)

plt.xticks(rotation=60, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# [150] (Optioneel) Toon de top-15 kenmerk-namen expliciet
top15_labels = pd.Series(top15_cols, name="Top15_kenmerken_op_cluster_1")
#display(top15_labels)

# COMMAND ----------

# MAGIC %md
# MAGIC # Visuele verdeling van de clusters

# COMMAND ----------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# [010] Basis
df = df_clusters.copy()
df["Cluster"] = df["Cluster"].astype(str)

# [020] Feature-indeling
numeric_cols = [
    c for c in df.columns
    if c != "Cluster"
    and df[c].dtype.kind in ("i", "u", "f")
]

categorical_cols = [
    c for c in df.columns
    if c != "Cluster"
    and c not in numeric_cols
]


# COMMAND ----------

# MAGIC %md
# MAGIC **Leeftijd & levensfase (sterk onderscheidend)**
# MAGIC Dominante leeftijdsgroepen
# MAGIC - 55 jaar en ouder: ± 60–65% van het cluster
# MAGIC - 65+: ± 35–40%
# MAGIC - <40 jaar: < 10%
# MAGIC
# MAGIC _➡️ Dit is duidelijk een ouder cluster, met veel (pre-)pensioenleeftijden._
# MAGIC Levensfase / huishouden
# MAGIC - Paren zonder kinderen: ± 40%
# MAGIC - Alleenstaanden (middelbaar/oud): ± 30%
# MAGIC - Gezinnen met (jonge of oudere) kinderen: ± 30%
# MAGIC
# MAGIC _➡️ Oververtegenwoordiging van kinderloze huishoudens en alleenstaanden._
# MAGIC
# MAGIC **Woning & woonomgeving**
# MAGIC Woningbezit
# MAGIC - Koopwoning: ± 70–75%
# MAGIC - Huurwoning: ± 25–30%
# MAGIC
# MAGIC Woningtype
# MAGIC - Rijtjeshuis / twee-onder-een-kap / vrijstaand: ± 65%
# MAGIC - Appartement / etagewoning: ± 35%
# MAGIC
# MAGIC WOZ-waarde (indicatief)
# MAGIC - €250k – €450k: grootste groep (± 50%)
# MAGIC - > €500k: ± 20–25%
# MAGIC - < €250k: ± 25%
# MAGIC
# MAGIC _➡️ Overwegend koopwoningen met middelhoge tot hoge waarde._
# MAGIC
# MAGIC **Energie & verduurzaming**
# MAGIC Zonnepanelen
# MAGIC - Zonnepanelen aanwezig: ± 60–65%
# MAGIC - Geen zonnepanelen / onbekend: ± 35–40%
# MAGIC
# MAGIC Energieverbruik
# MAGIC - Elektriciteit: vooral 2.500 – 3.750 kWh
# MAGIC - Gas: vaak 1.100 – 1.600 m³
# MAGIC
# MAGIC Veel huishoudens vallen in:
# MAGIC - Luxe grootverbruikers
# MAGIC - Modale gezinsverbruikers
# MAGIC - Gedwongen kleinverbruikers
# MAGIC
# MAGIC _➡️ Gemiddeld tot hoger verbruik, passend bij grotere koopwoningen._
# MAGIC
# MAGIC **Financieel profiel**
# MAGIC Inkomen
# MAGIC - €70k – €100k: grootste groep (± 35–40%)
# MAGIC - > €100k: ± 25–30%
# MAGIC - < €50k: ± 15–20%
# MAGIC
# MAGIC Vermogen
# MAGIC - Veel klanten in vermogensdeciel 7–10
# MAGIC
# MAGIC _➡️ Financieel relatief sterk en behoudend._
# MAGIC
# MAGIC **Gedrag & houding (samenvattend)**
# MAGIC Segmentlabels die vaak voorkomen
# MAGIC - Actieve Vijftigplussers
# MAGIC - Geïnteresseerde Conservatieven
# MAGIC - Traditionele Dorpelingen
# MAGIC - Behulpzame / Behoudende Weldoeners
# MAGIC
# MAGIC Kenmerkend gedrag
# MAGIC - Minder online/extreem digitaal dan jongere clusters
# MAGIC - Loyaler, minder switch-gericht
# MAGIC - Minder impulsief, meer behouden
# MAGIC - Relatief hoge retentie / contractduur
# MAGIC
# MAGIC
# MAGIC **Samenvattende clusterbeschrijving**
# MAGIC
# MAGIC Cluster 3 bestaat voornamelijk uit oudere, financieel stabiele huishoudens, vaak zonder thuiswonende kinderen, die in koopwoningen wonen met een middelhoge tot hoge woningwaarde. Ze hebben gemiddeld tot hoog energieverbruik, relatief vaak zonnepanelen en een behoudend financieel en consumptieprofiel. Dit cluster is minder prijs- en switch-gedreven, maar wel gevoelig voor betrouwbaarheid, continuïteit en duidelijke waardeproposities rond comfort en duurzaamheid.

# COMMAND ----------

def plot_categorical_distribution(df, col):
    df_plot = df.copy()
    df_plot["Cluster"] = df_plot["Cluster"].astype(str)
    dist = pd.crosstab(df_plot["Cluster"], df_plot[col], normalize='index')
    dist.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 4),
        colormap="tab20"
    )
    plt.title(f"Verdeling {col} per cluster\n(Let vooral op Cluster 2)", fontsize=13)
    plt.ylabel("Aandeel")
    plt.xlabel("Cluster")
    plt.legend(title=col, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

categorical_columns = [
    "EDM_HH_SOCKLAS", "EDM_HH_WONTYP2", "EDM_HH_MBEWUST", "EDM_HH_E_TYPE", "EDM_HH_WON_EIG",
    "ENERGIEPROFIEL_INGEVULD", "EDM_HH_LVNSFS2", "CONSENT_GEGEVEN", "EDM_HH_KOOPKR", "LOYALTY_KLANT",
    "AGE", "EDM_HH_LVNSFS2", "EDM_HH_WRKUREN", "EDM_HH_WONWOZW", "EDM_HH_ZP_AANW", "EDM_HH_KWH_VBR",
    "EDM_HH_GAS_VBR", "EDM_HH_E_TYPE", "EDM_HH_VERD_SB", "EDM_HH_FINTYPE", "EDM_HH_PA_AUTO", "EDM_HH_VAK_ACC"
]

for col in categorical_columns:
    plot_categorical_distribution(df_clusters, col)
