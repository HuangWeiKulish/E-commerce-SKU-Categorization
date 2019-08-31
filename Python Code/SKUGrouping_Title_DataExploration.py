import pandas as pd

### Read Data------------------------------
Baby = pd.read_csv('shopee_ID_data_Baby_20170820_20171120.csv')
Home = pd.read_csv('shopee_ID_data_Home_20170820_20171120.csv')

### Data exploration ----------------------
#################### Baby ####################
list(Baby)# Check the column names
Baby.cat1_name.unique() # Unique values in 'cat1_name' (only 1 category 'Baby & Kids', not useful)
Baby['cat2_name'].value_counts() # Frequency of each unique value in 'cat2_name' (category unlikely to overlap, used for category segmentation before training)
len(Baby.cat3_name.unique()) # Number of unique values in 'cat3_name' (36 categories)
Baby['cat3_name'].value_counts() # Frequency of each unique value in 'cat3_name' (category can overlap: combine with item name for training)
1.0*Baby['model_name'].isnull().sum()/len(Baby) # Percentage of missing value in column 'model_name' (34.0% missing: combine with item name for training)
1.0*Baby['brand'].isnull().sum()/len(Baby) # Percentage of missing value in column 'brand' (32.1% missing: combine with item name for training)
Baby['brand'].value_counts() # many brand names are not really brand but part of item names
1.0*Baby['item_name'].isnull().sum()/len(Baby) # Percentage of missing value in column 'item_name' (0.0% missing)

Baby_cat_freq = Baby.groupby(['cat2_name','cat3_name'])['cat3_name'].count().unstack('cat3_name').fillna(0)
Baby_cat_freq.plot(kind='bar', stacked=True, legend=False).legend(loc='center left', bbox_to_anchor=(1.05,0.5), ncol=2, prop={'size': 8})

#################### Home ####################
list(Home)# Check the column names
Home.cat1_name.unique() # Unique values in 'cat1_name' (only 1 category 'Home & Living', not useful)
Home['cat2_name'].value_counts() # Frequency of each unique value in 'cat2_name' (category less likely to overlap, used pretraining)
len(Home.cat3_name.unique()) # Number of unique values in 'cat3_name' (28 categories)
Home['cat3_name'].value_counts() # Frequency of each unique value in 'cat3_name' (category can overlap: combine with item name for training)
# Need to remove 'Others' from cat3_name
1.0*Home['model_name'].isnull().sum()/len(Home) # Percentage of missing value in column 'model_name' (51.9.0% missing: combine with item name for training)
1.0*Home['brand'].isnull().sum()/len(Home) # Percentage of missing value in column 'brand' (33.4% missing: combine with item name for training)
Home['brand'].value_counts() # many brand names are not really brand but part of item names
list(Home['brand'].value_counts().index)
1.0*Home['item_name'].isnull().sum()/len(Home) # Percentage of missing value in column 'item_name' (0.0% missing)

Home_cat_freq = Home.groupby(['cat2_name','cat3_name'])['cat3_name'].count().unstack('cat3_name').fillna(0)
Home_cat_freq.plot(kind='bar', stacked=True, legend=False).legend(loc='center left', bbox_to_anchor=(1.05,0.5), ncol=2, prop={'size': 8})

