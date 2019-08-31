import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import math
import json
import SKUFunctions
import time
import pickle
import scipy
### Read Data------------------------------
# Datasets:
Baby1 = pd.read_csv('Baby1.csv')
Home1 = pd.read_csv('Home1.csv')


# Dictionaries
with open('Baby1_dictionary.json') as json_data:
    Baby1_dict = json.load(json_data)
print(Baby1_dict)

with open('Home1_dictionary.json') as json_data:
    Home1_dict = json.load(json_data)
print(Home1_dict)

### Grouping based on title only ----------------------
## check data --------------------------
list(Baby1)
#Baby1.drop('Unnamed: 0', axis=1, inplace=True)
list(Home1)
#Home1.drop('Unnamed: 0', axis=1, inplace=True)

# Subset useful columns
Baby1_1 = Baby1[['itemid','cat2_name','item_name_full']]
Home1_1 = Home1[['itemid','cat2_name','item_name_full']]

'''
DataSet=Baby1_1
list(DataSet)
Dict=Baby1_dict
DataSet['cat2_name'].value_counts()
Cat1Name = 'Baby'
'''

'''
DataSet=Home1_1
list(DataSet)
Dict=Home1_dict
DataSet['cat2_name'].value_counts()
Cat1Name = 'Home'
'''
###################### Translate Title ###################
def TransTitle(DataSet, Dict, Cat1Name):
    Cats = list(DataSet['cat2_name'].unique())
    # Create more columns in dataset
    DataSet = pd.concat([DataSet, pd.Series('', name='ItemTransTitle', index=DataSet.index)], axis=1)

    for cat in Cats: # cat = Cats[3]
        print('Processing Category: {}'.format(cat))
        SubSet = DataSet.loc[DataSet['cat2_name'] == cat]
        SubSet['cat2_name']
        Titles = SubSet['item_name_full']

        count_vect = CountVectorizer(binary=True)  # Vectorization
        SubSet_DTMatrix = count_vect.fit_transform(Titles)  # Document Term Matrix
        SubSet_DTMatrix_csc = SubSet_DTMatrix.tocsc()  # convert Compressed Sparse Row format to Compressed Sparse Column format
        SubSet_Features = count_vect.get_feature_names()  # get all features
        # Find out the start of non number features' indices
        SubSet_Features_NonNumber_start = SKUFunctions.getStartNonNumberIndex_Features(SubSet_Features)

        MatchIndex = SKUFunctions.MapTranslation(SubSet_Features, SubSet_Features_NonNumber_start, Dict)

        SubSet_DTMatrix_array = SubSet_DTMatrix.toarray()

        FeatureGroup = SKUFunctions.GroupFeatures(MatchIndex)

        # for a column A's matched column B (in the same feature group): column A has non zero rows A_rows, change A_rows of column B to 2
        for i in range(0, len(FeatureGroup)):
            RowsToModify = []
            for j in range(0, len(FeatureGroup[i])):
                RowsToModify = RowsToModify + list(SubSet_DTMatrix_csc[:, FeatureGroup[i][j]].indices)
            RowsToModify = np.array(list(set(RowsToModify)))
            SubSet_DTMatrix_array[RowsToModify[:, None], FeatureGroup[i]] = 1
            print(
            'Category: {} --- Map words of different languages: {} groups out of {}'.format(cat, i, len(FeatureGroup)))

        # get all original document row numbers
        DocsID = list(Titles.index)

        # Get the translation modified titles for each document:
        for i in range(0, len(Titles)):
            iCol_Non0Rows = list(np.where(SubSet_DTMatrix_array[i]!=0)[0])
            SubSet.set_value(DocsID[i], 'ItemTransTitle', ' '.join(list(np.array(SubSet_Features)[iCol_Non0Rows])))
            print('Category: {} --- Save translated titles: {} out of {}'.format(cat, i, len(Titles)))

        # Save the dataset
        print('Save data')
        DatasetName = Cat1Name+'_'+'_'.join(SKUFunctions.remove_punctuations_full(cat).split())+'_TransTitles.csv'
        SubSet.to_csv(DatasetName,index=False)


TransTitle(Baby1_1, Baby1_dict,'Baby')
TransTitle(Home1_1, Home1_dict,'Home')
