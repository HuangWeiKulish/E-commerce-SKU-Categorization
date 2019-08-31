import pandas as pd
import json
import SKUFunctions

# Read Data: Subset useful columns
Baby1 = pd.read_csv('shopee_ID_data_Baby_20170820_20171120.csv').iloc[:,0:10]
Home1 = pd.read_csv('shopee_ID_data_Home_20170820_20171120.csv').iloc[:,0:10]


#################### Baby ####################
# Clean 'brand'
Baby1['brand']=Baby1['brand'].apply(SKUFunctions.remove_floatNaN).replace(['None', '0'], '').apply(SKUFunctions.remove_nonName)
Baby1['brand'].value_counts()
list(Baby1['brand'].value_counts().index)

# Clean 'model_name'
Baby1['model_name']=Baby1['model_name'].apply(SKUFunctions.remove_floatNaN)
Baby1['model_name'].value_counts()

# Clean 'model_name'
Baby1['cat3_name']=Baby1['cat3_name'].apply(SKUFunctions.remove_floatNaN).replace(['Others'], '')
Baby1['cat3_name'].value_counts()

# Combine column 'item_name', 'model_name', 'brand' and 'cat3_name' into a new column 'item_name_model'
# brand cannot remove punctuation!
Baby1['item_name_full'] = Baby1['item_name'].apply(SKUFunctions.remove_punctuations_full).fillna(' ') + ' ' + \
                          Baby1['model_name'].apply(SKUFunctions.remove_punctuations_full).fillna(' ') + ' ' +\
                          Baby1['brand'].apply(SKUFunctions.remove_punctuations_part).fillna(' ') + ' ' + \
                          Baby1['cat3_name'].apply(SKUFunctions.remove_punctuations_full).fillna(' ')
# For 'item_name_model': lower the case, add space between numbers and letters, remove punctuations and stopwords
Baby1['item_name_full'] = Baby1['item_name_full'].str.lower().apply(SKUFunctions.Space_L_D).apply(SKUFunctions.remove_stopwords)

# get all features in Baby1:
Baby1_all_tokens, Baby1_cat2_sparse_matrix = SKUFunctions.getAllTokens(Baby1)
Baby1_all_tokens_NonNumberStartIndex = SKUFunctions.getStartNonNumberIndex_Features(Baby1_all_tokens)

# Build a dictionary for Baby1
Baby1_Dict = SKUFunctions.GetDic(Baby1_all_tokens,Baby1_all_tokens_NonNumberStartIndex)
Baby1_Dict_dictionary = SKUFunctions.BuildDict(Baby1_all_tokens[Baby1_all_tokens_NonNumberStartIndex:len(Baby1_all_tokens)],
                                               [str(word)[2:-1] for word in
                                                Baby1_Dict[Baby1_all_tokens_NonNumberStartIndex:len(Baby1_all_tokens)]])
# Save dictionary
with open('Baby1_dictionary.json', 'w') as outfile:
    json.dump(Baby1_Dict_dictionary, outfile)

# Check： load dictionary
with open('Baby1_dictionary.json') as json_data:
    Baby1_Dict_dictionary = json.load(json_data)
print(Baby1_Dict_dictionary)

# Convert number written in words to digit
Baby1['item_name_full'] = Baby1['item_name_full'].apply(SKUFunctions.conver_number_digit,args=(Baby1_Dict_dictionary,))

# Save cleaned file
Baby1.to_csv('Baby1.csv',index=False)


#################### Home ####################
# Clean 'brand'
Home1['brand']=Home1['brand'].apply(SKUFunctions.remove_floatNaN).replace(['None', '0'], '').apply(SKUFunctions.remove_nonName)
Home1['brand'].value_counts()
list(Home1['brand'].value_counts().index)

# Clean 'model_name'
Home1['model_name']=Home1['model_name'].apply(SKUFunctions.remove_floatNaN)
Home1['model_name'].value_counts()

# Clean 'model_name'
Home1['cat3_name']=Home1['cat3_name'].apply(SKUFunctions.remove_floatNaN).replace(['Others'], '')
Home1['cat3_name'].value_counts()

# Combine column 'item_name', 'model_name', 'brand' and 'cat3_name' into a new column 'item_name_model'
# brand cannot remove punctuation!
Home1['item_name_full'] = Home1['item_name'].apply(SKUFunctions.remove_punctuations_full).fillna(' ') + ' ' + \
                          Home1['model_name'].apply(SKUFunctions.remove_punctuations_full).fillna(' ') + ' ' + \
                          Home1['brand'].apply(SKUFunctions.remove_punctuations_part).fillna(' ') + ' ' + \
                          Home1['cat3_name'].apply(SKUFunctions.remove_punctuations_full).fillna(' ')
# For 'item_name_model': lower the case, add space between numbers and letters, remove punctuations and stopwords
Home1['item_name_full'] = Home1['item_name_full'].str.lower().apply(SKUFunctions.Space_L_D).apply(SKUFunctions.remove_stopwords)

# get all features in Home1:
Home1_all_tokens, Home1_cat2_sparse_matrix = SKUFunctions.getAllTokens(Home1)
Home1_all_tokens_NonNumberStartIndex = SKUFunctions.getStartNonNumberIndex_Features(Home1_all_tokens)

# Build a dictionary for Home1
Home1_Dict = SKUFunctions.GetDic(Home1_all_tokens, Home1_all_tokens_NonNumberStartIndex)
Home1_Dict_dictionary = SKUFunctions.BuildDict(Home1_all_tokens[Home1_all_tokens_NonNumberStartIndex:len(Home1_all_tokens)],
                                               [str(word)[2:-1] for word in
                                                Home1_Dict[Home1_all_tokens_NonNumberStartIndex:len(Home1_all_tokens)]])

# Save dictionary
with open('Home1_dictionary.json', 'w') as outfile:
    json.dump(Home1_Dict_dictionary, outfile)

#Check：load dictionary
with open('Home1_dictionary.json') as json_data:
    Home1_dictionary = json.load(json_data)
print(Home1_dictionary)

# Convert number written in words to digit
Home1['item_name_full'] = Home1['item_name_full'].apply(SKUFunctions.conver_number_digit,args=(Home1_dictionary,))

# Save cleaned file
Home1.to_csv('Home1.csv',index=False)