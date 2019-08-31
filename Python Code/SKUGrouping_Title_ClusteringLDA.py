import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
import math
import json
import SKUFunctions
import time
import pickle
import scipy



# A function to return LDA grouping based on the titles
def Grouping_Title(Data, Ave_Items_per_Gourp, Category):
    '''
    Data=Home_Decoration_TransTitles
    Ave_Items_per_Gourp=500
    Category='Home_Decoration'
    '''

    # Create more columns in dataset
    Data = pd.concat([Data, pd.Series('', name = 'GroupName',index = Data.index),
               pd.Series('', name='GroupTitle', index=Data.index)], axis=1)
    Titles = Data['ItemTransTitle']

    # set start timing
    time_start=time.clock()

    # Pre LDA processing
    print('Processing data')
    count_vect = CountVectorizer(binary=True)  # Vectorization
    DTMatrix = count_vect.fit_transform(Titles.values.astype('U'))  # Document Term Matrix
    DTMatrix_csc = DTMatrix.tocsc()  # convert Compressed Sparse Row format to Compressed Sparse Column format
    Features = count_vect.get_feature_names()  # get all features
    # Find out the start of non number features' indices
    Features_NonNumber_start = SKUFunctions.getStartNonNumberIndex_Features(Features)

    print('Add weight to Non-digit tokens')
    DTMatrix_array = DTMatrix.toarray() # convert sparse matrix for faster calculation
    # Add weight for features which are not pure numbers
    # add weight to non number features (change weight from 1 to 2)
    for ft in range(Features_NonNumber_start, len(Features)):
        # slice column ft (show non-zero row numbers):
        Non0Rows_col_ft = DTMatrix_csc[:, ft].indices
        DTMatrix_array[Non0Rows_col_ft[:, None],ft]=2
        print('Add weight to Non-Number features: {} out of {}'.format(ft, len(Features)))
    DTMatrix1 = scipy.sparse.csr_matrix(DTMatrix_array) # Convert array back to sparse matrix

    print('Calculate tf-idf matrix')
    tf_transformer = TfidfTransformer(use_idf=True).fit(DTMatrix1)
    TfIdfMatrix = tf_transformer.transform(DTMatrix1)

    # training with NoGroups groups (in average 'Ave_Items_per_Gourp' items per group)
    if len(Titles)>500:
        NoGroups = int(1.0 * len(Titles) / Ave_Items_per_Gourp)
    else:
        NoGroups = 1

    print('Train LDA model, number of groups = {}'.format(NoGroups))
    # Train LDA model
    LDA_model = LatentDirichletAllocation(n_components=NoGroups, max_iter=8, learning_method='online',
                                           learning_offset=50., random_state=0).fit(TfIdfMatrix)
    # Subset grouping
    Group_prob = LDA_model.transform(TfIdfMatrix)

    # Assign group number (using topic number) to each row
    print('Assign group numbers')
    Group_number = []
    for i in range(0, TfIdfMatrix.shape[0]):
        Group_number.append(list(Group_prob[i]).index(max(Group_prob[i])))
        Data.set_value(i, 'GroupName', Category+'_'+repr(Group_number[i]))
        print('Assign group names: {} out of {}'.format(i, TfIdfMatrix.shape[0]))

    TitleNumberWords=[]
    for i in range(0, DTMatrix.shape[0]):
        print('Calculate number of tokens in each item: {} '.format(i))
        TitleNumberWords.append( sum(DTMatrix[i].data) )

    # Get average number of tokens to be used in each topic: for each topic, average the words used in top 10 documents according to probability
    Topic_nWords = []
    for i in range(0, Group_prob.shape[1]):
        Topic_nWords.append(int(math.ceil( np.array(TitleNumberWords)[list(np.argsort(-Group_prob[:, i])[0:10])].mean() )))
        print('Calculate # Tokens to be used in each topic: {} out of {}'.format(i, Group_prob.shape[1]))

    # Get the most representative title of each group, number of words are determined using above method, and stored in Topic_nWords
    Group_title = []
    for topic_idx, topic in enumerate(LDA_model.components_):
        Group_title.append(" ".join([Features[i] for i in topic.argsort()[:-Topic_nWords[topic_idx] - 1:-1]]))
    # Add title into data frame
    for i in range(0, Data.shape[0]):
        Data.set_value(i, 'GroupTitle', Group_title[Group_number[i]])
        print('Assign group titles: {} out of {}'.format(i, Data.shape[0]))

    # Save CSV file
    print('Save dataset')
    DataName = Category+'_'+repr(len(Data['GroupName'].unique()))+'Groups_'+repr(NoGroups)+'Topics.csv'
    Data.to_csv(DataName, index=False)

    # Access the Grouping Quality: calculate average percentage match:
    # for the given group title, calculate percentage match with each item title within that group, and then take average
    # percentage match is calculated by:
    #    number of tokens overlap in item title and group title / total number of tokens in the group title
    AvePctMatch = []
    for i in set(Group_number):
        group_title = str(Group_title[i]).split() # change from unicode to string type, and then tokenize
        DocsNo_i = list(np.where(np.array(Group_number) == i)[0])  # Get all documents belonging to group i
        PctMatch = 0.0
        for docno_i in DocsNo_i:
            if isinstance(Data['ItemTransTitle'][docno_i], float):
                Data['ItemTransTitle'][docno_i] = repr(Data['ItemTransTitle'][docno_i])
                print(docno_i,'yes')
            else:
                item_title = Data['ItemTransTitle'][docno_i].split()
            PctMatch = PctMatch + 1.0 * len(set(group_title) & set(item_title)) / len(group_title)
        AvePctMatch.append(PctMatch / len(DocsNo_i))

    time_end = time.clock()# end time

    # set evaluation metrics
    print('Evaluate Grouping')
    evaluation={}
    evaluation['NumberItems'] = Data.shape[0]
    evaluation['NumberGroups'] = len(Data['GroupName'].unique())
    evaluation['LDA_topics'] = NoGroups
    evaluation['AvePctTitleMatch_Distribution'] = AvePctMatch
    evaluation['ProcessingTime'] = time_end - time_start
    evaluation_Name = DataName[0:-4] + '_evalu.json'
    with open(evaluation_Name, 'w') as outfile:
        json.dump(evaluation, outfile)

    # dump model for future use
    print('Save LDA model')
    model = (Features, LDA_model.components_, LDA_model.exp_dirichlet_component_, LDA_model.doc_topic_prior_)
    modelName = DataName[0:-4] + '_LDA.pkl'
    with open(modelName, 'wb') as fp:
        pickle.dump(model, fp)


###################### GROUPING ##########################
Baby_Baby_Clothes_TransTitles = pd.read_csv('Baby_Baby_Clothes_TransTitles.csv')
Grouping_Title(Baby_Baby_Clothes_TransTitles, 1000, 'Baby_Baby_Clothes') # adjust the numbers: larger number => less topics will be generated

Baby_Baby_Diapers_TransTitles = pd.read_csv('Baby_Baby_Diapers_TransTitles.csv')
Grouping_Title(Baby_Baby_Diapers_TransTitles, 1000, 'Baby_Baby_Diapers')

Baby_Baby_Food_TransTitles = pd.read_csv('Baby_Baby_Food_TransTitles.csv')
Grouping_Title(Baby_Baby_Food_TransTitles, 1000, 'Baby_Baby_Food')

Baby_Baby_Kids_Toys_TransTitles = pd.read_csv('Baby_Baby_Kids_Toys_TransTitles.csv')
Grouping_Title(Baby_Baby_Kids_Toys_TransTitles, 1000, 'Baby_Baby_Kids_Toys')

Baby_Boy_Clothes_TransTitles = pd.read_csv('Baby_Boy_Clothes_TransTitles.csv')
Grouping_Title(Baby_Boy_Clothes_TransTitles, 1000, 'Baby_Boy_Clothes')

Baby_Girl_Clothes_TransTitles = pd.read_csv('Baby_Girl_Clothes_TransTitles.csv')
Grouping_Title(Baby_Girl_Clothes_TransTitles, 1000, 'Baby_Girl_Clothes')


Home_Storage_TransTitles = pd.read_csv('Home_Storage_TransTitles.csv')
Grouping_Title(Home_Storage_TransTitles, 1000, 'Home_Storage')

Home_Kitchen_Dining_TransTitles = pd.read_csv('Home_Kitchen_Dining_TransTitles.csv')
Grouping_Title(Home_Kitchen_Dining_TransTitles, 1000, 'Home_Kitchen_Dining')

Home_Decoration_TransTitles = pd.read_csv('Home_Decoration_TransTitles.csv')
Grouping_Title(Home_Decoration_TransTitles, 1000, 'Home_Decoration')

Home_Bedroom_TransTitles = pd.read_csv('Home_Bedroom_TransTitles.csv')
Grouping_Title(Home_Bedroom_TransTitles, 1000, 'Home_Bedroom')