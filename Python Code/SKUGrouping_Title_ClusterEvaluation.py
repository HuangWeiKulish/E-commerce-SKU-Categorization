import json
import pandas as pd
import matplotlib.pyplot as plt


# A function to evaluate the clustring of titles
def Evaluation(DataEvlList):
    # build a data frame to store the result:
    EvalData = pd.DataFrame(index=DataEvlList.keys(),
                            columns=['NumberItems','LDA_topics','NumberGroups','ProcessingTime',
                                     'AvePctTitleMatch_min','AvePctTitleMatch_mean','AvePctTitleMatch_max'])
    for i in range(0,EvalData.shape[0]):
        catName = EvalData.index[i]
        # Add data to data frame
        EvalData['NumberItems'][i] = DataEvlList[catName]['NumberItems']
        EvalData['LDA_topics'][i] = DataEvlList[catName]['LDA_topics']
        EvalData['NumberGroups'][i] = DataEvlList[catName]['NumberGroups']
        EvalData['ProcessingTime'][i] = DataEvlList[catName]['ProcessingTime']
        EvalData['AvePctTitleMatch_min'][i] = min(DataEvlList[catName]['AvePctTitleMatch_Distribution'])
        EvalData['AvePctTitleMatch_mean'][i] = sum(DataEvlList[catName]['AvePctTitleMatch_Distribution'])/\
                                               len(DataEvlList[catName]['AvePctTitleMatch_Distribution'])
        EvalData['AvePctTitleMatch_max'][i] = max(DataEvlList[catName]['AvePctTitleMatch_Distribution'])

        # save data frame
        EvalData.to_csv('TitleCluster_Evaluation.csv')

        # Save Average Percentage Title Match Distribution plot
        DistributionData = DataEvlList[catName]['AvePctTitleMatch_Distribution']
        figure = plt.figure()
        plt.hist(DistributionData, normed=False, bins=len(DistributionData))
        plt.xlabel('Average Percentage Title Match')
        plt.ylabel('Count')
        plt.title('Average Percentage Title Match Distribution \n' + catName)
        PlotName = catName[0:-5]+'AvePctTitleMatch_Distribution.png'
        plt.xlim([0,1])
        plt.savefig(PlotName)
        plt.close(figure)

DataEvlList = {}
### Open files ###
# Baby
with open('Baby_Girl_Clothes_69Groups_137Topics_evalu.json') as json_data:
    DataEvlList['Baby_Girl_Clothes_69Groups_137Topics_evalu']=json.load(json_data)
with open('Baby_Girl_Clothes_44Groups_274Topics_evalu.json') as json_data:
    DataEvlList['Baby_Girl_Clothes_44Groups_274Topics_evalu']=json.load(json_data)

with open('Baby_Boy_Clothes_78Groups_184Topics_evalu.json') as json_data:
    DataEvlList['Baby_Boy_Clothes_78Groups_184Topics_evalu'] = json.load(json_data)
with open('Baby_Boy_Clothes_69Groups_92Topics_evalu.json') as json_data:
    DataEvlList['Baby_Boy_Clothes_69Groups_92Topics_evalu'] = json.load(json_data)

with open('Baby_Baby_Kids_Toys_95Groups_130Topics_evalu.json') as json_data:
    DataEvlList['Baby_Baby_Kids_Toys_95Groups_130Topics_evalu'] = json.load(json_data)
with open('Baby_Baby_Kids_Toys_63Groups_65Topics_evalu.json') as json_data:
    DataEvlList['Baby_Baby_Kids_Toys_63Groups_65Topics_evalu'] = json.load(json_data)

with open('Baby_Baby_Food_39Groups_40Topics_evalu.json') as json_data:
    DataEvlList['Baby_Baby_Food_39Groups_40Topics_evalu'] = json.load(json_data)
with open('Baby_Baby_Food_20Groups_20Topics_evalu.json') as json_data:
    DataEvlList['Baby_Baby_Food_20Groups_20Topics_evalu'] = json.load(json_data)

with open('Baby_Baby_Diapers_40Groups_41Topics_evalu.json') as json_data:
    DataEvlList['Baby_Baby_Diapers_40Groups_41Topics_evalu'] = json.load(json_data)
with open('Baby_Baby_Diapers_20Groups_20Topics_evalu.json') as json_data:
    DataEvlList['Baby_Baby_Diapers_20Groups_20Topics_evalu'] = json.load(json_data)

with open('Baby_Baby_Clothes_72Groups_138Topics_evalu.json') as json_data:
    DataEvlList['Baby_Baby_Clothes_72Groups_138Topics_evalu']=json.load(json_data)
with open('Baby_Baby_Clothes_60Groups_69Topics_evalu.json') as json_data:
    DataEvlList['Baby_Baby_Clothes_60Groups_69Topics_evalu'] = json.load(json_data)


#Home
with open('Home_Storage_TransTitles_59Groups_89Topics_evalu.json') as json_data:
    DataEvlList['Home_Storage_TransTitles_59Groups_89Topics_evalu'] = json.load(json_data)
with open('Home_Storage_TransTitles_40Groups_44Topics_evalu.json') as json_data:
    DataEvlList['Home_Storage_TransTitles_40Groups_44Topics_evalu'] = json.load(json_data)

with open('Home_Kitchen_Dining_58Groups_255Topics_evalu.json') as json_data:
    DataEvlList['Home_Kitchen_Dining_58Groups_255Topics_evalu'] = json.load(json_data)
with open('Home_Kitchen_Dining_83Groups_127Topics_evalu.json') as json_data:
    DataEvlList['Home_Kitchen_Dining_83Groups_127Topics_evalu'] = json.load(json_data)

with open('Home_Decoration_79Groups_179Topics_evalu.json') as json_data:
    DataEvlList['Home_Decoration_79Groups_179Topics_evalu'] = json.load(json_data)
with open('Home_Decoration_64Groups_89Topics_evalu.json') as json_data:
    DataEvlList['Home_Decoration_64Groups_89Topics_evalu'] = json.load(json_data)

with open('Home_Bedroom_74Groups_144Topics_evalu.json') as json_data:
    DataEvlList['Home_Bedroom_74Groups_144Topics_evalu'] = json.load(json_data)
with open('Home_Bedroom_66Groups_72Topics_evalu.json') as json_data:
    DataEvlList['Home_Bedroom_66Groups_72Topics_evalu'] = json.load(json_data)

Evaluation(DataEvlList)
