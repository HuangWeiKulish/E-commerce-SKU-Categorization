from sklearn.feature_extraction.text import CountVectorizer
import string
import math
import re
from googletrans import Translator
from word2number import w2n

# Define a stop word set:
stop_words = ['sell', 'buy', 'deal', 'cheap', 'affordable', 'sale', 'sales', 'others', 'other', 'none', 'or']
# Expend stop_words to include malay (ms), indonesian (id), and filipino (tl)
translator = Translator()
for i in range(0, len(stop_words)):
    stop_words.append(str(translator.translate(stop_words[i], dest='ms').text))
    stop_words.append(str(translator.translate(stop_words[i], dest='id').text))
    stop_words.append(str(translator.translate(stop_words[i], dest='tl').text))
stop_words = list(set(stop_words))


# A function to change float nan value to ''
def remove_floatNaN(text):
    if isinstance(text, float):
        if math.isnan(text):
            return ''
    else:
        return text

# A function to remove not valid names or categories (the names which contain no alphabet)
def remove_nonName(text):
    text=repr(text)
    if any(c.isalpha() for c in text)==False:
        text = text.replace(text, '')
    return text

# A function to add space between letters and numbers
def Space_L_D(Str):
    Str2=''
    j=0
    for i in range(0,len(Str)-1):
        if Str[i].isdigit() != Str[i+1].isdigit():
            Str2 = Str2+" "+Str[j:i+1]
            j=i+1
    Str2=Str2+" "+Str[j:len(Str)]
    return Str2.strip()

# A function to remove stop words from string
def remove_stopwords(Str):
    for w in stop_words:
        pattern = r'\b'+w+r'\b'
        Str = re.sub(pattern, '', Str)
    return Str

# A function to convert number written in words to digits
def conver_number_digit(Str, Dict):
    tokens = Str.split()
    tokens2=''
    for t in tokens:
        try:
            Trans_t = Dict[t]
        except KeyError:
            Trans_t = t
        try:
            ToDigit = w2n.word_to_num(Trans_t)
            tokens2 = tokens2 + ' ' + str(ToDigit)
        except ValueError:
            tokens2 = tokens2 + ' ' + t
    return tokens2.lstrip()

# A function to remove punctuations (all in string.punctuation) in a string
def remove_punctuations_full(Str):
    Str=repr(Str).replace("\\", "") # in case if the string contains single back slash, remove the single back slash first
    for punctuation in string.punctuation:
        Str = Str.replace(punctuation, ' ')
    return Str

# A function to remove punctuations (except &) in a string
def remove_punctuations_part(Str):
    Str=repr(Str).replace("\\", "")  # in case if the string contains single back slash, remove the single back slash first
    for punctuation in '!"#$%\'()*+,-./:;<=>?@[\\]^_`{|}~':
        Str = Str.replace(punctuation, ' ')
    return Str

# A funtion to translate one token to english
def TokenTrans_en(token):
    translator = Translator()
    try:
        return repr(translator.translate(token, dest='en').text).replace("\\", "")
    except ValueError:
        translator = Translator()
        return repr(translator.translate(token, dest='en').text).replace("\\", "")

# A function to get all tokens, and sparse matrix of a given dataset (the dataset must contain a column called 'item_name_full')
def getAllTokens(dataset):
    count_vect = CountVectorizer(binary=True)  # Vectorization
    DTMatrix = count_vect.fit_transform(dataset['item_name_full']) # Document Term Matrix
    Features = count_vect.get_feature_names()  # get all features
    return (Features,DTMatrix)

# A function to find out the start of a non number features' indices in a given list with all ordered tokens
def getStartNonNumberIndex_Features(Features):
    Features_NonNumber_start=0
    for i in range(0, len(Features)):
        if Features[i].isdigit() == True:
            Features_NonNumber_start=Features_NonNumber_start+1
        else:
            break
    return Features_NonNumber_start

# A function to change value in sparse matrix after for non number tokens
def changeWeight_NonNumberTokens_SparseMatrix(DTMatrix,Features_NonNumber_start,ToWeight):
    for i in range(0,DTMatrix.shape[0]):
        for j in list(DTMatrix[i,:].indices):
            if j>=Features_NonNumber_start:
                DTMatrix[i,j]=ToWeight
    return DTMatrix

# A function to get all translation of a give list of features, and the start index of non number tokens
def GetDic(all_tokens,all_tokens_NonNumberStartIndex):
    import time
    start_time = time.time()
    Dict = [0]*len(all_tokens)
    for i in range(all_tokens_NonNumberStartIndex,len(all_tokens)):
        print (i," out of ", len(all_tokens), ' seconds:', time.time() - start_time)
        Dict[i]=TokenTrans_en(all_tokens[i])
    return Dict

# A function to build dictionary
def BuildDict(keys, values):
    dicts = {}
    for i in range(len(keys)):
        print(i)
        dicts[keys[i]] = values[i]
    return dicts

# A function to find out the index of semantically (between different languages: id, ms, tl, en) matched words in the same list
def MapTranslation(SubSet_Features, SubSet_Features_NonNumber_start, Dict):
    map = {}
    for tk in range(SubSet_Features_NonNumber_start, len(SubSet_Features)):
        try:
            mapword = Dict[SubSet_Features[tk]]
            if mapword != SubSet_Features[tk]:
                try:
                    map[tk] = SubSet_Features.index(mapword)
                except ValueError:
                    pass
        except KeyError:
            pass
    return map

# A function to group all values which are semantically equivalent
'''def GroupFeatures(Dic):
    GroupDic = {}
    values = []
    keys = []
    for k, v in Dic.iteritems():
        keys.append(k)
        values.append(v)
    for uniquevalue_i in set(values):
        group = [keys[i] for i, x in enumerate(values) if x == uniquevalue_i]
        group.append(uniquevalue_i)
        for member in group:
            GroupDic[member] = [x for x in group if x != member]
    return GroupDic'''

def GroupFeatures(Dic):
    FeaturesGroup = []
    values = []
    keys = []
    for k, v in Dic.iteritems():
        keys.append(k)
        values.append(v)
    for uniquevalue_i in set(values):
        group = [keys[i] for i, x in enumerate(values) if x == uniquevalue_i]
        group.append(uniquevalue_i)
        FeaturesGroup.append(group)
    return FeaturesGroup






