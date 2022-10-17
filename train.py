#libraries
import torch
import flair
from flair.data import Sentence
import random
import pandas as pd
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.embeddings import WordEmbeddings, StackedEmbeddings,TokenEmbeddings
from typing import List
from flair.tokenization import SegtokSentenceSplitter
device = torch.device('cpu')
flair.device="cpu"


tagger = SequenceTagger.load('ner-ontonotes')
data = pd.read_csv("model/ner.csv")
ids = data['Sent'].unique()
random_select = list(random.choices(ids, k=5000))
data = data[data['Sent'].isin(random_select)]
data.columns = ["Sent","Word","Tag","POS"]
data = data[["Sent","Word","Tag"]]

list_id=data["Sent"].values.tolist()
set_listid=set(list_id)

training_count=int((80*len(set_listid)) /100)
dev_count= int((10*len(set_listid)) /100)
test_count = len(set_listid) - training_count - dev_count

unique_list = list(set_listid)

random.shuffle(unique_list)

train_data_id = unique_list[:training_count]
dev_data_id = unique_list[training_count:training_count+dev_count]
test_data_id = unique_list[training_count+dev_count:]


# Script to fetch data and write in flair format

def write_data(li,file):
    wr = open(file,'w')
    
    for id_line in li:
        filter_df = data.loc[data["Sent"]==id_line]
        
        ph=filter_df.Word.values.tolist()
        an=filter_df.Tag.values.tolist()
        
        i=0
        while i < len(ph):
            string = ph[i]+" "+an[i]+"\n"
            wr.write(string)
            i=i+1
        wr.write("\n")
    
    wr.close()

write_data(train_data_id,"model/train.txt")
write_data(test_data_id,"model/test.txt")
write_data(dev_data_id,"model/dev.txt")


columns = {0 : 'text', 1 : 'ner'}
# directory where the data resides
data_folder = 'model/'
# initializing the corpus
Corpus = ColumnCorpus(data_folder, columns,
                              train_file = 'train.txt',
                              test_file = 'test.txt',
                              dev_file = 'dev.txt')
    
print(len(Corpus.train))

print(Corpus.train[0].to_tagged_string('ner'))

# tag to predict
tag_type = 'ner'
# make tag dictionary from the corpus
tag_dictionary = Corpus.make_tag_dictionary(tag_type=tag_type)

# custom_embedding = WordEmbeddings('en')

word_embeddings = [ WordEmbeddings('en')]
embedding_types : List[TokenEmbeddings] = word_embeddings
embeddings : StackedEmbeddings = StackedEmbeddings(
                                 embeddings=embedding_types)

from flair.models import SequenceTagger
tagger : SequenceTagger = SequenceTagger(hidden_size=256,
                                       embeddings=embeddings,
                                       tag_dictionary=tag_dictionary,
                                       tag_type=tag_type,
                                       use_crf=True)

from flair.trainers import ModelTrainer
trainer : ModelTrainer = ModelTrainer(tagger, Corpus)
trainer.train('model/resources/taggers/example-ner',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=10)