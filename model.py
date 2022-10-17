from flair.data import Sentence
import re
import json
import logging
# import nltk
import base64
import os

greeting_words = json.loads(open('model/greetings.txt', 'r').read())


def return_predictions(sent,model):
    
    all_ents =[]
    for sents in sent:
        if sents.strip(" ")==" ":
            all_ents.append({"entities":[],"labels":[],"text":"INPUT A VALID SENTENCE"})
        else:
            print (sents)
            sentence = Sentence(sents.strip(" "))

            # predict the tags
            model.predict(sentence)
            entities_json = sentence.to_dict(tag_type='ner')
            entities_json["length_in_terms_words"] = len(sents.strip(" ").split(' '))
            for word in greeting_words:
                ent_dict ={}
                m = re.search(word, sents[0])
                if m!=None:
                    ent_dict['start_pos'] = m.span()[0]
                    ent_dict['end_pos'] = m.span()[1]
                    ent_dict['text'] = m.group()
                    ent_dict['labels'] =['GREETINGS']
                    entities_json['entities'].append(ent_dict)
                else:
                    pass
            for i in range(len(entities_json['entities'])):
                entities_json['entities'][i]['labels'] = str(entities_json['entities'][i]['labels'])
            all_ents.append(entities_json)
    ent_all ={}
    ent_all["entities_list"]=all_ents
    ent_all["number_of_sentences"] = len(sent)
    return ent_all,200



    


