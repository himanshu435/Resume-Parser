import spacy
import random
import pickle

train_data = pickle.load(open('data.pkl','rb'))

nlp=spacy.load('nlp_model100it')
def train_model(train_data):
    if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner,last=True)

    #for _,annotation in train_data:
            #for ent in annotation['entities']:
                #ner.add_label(ent[2])

    other_pipes =[pipe for pipe in nlp.pipe_names if pipe!= 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(50):
            print("starting iteration " + str(itn))
            random.shuffle(train_data)
            losses={}
            index=0
            for text,annotations in train_data:
                try:
                    nlp.update([text],[annotations],drop=0.2,sgd=optimizer,losses=losses)
                except Exception as e:
                    pass

            print(losses)

#train_model(train_data)
#nlp.to_disk('nlp_model100it')
#nlp_model=spacy.load('nlp_model')
#doc=nlp_model(train_data[0][0])

doc=nlp(train_data[0][0])

skills=['c++','java']
score=0

for ent in doc.ents:
        print(f'{ent.label_.upper():{30}}-{ent.text}')

import fitz
fname='Smith Resume.pdf'
doc=fitz.open(fname)
text=""
for page in doc:
    text = text + str(page.getText())
tx = " ".join(text.split('\n'))
#print(tx)
#doc=nlp_model(tx)
doc=nlp(tx)
for ent in doc.ents:
        if ent.label_ == "Skills":
            news=ent.text.lower().split(",")
            for i,x in enumerate(news):
                news[i]=x.replace(" ","")
            for x in news:
                if x in skills:
                    score+=1
        print(f'{ent.label_.upper():{30}}-{ent.text}')

print('\ncurrent resume fulfills about ',(score/len(skills))*100,'% of what we want')
print('PASS For 2nd round? ',score>75)