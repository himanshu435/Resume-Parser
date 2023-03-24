import random
import pickle
import spacy

train_data = pickle.load(open('train_data.pkl','rb'))#loading dataset string and entity

#nlp=spacy.load('nlp_model100it')#loads premade model
nlp=spacy.blank('en')#fresh model
def train_model(train_data):
    if 'ner' not in nlp.pipe_names: #create ner module in the spacy nlp pipeline
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner,last=True)

    for _,annotation in train_data: #scan the training data & store the entity labels in the ner pipeline
            for ent in annotation['entities']:
                ner.add_label(ent[2])

    optimizer = nlp.begin_training()#ner training is initiated
    for itn in range(50):
        print("iteration-" + str(itn))
        random.shuffle(train_data)
        losses={}
        for text,annotations in train_data:#each iteration, we shuffle the training data,
            try:#nlp.update will update the weights of the 'nlp' named model based on input training data.
                nlp.update([text],[annotations],drop=0.2,sgd=optimizer,losses=losses)
            except Exception as e:
                pass

        print(losses)

train_model(train_data) #trains the model
#nlp.to_disk('nlp_model100it') #stores the model in directory for later use
#nlp_model=spacy.load('nlp_model') #loads already existing trained model
#doc=nlp_model(train_data[0][0]) #prediction on input string

doc=nlp(train_data[0][0])

for ent in doc.ents:
    print(ent.label_.upper(), '-', ent.text)


skills=['c++','java'] #the skills we want
score=0

#fitz along with pymupdf is used to get raw text of a pdf
import fitz
fname='Dheena.pdf'
doc=fitz.open(fname)

#the data is formatted to a single line string without \ns
text=""
for page in doc:
    text = text + str(page.getText())
tx = " ".join(text.split('\n'))

#print(tx)
#doc=nlp_model(tx)#prediction on test resume
doc=nlp(tx)
for ent in doc.ents:
        if ent.label_ == "Skills": # if category is skills and matches our requirements, ++
            news=ent.text.lower().split(",")
            for i,x in enumerate(news):
                news[i]=x.replace(" ","")
            for x in news:
                if x in skills:
                    score+=1
        print(ent.label_.upper(),'-',ent.text)#outputes resume

print('\ncurrent resume fulfills about ',(score/len(skills))*100,'% of what we want')
print('PASS For 2nd round? ',score>75)

