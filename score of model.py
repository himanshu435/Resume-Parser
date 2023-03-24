from spacy.scorer import Scorer
from spacy.gold import GoldParse
import spacy

def evaluate():
    scorer = Scorer()
    examples = [
        ('Who is Java?', {
            'entities': [(7, 11, 'Skill')]
        }),
        ('I like C++ and C.', {
            'entities': [(7, 10, 'Skill'), (15, 17, 'Skill')]
        })
    ]
    ner_model = spacy.load('nlp_model20it')
    # resumeText = readFile(examples)
    for input_, annot in examples:
        doc_gold_text = ner_model.make_doc(input_)
        print(doc_gold_text)
        gold = GoldParse(doc_gold_text, entities=annot['entities'])

        pred_value = ner_model(input_)
        for ent in pred_value.ents:
            print(f'{ent.label_.upper():{30}}-{ent.text}')
        scorer.score(pred_value, gold)
        print('\n')
    print(scorer.scores)

evaluate()