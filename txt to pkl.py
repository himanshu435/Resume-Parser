import pickle
import ast
import json
# Read as bytes

list=[]
col=[]
with open('train_data.txt', 'r',encoding="utf8") as f:
    data =  f.readline()
    list.append(data)
    cnt = 1
    loc = data.find("entities")
    s = data[loc - 2:-2]
    dict = ast.literal_eval(s)
    col.append((data[2:loc - 5], dict))

    while data:
        data = f.readline()
        cnt += 1
        if data!="\n" and cnt<599:
            loc=data.find("entities")
            s=data[loc - 2:-2]
            dict=ast.literal_eval(s)
            col.append((data[2:loc - 5],dict))

# Save as pickle
with open('data.pkl', 'wb') as f:
    pickle.dump(col, f)

train_data = pickle.load(open('data.pkl', 'rb'))

