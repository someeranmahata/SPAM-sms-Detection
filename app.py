import pickle
from flask import Flask, request, render_template
import nltk
from string import punctuation
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()    

model = pickle.load(open("model.pkl", "rb"))
vector = pickle.load(open("vectorizer.pkl", "rb"))


nltk.download("punkt")
nltk.download("stopwords")

stpword = set(stopwords.words('english'))

def convert(text):
    text = text.lower()
    x = nltk.word_tokenize(text)

    #keep alpha nuberic characters only
    res = []
    for let in x:
        if let.isalnum():
            res.append(let)

    res2 = res[:]
    res.clear()
    for let in res2:
        if let not in punctuation and let not in stpword:
            res.append(let)


    res2 = res[:]
    res.clear()

    for let in res2:
        res.append(ps.stem(let))


    return " ".join(res)

def vect(text):
    transformed = vector.transform([text])
    # return transformed.shape
    # print(transformed.shape)
    return transformed




st = convert("hello sommeran whats up !!! lets grab some drink, how you doing?")
print(st)
print(vect(st))
print(type(vector))

# print(model.n_features_in_)
# print(len(vector.get_feature_names_out()))




#random 5 spam and 5 ham message from stemming data
data = {'ringtoneking 84484': 1, 'Thanks for the Vote. Now sing along with the stars with Karaoke on your mobile. For a FREE link just reply with SING now.': 1, 'Want a new Video Phone? 750 anytime any network mins? Half price line rental free text for 3 months? Reply or call 08000930705 for free delivery': 1, 'Mobile Club: Choose any of the top quality items for your mobile. 7cfca1a': 1, 'Spook up your mob with a Halloween collection of a logo & pic message plus a free eerie tone, txt CARD SPOOK to 8007 zed 08701417012150p per logo/pic ': 1, 'Hey whats up? U sleeping all morning?': 0, "Sorry, got a late start, we're on the way": 0, 'I wish that I was with you. Holding you tightly. Making you see how important you are. How much you mean to me ... How much I need you ... In my life ...': 0, 'Thank You for calling.Forgot to say Happy Onam to you Sirji.I am fine here and remembered you when i met an insurance person.Meet You in Qatar Insha Allah.Rakhesh, ex Tata AIG who joined TISSCO,Tayseer.': 0, 'Japanese Proverb: If one Can do it, U too Can do it, If none Can do it,U must do it Indian version: If one Can do it, LET HIM DO it.. If none Can do it,LEAVE it!! And finally Kerala version: If one can do it, Stop him doing it.. If none can do it, Make a strike against it ...': 0}

# for key in data:
#     processed = convert(key)
#     X = vect(processed)

    
#     pred = model.predict(X)[0]

#     print(f"Actual: {data[key]}, Predicted: {pred}")



app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    status = None
    para = False
    
    if(request.method == 'POST'):
        text = request.form.get('user_input')
        para = True
        
        print(text)
        
        processed = convert(text)
        vt = vect(processed)

    
        pred = model.predict(vt)[0]
        status = pred
        
        
    return render_template(
        "index.html",
        para = para,
        status = status
        )

    
if __name__ == "__main__":
    app.run(debug=True)