from flask import Flask,render_template,request
import pickle
import requests
from bs4 import BeautifulSoup as soup
import numpy as np

app = Flask(__name__)
word_list = pickle.load(open('word_list.pkl','rb'))

headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36',
          'accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
           'accept-encoding': 'gzip, deflate, br',
           'accept-language': 'en-US,en;q=0.9,hi;q=0.8',
          }

clf=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    movie_id = request.form.get('message')
    if movie_id:
        sample_list, review_text, title = sentiments(movie_id,word_list)
        #print(sample_list[1])
        predict_list=[]
        for sample in sample_list:
            sampl = np.array(sample).reshape(1,5000)
            x = clf.predict(sampl)[0]
            predict_list.append(str(x))
        return_element=zip(predict_list,review_text)
    else:
        return_element=None
    return render_template('index.html', prediction=return_element,title=title)


def sentiments(movie_id,word_list):
    #print(movie_id)
    url = "https://www.imdb.com/title/{}/reviews?ref_=tt_ql_3".format(movie_id)
    page = requests.get(url, headers=headers)
    raw = soup(page.content, 'html.parser')
    reviews = raw.find_all('div', class_='imdb-user-review')
    title = raw.find_all('div', class_='subpage_title_block')[0].find('h3').text
    j = 0
    sample = []
    review_return=[]
    for item in reviews:
        review = item.find('div', class_='text show-more__control').text.lower().split()
        feature=[]
        for i in word_list:
            feature.append(review.count(i[0]))
        j += 1
        sample.append(feature)
        review_return.append(item.find('div',class_='text show-more__control').text[:300])
    return np.array(sample), review_return, title


if __name__ == "__main__":
    app.run(debug=True)
