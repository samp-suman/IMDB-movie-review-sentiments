from flask import Flask,render_template,request
import pickle
import requests
from bs4 import BeautifulSoup as soup
import numpy as np

app = Flask(__name__)
word_list = pickle.load(open('word_list.pkl','rb'))
clf=pickle.load(open('model.pkl','rb'))
imdb_data=pickle.load(open('imdb_title.pkl','rb'))

headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36',
          'accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
           'accept-encoding': 'gzip, deflate, br',
           'accept-language': 'en-US,en;q=0.9,hi;q=0.8',
          }


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    return_element = None
    movieTitle = None
    movie_details = None
    details_shape=None
    flag=None

    movie_title = request.form.get('message')
    movie_id = request.form.get('title-id')

    if movie_title:
        movie_ids = getId(movie_title.lower())

        if movie_ids.shape[0]>1:
            movie_details=getDetails(movie_ids)
            details_shape=movie_details.shape[0]
        if movie_ids.shape[0]==1:
            movie_id=movie_ids[0]
        if movie_ids.shape[0] == 0:
            flag=-1
    if movie_id :
        sample_list, review_text, movieTitle = fetchData(movie_id)
        predict_list=getPrediction(sample_list)
        return_element=zip(predict_list,review_text)
    return render_template('index.html',prediction=return_element,title=movieTitle,details=movie_details,shape=details_shape,flag=flag)




def getId(title):
    movie_ids=imdb_data[(imdb_data['title'].str.contains(title))|(imdb_data['original_title'].str.contains(title))]['imdb_title_id'].values
    print(movie_ids)
    # if movie_ids.shape[0]==0:
    #     movie_ids=None
    return movie_ids


def getPrediction(samples):
    predict_list = []
    for sample in samples:
        sampl = np.array(sample).reshape(1, 5000)
        x = clf.predict(sampl)[0]
        predict_list.append(str(x))
    return predict_list



def fetchData(imdbId):
    url = "https://www.imdb.com/title/{}/reviews?ref_=tt_ql_3".format(imdbId)
    page = requests.get(url, headers=headers)
    raw = soup(page.content, 'html.parser')
    reviews = raw.find_all('div', class_='imdb-user-review')
    movieTitle = raw.find_all('div', class_='subpage_title_block')[0].find('h3').text
    j = 0
    sample = []
    review_return = []
    for item in reviews:
        review = item.find('div', class_='text show-more__control').text.lower().split()
        feature = []
        for i in word_list:
            feature.append(review.count(i[0]))
        j += 1
        sample.append(feature)
        review_return.append(item.find('div', class_='text show-more__control').text[:300])

    return np.array(sample), review_return, movieTitle

def getDetails(ids):
    details=imdb_data[imdb_data['imdb_title_id'].isin(ids)][['imdb_title_id','title','year']].values
    return details

if __name__ == "__main__":
    app.run(debug=True)