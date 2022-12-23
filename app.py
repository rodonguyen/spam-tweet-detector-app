import streamlit as st
import streamlit.components.v1 as components
import utils_lstm, utils_naive_bayes, pandas

model = 0

st.title('Is it a spam tweet? 🤖')

st.header('About:')
st.markdown('''This is a demo app for my Twitter Spam Detection app.  
You can read more about it in this [GitHub Repo](https://github.com/rodonguyen/showcase_AI_ML/tree/master/1.%20Twitter%20Spam%20Detection) (report and code included).  
This app uses LSTM and Naive Bayes. But SVM and Transformer model are not used due to dependencies problem. For accuracy, these models reached **95%** accuracy in the validation dataset.  
Author: [Rodo Nguyen](https://rodonguyen.dev/)''')

st.header('Instructions:')
st.markdown('''Note: the numbers placed in advance are *mean* values and \ndo not affect the final prediction.
Here are some sample tweets:
- Quality:  Big day.  #OLYMPIC #australia #skyline @ The Six https://www.instagram.com/p/BFgrA9gBZay/
- Spam:  Win now! Collect cash prize of 1000$ by joining this lucky draw: https://rodonguyen-spam-tweet-detector-app-app-ixl0vb.streamlit.app/'
- Quality:  I posted a new photo to Facebook http://fb.me/2Be7LiyuJ'  
- Spam:  Merkel Wants to Use Failed Iran Deal as Model to Solve North Korea Problem https://t.co/DAkGReM4js  
- Quality: I'm not a spam tweet because I don't carry harmful content''')

tweet_content = st.text_input("Input Tweet Content. \n Below is a Spam example:", 'Win now! Collect cash prize of 1000$ by joining this lucky draw: https://rodonguyen-spam-tweet-detector-app-app-ixl0vb.streamlit.app/') 

following = st.number_input("Input Following number of the author account", 0,10000000, 4743)
followers = st.number_input("Input Followers number of the author account", 0,10000000, 366142)
actions = st.number_input("Input Actions number - The total number of favourites, replies, and retweets associated with the tweet.", 0,1000000, 7232)
is_retweet = st.selectbox("Is it a Retweet?",[0,1], 0)


# Load Model
lstm_model = utils_lstm.load_LSTM_model()
nb_model = utils_naive_bayes.load_naive_bayes_model()

def predict():

  # Prepare data
  lstm_tweet_tensor, lstm_others_col_std = utils_lstm.preprocessing_input([[tweet_content, following, followers, actions, is_retweet]])
  nb_input = utils_naive_bayes.preprocess_input([tweet_content])
  # svm_input = utils_svm.preprocess_input([[tweet_content, following, followers, actions, is_retweet]])    

  prediction = pandas.DataFrame([
      ['LSTM', (lstm_model.predict([lstm_tweet_tensor, lstm_others_col_std]) >= 0.5)],
      ['Naive Bayes', nb_model.predict(nb_input)[0] == 'Spam']
    ], 
    columns=['model', 'prediction'])

  print(prediction)

  if prediction.loc[0,'prediction'] == True: 
    st.error('LSTM Model Prediction: It\'s a Spam Tweet 🤖')
  else: 
    st.success('LSTM Model Prediction: It\'s a Quality Tweet 👍') 

  if prediction.loc[0,'prediction'] == True: 
    st.error('Naive Bayes  Model Prediction: It\'s a Spam Tweet 🤖')
  else: 
    st.success('Naive Bayes Model Prediction: It\'s a Quality Tweet 👍') 

trigger = st.button('Predict', on_click=predict)
st.markdown('Please scroll up to see the prediction results (sorry for the inconvenience)!')