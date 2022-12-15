import streamlit as st
import utils_lstm, utils_naive_bayes, pandas



model = 0

st.title('Is it a spam tweet? 🤖')

tweet_content = st.text_input("Tweet Content", 'Big day.  #WeTheNorth #yyz #thesix #sunset #skyline @ The Six https://www.instagram.com/p/BFgrA9gBZay/') 
following = st.number_input("Input Following Number of the author account", 0,10000000, 4743)
followers = st.number_input("Input Followers Number", 0,10000000, 366142)
actions = st.number_input("Input Actions Number", 0,1000000, 7232)
is_retweet = st.selectbox("Is it a Retweet",[0,1], 0)

def predict():

  # Load Model
  lstm_model = utils_lstm.load_LSTM_model()
  nb_model = utils_naive_bayes.load_naive_bayes_model()
  # svm_model = utils_svm.load_svm_model()

  # Prepare data
  lstm_tweet_tensor, lstm_others_col_std = utils_lstm.preprocessing_input([[tweet_content, following, followers, actions, is_retweet]])
  nb_input = utils_naive_bayes.preprocess_input([tweet_content])
  # svm_input = utils_svm.preprocess_input([[tweet_content, following, followers, actions, is_retweet]])    

  prediction = pandas.DataFrame(
  [
    ['LSTM', (lstm_model.predict([lstm_tweet_tensor, lstm_others_col_std]) >= 0.5)],
    ['Naive Bayes', nb_model.predict(nb_input)[0]],
    # ['SVM', svm_model.predict(svm_input)[0]]
  ], 
  columns=['model', 'prediction'])


  if prediction.iloc[0,1] == True: 
    st.error('It\'s a Spam Tweet 🤖')
  else: 
    st.success('It\'s a Quality Tweet 👍') 

trigger = st.button('Predict', on_click=predict)