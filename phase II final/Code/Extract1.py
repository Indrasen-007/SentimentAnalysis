#import config
import tweepy
import time
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time
import csv
import sentiment_mod as s
class listener(StreamListener):
    def on_data(self,data):
        try:
          tweet =data.split(',"text":"')[1].split('","')[0]
        except:
          return True
   #Convert to lower case
        tweet = tweet.lower()
    #Convert www.* or https?://* to URL
        tweet = re.sub('((www\.[^\s]+)|(https?[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
        tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
        tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
        tweet = tweet.strip('\'"')
    
#end

        
      
        sentiment_value,confidence=s.sentiment(tweet)
        print (tweet,'\n',sentiment_value,confidence,'\n')
        output=open("twitter-out.txt","a")
        #output.write(tweet)
        #output.write("   ")
        output.write(sentiment_value)
        output.write('\n')
        output.close()

        if(confidence > .80 ):
            saveOutput=open("twitter-res.csv","a")
            c = csv.writer(saveOutput)
            c.writerow([tweet,sentiment_value,confidence])
            saveOutput.close()

        
        return True

    def on_error(self,status):
        print (status)

consumer_key = 'GNeX5TXMdfGovlox7N7OumfJ8'
consumer_secret = 'rRfLqQwb4i3RKlbSJnZU5nYxEfPMfoZ7pZ1O2QBOuO5y3SppHo'
access_token = '2993015678-bapD0BwOkMwIum5oBnqnc4Be4pHRfeHbkVfN0qC'
access_secret = 'jklrTiaWjvedDwsNNsfs1tnwwyjwcm8LnewhcUglGWSQc'


auth =OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)
twitterStream = Stream(auth,listener())
twitterStream.filter(track=["music"])
