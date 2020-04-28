from textgenrnn import textgenrnn  
import pandas as pd
import numpy as np

data = pd.read_csv('~/debate_transcripts_v3_2020-02-26.csv', encoding='cp1252')

joe = np.array(data[data['speaker'] == 'Joe Biden']['speech'].reset_index().drop('index',axis=1))
bernie = np.array(data[data['speaker'] == 'Bernie Sanders']['speech'].reset_index().drop('index',axis=1))

joe_text = open("joe.txt","w+")
for item in joe:
    joe_text.write(item[0])
    joe_text.write('\n')
joe_text.close()

bernie_text = open("bernie.txt","w+")
for item in bernie:
    bernie_text.write(item[0])
    bernie_text.write('\n')
bernie_text.close()


textgen = textgenrnn()
textgen.train_from_file('joe.txt', num_epochs=30)
textgen.generate()
