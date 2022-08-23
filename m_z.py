import os
from os import path
import pandas as pd
url = os.getcwd()
input_file = 'm-z'
output_file = 'without_phaseshift'
file = os.listdir(path.join(url, input_file))
all_title = pd.DataFrame()
for f in file:
    real_url = path.join(url,input_file, f)
    write_url = path.join(url, output_file, f)
    df = pd.read_csv(real_url)
    try:
        df_without_phaseshift = df.drop('Phaseshift', axis=1)
        df_without_phaseshift.to_csv(write_url,index=False)
    except:
        df.to_csv(write_url,index=False)

print('end')
