# Word Lists

## english-{1,2,5}K.txt

List of 1000/2000/5000 most common English words in the COCOA corpus. Source: https://www.wordfrequency.info

```sh
wget https://www.wordfrequency.info/samples/wordFrequency.xlsx
```

Prepare data:

```py
import pandas as pd
df = pd.read_excel('wordFrequency.xlsx', sheet_name='4 forms (219k)').set_index('rank')
df['word'].iloc[:1000].to_csv('english-1k.txt', index=False, header=False)
df['word'].iloc[:2000].to_csv('english-2k.txt', index=False, header=False)
df['word'].iloc[:5000].to_csv('english-5k.txt', index=False, header=False)
```
