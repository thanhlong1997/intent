import pandas as pd
from ai.ai_ultis.text_utils import get_normal_word,remove_stop_word
df=pd.read_excel("D:\\ITSOL\\detect_sentiment_\\storage\\data\\stopword.xlsx", sheetname="Sheet1", encoding="UTF-8")
list2 = []
for item in df.index:
    list2.append(df['stop'][item])
# print(ascii(list2))
# print(ascii(get_normal_word("trong năm công ty  việt nam và là  trong số các đó trong tỷ đồng usd với")))
for item in get_normal_word("trong năm công ty  việt nam và là  trong số các đó trong tỷ đồng usd với"):
    if item in list2:
        list2.remove(item)
print(list2)