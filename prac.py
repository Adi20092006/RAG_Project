import pandas as pd

my_dict={"ages":[1,2,3,4,5,6]}

df1=pd.DataFrame(my_dict)
df2=pd.DataFrame.from_records(my_dict)

print(df1)
print(df2)