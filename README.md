# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
```
import numpy as np
import pandas as pd
data=pd.read_csv("/content/sample_data/california_housing_test.csv")
data
```

<img width="1352" height="511" alt="Screenshot 2025-09-10 114129" src="https://github.com/user-attachments/assets/3f7b28f9-2dbe-4060-a733-f90c9e3de77f" />

```
data.head(5)
```
<img width="1302" height="250" alt="Screenshot 2025-09-10 114427" src="https://github.com/user-attachments/assets/7fca7169-159c-42a2-9fba-1427847b8a65" />

```
data.tail(5)
```

<img width="1369" height="251" alt="Screenshot 2025-09-10 114533" src="https://github.com/user-attachments/assets/acbf1323-ad16-46e5-b0b3-056d439be933" />

```
data.isnull()
```

<img width="1345" height="517" alt="Screenshot 2025-09-10 114638" src="https://github.com/user-attachments/assets/be22af72-8f45-4f25-968c-21c20e2887cc" />

```
data.notnull()
```

<img width="1395" height="538" alt="Screenshot 2025-09-10 114933" src="https://github.com/user-attachments/assets/74b1e95d-ed7e-49b4-bbde-68e3ed21c683" />

```
data.isnull().sum()
```

<img width="328" height="451" alt="Screenshot 2025-09-10 115112" src="https://github.com/user-attachments/assets/6563be67-c280-46fd-b611-d29e137b511e" />

```
data.isnull().any()
```

<img width="401" height="457" alt="Screenshot 2025-09-10 115143" src="https://github.com/user-attachments/assets/f25af698-5c5c-47d6-b430-1bb464ee641d" />

```            
data.dropna(axis=1)
```

<img width="1353" height="505" alt="Screenshot 2025-09-10 115300" src="https://github.com/user-attachments/assets/e31e34a3-08c8-40c8-a60e-ef267524e206" />

```
data.dropna(axis=0)
```

<img width="1512" height="518" alt="Screenshot 2025-09-10 115422" src="https://github.com/user-attachments/assets/f7189435-cb66-4fe5-b25d-5dad9fa73f53" />

```
data.fillna(0)
```

<img width="1375" height="510" alt="Screenshot 2025-09-10 115645" src="https://github.com/user-attachments/assets/c45f3c82-b6fb-40e8-aa0a-1672422692f7" />

```
data.fillna(method="ffill")
```

<img width="1752" height="561" alt="Screenshot 2025-09-10 115742" src="https://github.com/user-attachments/assets/9aedfb73-a952-4c5c-bb3a-cd9fdefa4dcb" />

```
data.bfill()
```

<img width="1356" height="536" alt="Screenshot 2025-09-10 115848" src="https://github.com/user-attachments/assets/c2bf9c94-dc03-4a3c-8b03-d5a09aa19db7" />

```
data.fillna({'REGNO':0, 'NAME':'SURYA'})
```

<img width="1375" height="506" alt="Screenshot 2025-09-10 115943" src="https://github.com/user-attachments/assets/69932353-02a5-47b1-95fc-0c97d0b21c11" />

```
ir=pd.read_csv("/content/iris.csv")
ir
```

<img width="780" height="529" alt="Screenshot 2025-09-10 120051" src="https://github.com/user-attachments/assets/466049b2-a932-4926-bd16-c877d4450b96" />

```
ir.describe()
```

<img width="649" height="365" alt="Screenshot 2025-09-10 120140" src="https://github.com/user-attachments/assets/877e241b-6006-4231-8b62-b668414a996b" />

```
import seaborn as sns
sns.boxplot(x="sepal_width",data=ir)
```

<img width="736" height="584" alt="Screenshot 2025-09-10 120309" src="https://github.com/user-attachments/assets/36238c3d-c24c-454c-9422-02fc58a0994a" />

```
q1=ir.sepal_width.quantile(0.25)
q3=ir.sepal_width.quantile(0.75)
iqr=q3-q1
print(iqr)
```

<img width="145" height="39" alt="Screenshot 2025-09-10 120403" src="https://github.com/user-attachments/assets/6d187baa-6830-4102-8665-a70f6d55b0fa" />

```
rid=ir[((ir.sepal_width<(q1-1.5*iqr))|(ir.sepal_width>(q3+1.5*iqr)))]
rid['sepal_width']
```

<img width="252" height="270" alt="Screenshot 2025-09-10 120504" src="https://github.com/user-attachments/assets/8c02ab38-5948-4f23-b827-078853099efb" />

```
rid=ir[~((ir.sepal_width<(q1-1.5*iqr))|(ir.sepal_width>(q3+1.5*iqr)))]
rid
```

<img width="694" height="526" alt="Screenshot 2025-09-10 120608" src="https://github.com/user-attachments/assets/2465486a-3f4a-464a-b604-c74f76939fb5" />

```
rid=ir[((ir.sepal_width>(q1-1.5*iqr))&(ir.sepal_width<(q3+1.5*iqr)))]
rid['sepal_width']
```

<img width="286" height="577" alt="Screenshot 2025-09-10 120719" src="https://github.com/user-attachments/assets/8bb781f2-7d4e-4529-94e7-0b47f740dea4" />

```
import numpy as np
import scipy.stats as stats
z=np.abs(stats.zscore(ir.sepal_width))
z
```

<img width="725" height="671" alt="Screenshot 2025-09-10 120818" src="https://github.com/user-attachments/assets/67e92b24-653b-46a8-979d-2eee53951e7c" />







# Result
```
Thus the program was sucessesfully executed
```
