
# More Pandas

![more_pandas](https://media.giphy.com/media/H0Qi5W2KzU5UI/giphy.gif)

### Scenario
You have decided that you want to start your own animal shelter, but you want to get an idea of what that will entail and to get more information about planning. In this lecture, we'll look at a real data set collected by Austin Animal Center.  The code below will return the last 1000 animal outcomes that have occurred.  We will use our pandas skills from the last lecture and learn some new ones in order to explore these data further.




#### Our goals in this notebook are to be able to: <br/>

- Apply and use `.map()`, `apply()`, and `.applymap()` from the Pandas library
- Introduce lambda functions and use them in coordination with above functions
- Explain what a groupby object is and split a DataFrame using `.groupby()`


#### Getting started

Let's take a moment to download and to examine the [Austin Animal Center data set](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes/9t4d-g238/data). 

Let's take a look at the data:


```python
import numpy as np
import pandas as pd
import requests

%load_ext autoreload
%autoreload 2

from src.student_caller import one_random_student, three_random_students
from src.student_list import student_list
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload



```python
url = 'https://data.austintexas.gov/resource/9t4d-g238.json'
response = requests.get(url)
animals = pd.DataFrame(response.json())
animals.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>animal_id</th>
      <th>name</th>
      <th>datetime</th>
      <th>monthyear</th>
      <th>date_of_birth</th>
      <th>outcome_type</th>
      <th>animal_type</th>
      <th>sex_upon_outcome</th>
      <th>age_upon_outcome</th>
      <th>breed</th>
      <th>color</th>
      <th>outcome_subtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A821019</td>
      <td>Spot</td>
      <td>2020-12-08T12:37:00.000</td>
      <td>2020-12-08T12:37:00.000</td>
      <td>2017-04-03T00:00:00.000</td>
      <td>Adoption</td>
      <td>Dog</td>
      <td>Neutered Male</td>
      <td>3 years</td>
      <td>Pit Bull</td>
      <td>White/Black</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A824438</td>
      <td>*Rose</td>
      <td>2020-12-08T12:27:00.000</td>
      <td>2020-12-08T12:27:00.000</td>
      <td>2011-11-27T00:00:00.000</td>
      <td>Adoption</td>
      <td>Dog</td>
      <td>Spayed Female</td>
      <td>9 years</td>
      <td>German Shepherd</td>
      <td>Tan/Black</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A825587</td>
      <td>*Ludwig</td>
      <td>2020-12-08T12:22:00.000</td>
      <td>2020-12-08T12:22:00.000</td>
      <td>2011-11-06T00:00:00.000</td>
      <td>Adoption</td>
      <td>Cat</td>
      <td>Neutered Male</td>
      <td>9 years</td>
      <td>Domestic Medium Hair</td>
      <td>Cream Tabby</td>
      <td>Foster</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A819626</td>
      <td>NaN</td>
      <td>2020-12-08T11:53:00.000</td>
      <td>2020-12-08T11:53:00.000</td>
      <td>2020-06-25T00:00:00.000</td>
      <td>Adoption</td>
      <td>Cat</td>
      <td>Neutered Male</td>
      <td>5 months</td>
      <td>Domestic Shorthair</td>
      <td>White/Black</td>
      <td>Foster</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A819624</td>
      <td>NaN</td>
      <td>2020-12-08T11:52:00.000</td>
      <td>2020-12-08T11:52:00.000</td>
      <td>2020-06-25T00:00:00.000</td>
      <td>Adoption</td>
      <td>Cat</td>
      <td>Neutered Male</td>
      <td>5 months</td>
      <td>Domestic Shorthair</td>
      <td>Black</td>
      <td>Foster</td>
    </tr>
  </tbody>
</table>
</div>




```python
animals.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 12 columns):
     #   Column            Non-Null Count  Dtype 
    ---  ------            --------------  ----- 
     0   animal_id         1000 non-null   object
     1   name              655 non-null    object
     2   datetime          1000 non-null   object
     3   monthyear         1000 non-null   object
     4   date_of_birth     1000 non-null   object
     5   outcome_type      900 non-null    object
     6   animal_type       1000 non-null   object
     7   sex_upon_outcome  1000 non-null   object
     8   age_upon_outcome  1000 non-null   object
     9   breed             1000 non-null   object
     10  color             1000 non-null   object
     11  outcome_subtype   482 non-null    object
    dtypes: object(12)
    memory usage: 93.9+ KB


One way to become familiar with your data is to start asking questions. In your EDA notebooks, **markdown** will be especially helpful in tracking these questions and your methods of answering the questions.  

For example, a simple first question we might ask, after being presented with the above dataset, would be:

## What is the most commonly adopted animal type in the dataset?

We can then begin thinking about what parts of the DataFrame we need to answer the question.

    What features do we need?
     - 
    What type of logic and calculation do we perform?
     -  
    What type of visualization would help us answer the question?
     -


```python
# Your code here

```

Questions lead to other questions. For the above example, the visualization begs the question, what Other animals are being adopted?

To find out, we need to know where the type of animal for Other is encoded.   
    
    What features do we need to answer what the most commonly adopted type of animal within the Other category is?


```python
# Your code here
```

# Quick Exploration


```python
# Use info to check for na's, datatypes, and shape
```


```python
# Use describe to gain a bit more detail about certain features. 
```


```python
# Use value counts to check a categorical feature's distribution
```


```python
# Use isna() for a more legible output (than .info()) of na distributions of our dataset.
```

Use fillna to fill animals with no name to 'unnamed'


```python
three_random_students(student_list)
```

    ['Christa' 'Raf' 'DarigaSilverman']



```python

```


```python
animals.fillna('no_type_or_subtype', inplace=True)
```


```python
animals.isna().sum()
```




    animal_id           0
    name                0
    datetime            0
    monthyear           0
    date_of_birth       0
    outcome_type        0
    animal_type         0
    sex_upon_outcome    0
    age_upon_outcome    0
    breed               0
    color               0
    outcome_subtype     0
    dtype: int64



### Applying and using map and applymap from the Pandas library

The built in **map** operator takes a function and applies it to every element of an iterable


```python
def divisible_by_5(number):
    
    '''
    Parameter: an integer
    return numbers divisible by five
    '''
    
    if number % 5 == 0:
        return True
    else:
        return False

numbers = [17,29,30045, 125]

list(map(divisible_by_5, numbers))

```




    [False, False, True, True]



The Pandas library has several similar methods associated with Dataframes and Series. Let's explore them.

# DataFrame.applymap(), Series.map()  Series.apply()

## DataFrame.applymap()
The ```.applymap()``` method takes a function as input that it will then apply to every entry in the dataframe.


```python
animals.applymap(type)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>animal_id</th>
      <th>name</th>
      <th>datetime</th>
      <th>monthyear</th>
      <th>date_of_birth</th>
      <th>outcome_type</th>
      <th>animal_type</th>
      <th>sex_upon_outcome</th>
      <th>age_upon_outcome</th>
      <th>breed</th>
      <th>color</th>
      <th>outcome_subtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
    </tr>
    <tr>
      <th>2</th>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
    </tr>
    <tr>
      <th>3</th>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
    </tr>
    <tr>
      <th>4</th>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
    </tr>
    <tr>
      <th>996</th>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
    </tr>
    <tr>
      <th>997</th>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
    </tr>
    <tr>
      <th>998</th>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
    </tr>
    <tr>
      <th>999</th>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
      <td>&lt;class 'str'&gt;</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 12 columns</p>
</div>



# Series.map()

The **.map()** method takes a function as input that it will then apply to every entry in the Series.

Let's map a ternary class set to consolodate sex_upon_outcome to male, female, or unknown   

First, explore the unique values:


```python
animals['sex_upon_outcome'].unique()
```




    array(['Neutered Male', 'Spayed Female', 'Intact Female', 'Intact Male',
           'Unknown', 'NULL'], dtype=object)




```python
# we could also use np.unique() with the return_counts parameter

np.unique(animals['sex_upon_outcome'], return_counts=True)

```




    (array(['Intact Female', 'Intact Male', 'NULL', 'Neutered Male',
            'Spayed Female', 'Unknown'], dtype=object),
     array([140, 171,   3, 352, 284,  50]))




```python
# Your code here
```

# Series.apply()

Series.apply() is similar to .map, except it only takes a function as a parameter, whereas .map can take a list, dictionary, or function.  .apply() is meant for more complex functions.

Now let's define a custom function that converts all ages upon outcome to days, and create a new column with .apply():


```python
# First, checkout what happens when we split on a space

list(animals['age_upon_outcome'].str.split(' '))
```




    [['3', 'years'],
     ['9', 'years'],
     ['9', 'years'],
     ['5', 'months'],
     ['5', 'months'],
     ['5', 'months'],
     ['1', 'year'],
     ['5', 'years'],
     ['2', 'years'],
     ['1', 'year'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['7', 'years'],
     ['2', 'years'],
     ['5', 'years'],
     ['3', 'months'],
     ['2', 'years'],
     ['NULL'],
     ['5', 'years'],
     ['3', 'months'],
     ['1', 'year'],
     ['1', 'year'],
     ['6', 'months'],
     ['1', 'year'],
     ['8', 'months'],
     ['4', 'months'],
     ['6', 'months'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['5', 'years'],
     ['4', 'months'],
     ['11', 'years'],
     ['2', 'months'],
     ['5', 'months'],
     ['2', 'months'],
     ['12', 'years'],
     ['1', 'year'],
     ['6', 'months'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'years'],
     ['1', 'month'],
     ['3', 'years'],
     ['3', 'months'],
     ['8', 'years'],
     ['6', 'years'],
     ['8', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['3', 'months'],
     ['3', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['6', 'months'],
     ['6', 'months'],
     ['10', 'years'],
     ['NULL'],
     ['5', 'months'],
     ['4', 'months'],
     ['3', 'months'],
     ['2', 'months'],
     ['3', 'months'],
     ['6', 'months'],
     ['2', 'years'],
     ['2', 'years'],
     ['3', 'months'],
     ['10', 'months'],
     ['1', 'year'],
     ['NULL'],
     ['12', 'years'],
     ['2', 'months'],
     ['5', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['3', 'months'],
     ['1', 'year'],
     ['2', 'years'],
     ['2', 'years'],
     ['3', 'months'],
     ['4', 'months'],
     ['1', 'year'],
     ['2', 'years'],
     ['2', 'years'],
     ['7', 'months'],
     ['2', 'years'],
     ['6', 'months'],
     ['2', 'years'],
     ['2', 'months'],
     ['2', 'months'],
     ['4', 'months'],
     ['3', 'months'],
     ['3', 'months'],
     ['4', 'months'],
     ['4', 'months'],
     ['NULL'],
     ['1', 'year'],
     ['2', 'years'],
     ['8', 'years'],
     ['10', 'months'],
     ['1', 'year'],
     ['3', 'months'],
     ['1', 'year'],
     ['0', 'years'],
     ['2', 'years'],
     ['1', 'year'],
     ['5', 'months'],
     ['8', 'months'],
     ['1', 'month'],
     ['NULL'],
     ['2', 'months'],
     ['3', 'months'],
     ['2', 'years'],
     ['2', 'months'],
     ['3', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['1', 'year'],
     ['2', 'years'],
     ['6', 'months'],
     ['2', 'years'],
     ['6', 'years'],
     ['4', 'years'],
     ['2', 'years'],
     ['1', 'year'],
     ['3', 'months'],
     ['7', 'years'],
     ['6', 'months'],
     ['6', 'months'],
     ['1', 'year'],
     ['2', 'months'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'months'],
     ['2', 'years'],
     ['1', 'month'],
     ['1', 'month'],
     ['1', 'month'],
     ['1', 'month'],
     ['2', 'years'],
     ['1', 'month'],
     ['1', 'month'],
     ['1', 'month'],
     ['6', 'years'],
     ['2', 'years'],
     ['3', 'years'],
     ['2', 'years'],
     ['1', 'year'],
     ['2', 'months'],
     ['2', 'years'],
     ['1', 'month'],
     ['5', 'years'],
     ['1', 'year'],
     ['3', 'months'],
     ['2', 'years'],
     ['10', 'years'],
     ['2', 'months'],
     ['2', 'months'],
     ['7', 'years'],
     ['10', 'months'],
     ['2', 'years'],
     ['3', 'years'],
     ['1', 'year'],
     ['1', 'year'],
     ['2', 'months'],
     ['4', 'months'],
     ['4', 'months'],
     ['2', 'months'],
     ['3', 'months'],
     ['6', 'months'],
     ['6', 'months'],
     ['3', 'months'],
     ['8', 'months'],
     ['2', 'years'],
     ['3', 'years'],
     ['8', 'months'],
     ['3', 'days'],
     ['3', 'days'],
     ['3', 'days'],
     ['3', 'days'],
     ['7', 'years'],
     ['7', 'years'],
     ['3', 'years'],
     ['1', 'month'],
     ['1', 'month'],
     ['1', 'month'],
     ['1', 'month'],
     ['1', 'month'],
     ['1', 'month'],
     ['2', 'months'],
     ['1', 'month'],
     ['1', 'year'],
     ['10', 'years'],
     ['1', 'year'],
     ['2', 'years'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['1', 'year'],
     ['2', 'months'],
     ['2', 'years'],
     ['4', 'years'],
     ['2', 'weeks'],
     ['2', 'months'],
     ['2', 'months'],
     ['10', 'months'],
     ['2', 'months'],
     ['2', 'years'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['3', 'years'],
     ['10', 'years'],
     ['1', 'year'],
     ['3', 'years'],
     ['1', 'year'],
     ['2', 'months'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'weeks'],
     ['2', 'weeks'],
     ['2', 'weeks'],
     ['4', 'years'],
     ['2', 'years'],
     ['6', 'months'],
     ['10', 'months'],
     ['2', 'years'],
     ['5', 'months'],
     ['NULL'],
     ['1', 'year'],
     ['2', 'years'],
     ['1', 'year'],
     ['4', 'years'],
     ['2', 'years'],
     ['6', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['17', 'years'],
     ['10', 'years'],
     ['4', 'months'],
     ['4', 'months'],
     ['5', 'months'],
     ['5', 'months'],
     ['2', 'years'],
     ['1', 'year'],
     ['2', 'years'],
     ['2', 'years'],
     ['5', 'years'],
     ['2', 'years'],
     ['3', 'years'],
     ['3', 'years'],
     ['2', 'years'],
     ['4', 'months'],
     ['4', 'months'],
     ['3', 'years'],
     ['3', 'months'],
     ['7', 'months'],
     ['7', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['3', 'months'],
     ['2', 'months'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['3', 'years'],
     ['1', 'year'],
     ['2', 'months'],
     ['NULL'],
     ['2', 'months'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'months'],
     ['2', 'years'],
     ['3', 'months'],
     ['2', 'years'],
     ['2', 'years'],
     ['10', 'months'],
     ['1', 'year'],
     ['2', 'years'],
     ['5', 'years'],
     ['3', 'months'],
     ['2', 'months'],
     ['2', 'years'],
     ['1', 'year'],
     ['3', 'weeks'],
     ['1', 'year'],
     ['1', 'year'],
     ['1', 'year'],
     ['2', 'years'],
     ['3', 'years'],
     ['2', 'years'],
     ['1', 'year'],
     ['NULL'],
     ['1', 'month'],
     ['1', 'month'],
     ['2', 'months'],
     ['3', 'years'],
     ['5', 'months'],
     ['8', 'months'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['6', 'months'],
     ['1', 'year'],
     ['6', 'months'],
     ['1', 'year'],
     ['5', 'months'],
     ['4', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['3', 'years'],
     ['2', 'years'],
     ['2', 'months'],
     ['4', 'months'],
     ['8', 'months'],
     ['8', 'months'],
     ['2', 'years'],
     ['4', 'months'],
     ['6', 'months'],
     ['1', 'year'],
     ['6', 'months'],
     ['1', 'year'],
     ['2', 'years'],
     ['4', 'months'],
     ['1', 'month'],
     ['7', 'months'],
     ['10', 'years'],
     ['1', 'month'],
     ['1', 'month'],
     ['1', 'year'],
     ['1', 'year'],
     ['5', 'weeks'],
     ['2', 'months'],
     ['2', 'months'],
     ['6', 'years'],
     ['1', 'year'],
     ['2', 'months'],
     ['2', 'years'],
     ['9', 'years'],
     ['8', 'months'],
     ['1', 'month'],
     ['5', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['3', 'months'],
     ['1', 'year'],
     ['2', 'months'],
     ['5', 'years'],
     ['1', 'year'],
     ['8', 'months'],
     ['6', 'months'],
     ['3', 'months'],
     ['2', 'months'],
     ['2', 'years'],
     ['10', 'months'],
     ['1', 'year'],
     ['2', 'years'],
     ['10', 'years'],
     ['2', 'years'],
     ['3', 'months'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['8', 'years'],
     ['8', 'years'],
     ['5', 'months'],
     ['2', 'months'],
     ['7', 'months'],
     ['2', 'years'],
     ['2', 'years'],
     ['9', 'months'],
     ['4', 'months'],
     ['2', 'months'],
     ['8', 'months'],
     ['6', 'months'],
     ['4', 'months'],
     ['2', 'years'],
     ['2', 'months'],
     ['2', 'months'],
     ['8', 'months'],
     ['4', 'months'],
     ['5', 'months'],
     ['1', 'year'],
     ['3', 'years'],
     ['2', 'months'],
     ['2', 'months'],
     ['5', 'months'],
     ['5', 'months'],
     ['3', 'months'],
     ['1', 'year'],
     ['3', 'months'],
     ['3', 'weeks'],
     ['3', 'weeks'],
     ['3', 'weeks'],
     ['3', 'weeks'],
     ['3', 'weeks'],
     ['3', 'weeks'],
     ['3', 'weeks'],
     ['3', 'weeks'],
     ['10', 'months'],
     ['3', 'weeks'],
     ['8', 'years'],
     ['8', 'months'],
     ['8', 'years'],
     ['15', 'years'],
     ['6', 'years'],
     ['2', 'years'],
     ['6', 'months'],
     ['2', 'years'],
     ['2', 'months'],
     ['2', 'years'],
     ['3', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['3', 'months'],
     ['3', 'months'],
     ['2', 'years'],
     ['2', 'months'],
     ['1', 'year'],
     ['4', 'months'],
     ['4', 'months'],
     ['5', 'months'],
     ['5', 'months'],
     ['1', 'year'],
     ['1', 'year'],
     ['2', 'months'],
     ['2', 'years'],
     ['2', 'months'],
     ['10', 'months'],
     ['2', 'years'],
     ['2', 'months'],
     ['2', 'months'],
     ['6', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['1', 'year'],
     ['3', 'years'],
     ['1', 'year'],
     ['3', 'years'],
     ['2', 'years'],
     ['1', 'year'],
     ['17', 'years'],
     ['2', 'years'],
     ['2', 'weeks'],
     ['2', 'years'],
     ['5', 'weeks'],
     ['2', 'years'],
     ['5', 'years'],
     ['2', 'months'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['1', 'year'],
     ['2', 'weeks'],
     ['3', 'weeks'],
     ['2', 'weeks'],
     ['2', 'years'],
     ['3', 'years'],
     ['2', 'years'],
     ['6', 'years'],
     ['8', 'years'],
     ['1', 'year'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'months'],
     ['3', 'years'],
     ['6', 'months'],
     ['2', 'years'],
     ['9', 'months'],
     ['2', 'years'],
     ['2', 'years'],
     ['10', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['3', 'months'],
     ['3', 'years'],
     ['1', 'year'],
     ['6', 'years'],
     ['3', 'months'],
     ['2', 'years'],
     ['3', 'months'],
     ['4', 'months'],
     ['1', 'month'],
     ['1', 'month'],
     ['6', 'years'],
     ['2', 'years'],
     ['6', 'months'],
     ['1', 'year'],
     ['2', 'years'],
     ['7', 'years'],
     ['10', 'years'],
     ['3', 'months'],
     ['2', 'years'],
     ['3', 'months'],
     ['2', 'months'],
     ['8', 'months'],
     ['3', 'months'],
     ['2', 'months'],
     ['2', 'years'],
     ['3', 'months'],
     ['11', 'months'],
     ['7', 'years'],
     ['12', 'years'],
     ['2', 'months'],
     ['7', 'months'],
     ['7', 'months'],
     ['9', 'months'],
     ['3', 'months'],
     ['2', 'years'],
     ['2', 'months'],
     ['2', 'months'],
     ['6', 'months'],
     ['2', 'years'],
     ['4', 'years'],
     ['7', 'months'],
     ['3', 'months'],
     ['5', 'years'],
     ['2', 'years'],
     ['4', 'years'],
     ['2', 'years'],
     ['4', 'weeks'],
     ['2', 'months'],
     ['3', 'months'],
     ['3', 'months'],
     ['2', 'years'],
     ['8', 'years'],
     ['10', 'months'],
     ['1', 'month'],
     ['2', 'years'],
     ['1', 'year'],
     ['2', 'years'],
     ['11', 'years'],
     ['8', 'years'],
     ['1', 'year'],
     ['NULL'],
     ['3', 'years'],
     ['8', 'years'],
     ['4', 'weeks'],
     ['3', 'months'],
     ['1', 'year'],
     ['8', 'months'],
     ['2', 'years'],
     ['3', 'years'],
     ['3', 'months'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['5', 'years'],
     ['2', 'years'],
     ['3', 'years'],
     ['5', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['15', 'years'],
     ['4', 'months'],
     ['3', 'months'],
     ['7', 'years'],
     ['3', 'months'],
     ['6', 'months'],
     ['1', 'year'],
     ['1', 'year'],
     ['10', 'months'],
     ['2', 'months'],
     ['2', 'years'],
     ['6', 'months'],
     ['2', 'years'],
     ['2', 'years'],
     ['4', 'years'],
     ['1', 'year'],
     ['2', 'years'],
     ['1', 'year'],
     ['2', 'years'],
     ['2', 'years'],
     ['3', 'months'],
     ['3', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['3', 'months'],
     ['2', 'years'],
     ['1', 'month'],
     ['1', 'month'],
     ['1', 'month'],
     ['1', 'month'],
     ['1', 'year'],
     ['2', 'years'],
     ['5', 'months'],
     ['1', 'year'],
     ['3', 'months'],
     ['2', 'years'],
     ['1', 'year'],
     ['5', 'months'],
     ['2', 'months'],
     ['1', 'year'],
     ['9', 'months'],
     ['9', 'months'],
     ['2', 'years'],
     ['1', 'year'],
     ['1', 'year'],
     ['6', 'years'],
     ['1', 'year'],
     ['3', 'months'],
     ['3', 'months'],
     ['NULL'],
     ['3', 'years'],
     ['10', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['1', 'year'],
     ['5', 'years'],
     ['2', 'months'],
     ['16', 'years'],
     ['2', 'years'],
     ['1', 'year'],
     ['4', 'years'],
     ['1', 'year'],
     ['1', 'year'],
     ['2', 'years'],
     ['2', 'years'],
     ['1', 'year'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'months'],
     ['9', 'years'],
     ['4', 'months'],
     ['3', 'months'],
     ['3', 'years'],
     ['2', 'months'],
     ['1', 'year'],
     ['3', 'years'],
     ['2', 'years'],
     ['3', 'years'],
     ['2', 'years'],
     ['7', 'years'],
     ['4', 'years'],
     ['3', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['1', 'year'],
     ['1', 'year'],
     ['1', 'year'],
     ['3', 'months'],
     ['2', 'months'],
     ['3', 'months'],
     ['3', 'years'],
     ['4', 'years'],
     ['2', 'months'],
     ['10', 'years'],
     ['2', 'years'],
     ['6', 'months'],
     ['3', 'years'],
     ['4', 'months'],
     ['4', 'months'],
     ['NULL'],
     ['3', 'years'],
     ['11', 'months'],
     ['3', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'months'],
     ['2', 'years'],
     ['1', 'year'],
     ['1', 'year'],
     ['1', 'year'],
     ['2', 'years'],
     ['4', 'years'],
     ['11', 'months'],
     ['1', 'year'],
     ['2', 'years'],
     ['3', 'weeks'],
     ['7', 'months'],
     ['2', 'years'],
     ['2', 'years'],
     ['1', 'month'],
     ['2', 'months'],
     ['7', 'months'],
     ['3', 'years'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['7', 'years'],
     ['1', 'year'],
     ['2', 'years'],
     ['2', 'months'],
     ['2', 'years'],
     ['1', 'year'],
     ['10', 'months'],
     ['2', 'years'],
     ['NULL'],
     ['1', 'year'],
     ['2', 'years'],
     ['2', 'years'],
     ['6', 'years'],
     ['8', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['3', 'years'],
     ['1', 'year'],
     ['10', 'months'],
     ['5', 'years'],
     ['3', 'months'],
     ['10', 'years'],
     ['10', 'years'],
     ['2', 'months'],
     ['2', 'months'],
     ['4', 'months'],
     ['10', 'years'],
     ['2', 'years'],
     ['10', 'years'],
     ['2', 'years'],
     ['1', 'year'],
     ['3', 'years'],
     ['NULL'],
     ['1', 'year'],
     ['3', 'months'],
     ['3', 'months'],
     ['3', 'months'],
     ['10', 'months'],
     ['2', 'months'],
     ['1', 'year'],
     ['3', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'years'],
     ['6', 'years'],
     ['2', 'years'],
     ['6', 'years'],
     ['2', 'months'],
     ['2', 'years'],
     ['4', 'months'],
     ['4', 'months'],
     ['3', 'weeks'],
     ['3', 'weeks'],
     ['5', 'years'],
     ['1', 'year'],
     ['2', 'months'],
     ['2', 'months'],
     ['4', 'months'],
     ['2', 'months'],
     ['10', 'months'],
     ['4', 'months'],
     ['7', 'months'],
     ['1', 'year'],
     ['1', 'year'],
     ['2', 'years'],
     ['5', 'years'],
     ['5', 'years'],
     ['2', 'months'],
     ['3', 'years'],
     ['1', 'year'],
     ['17', 'years'],
     ['4', 'years'],
     ['3', 'months'],
     ['3', 'months'],
     ['1', 'year'],
     ['2', 'months'],
     ['3', 'months'],
     ['3', 'months'],
     ['15', 'years'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'years'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'years'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['1', 'year'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['NULL'],
     ['1', 'month'],
     ['1', 'year'],
     ['2', 'years'],
     ['2', 'years'],
     ['3', 'months'],
     ['2', 'years'],
     ['3', 'months'],
     ['2', 'months'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['3', 'years'],
     ['2', 'years'],
     ['1', 'year'],
     ['2', 'years'],
     ['3', 'weeks'],
     ['5', 'months'],
     ['1', 'year'],
     ['3', 'weeks'],
     ['3', 'weeks'],
     ['3', 'weeks'],
     ['3', 'weeks'],
     ['3', 'weeks'],
     ['3', 'weeks'],
     ['3', 'weeks'],
     ['3', 'weeks'],
     ['1', 'year'],
     ['10', 'years'],
     ['2', 'months'],
     ['1', 'year'],
     ['1', 'year'],
     ['4', 'years'],
     ['1', 'year'],
     ['3', 'weeks'],
     ['2', 'years'],
     ['3', 'years'],
     ['2', 'years'],
     ['15', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['2', 'months'],
     ['5', 'years'],
     ['2', 'months'],
     ['2', 'years'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['1', 'month'],
     ['2', 'years'],
     ['1', 'year'],
     ['1', 'year'],
     ['1', 'year'],
     ['2', 'months'],
     ['2', 'months'],
     ['3', 'years'],
     ['3', 'months'],
     ['2', 'years'],
     ['4', 'years'],
     ['2', 'years'],
     ['2', 'years'],
     ['7', 'years'],
     ['2', 'months'],
     ['4', 'years'],
     ['2', 'years'],
     ['1', 'year'],
     ['2', 'months'],
     ['4', 'years'],
     ['4', 'months'],
     ['1', 'year'],
     ['2', 'months'],
     ['2', 'months'],
     ['3', 'years'],
     ['4', 'years'],
     ['3', 'years'],
     ['8', 'years'],
     ['2', 'months'],
     ['2', 'months'],
     ['3', 'months'],
     ['2', 'months'],
     ['2', 'years'],
     ['3', 'months'],
     ['6', 'years'],
     ['2', 'months'],
     ['2', 'months'],
     ['1', 'year'],
     ['1', 'year'],
     ['7', 'years'],
     ['1', 'year'],
     ['2', 'years'],
     ['13', 'years'],
     ['2', 'weeks'],
     ['2', 'months'],
     ['1', 'year'],
     ['7', 'years'],
     ['3', 'years'],
     ['2', 'years'],
     ['2', 'months'],
     ['2', 'months'],
     ['3', 'months'],
     ['2', 'months'],
     ['3', 'months'],
     ['3', 'months'],
     ['3', 'months'],
     ['3', 'months'],
     ['2', 'months'],
     ['2', 'years'],
     ['5', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['2', 'months'],
     ['1', 'year'],
     ['4', 'years'],
     ['1', 'year'],
     ['4', 'months'],
     ['2', 'years'],
     ['2', 'years'],
     ['1', 'month'],
     ['2', 'years'],
     ['2', 'months'],
     ['8', 'months'],
     ['7', 'years'],
     ['7', 'years'],
     ['2', 'months'],
     ['4', 'months'],
     ['3', 'years'],
     ['1', 'month']]



# Pair program #1: 
Take 10 minutes to fill in the function below with code that converts age upon outcome to days upon outcome.


```python
# check what values we have for time frame
unit_values = [age[0] if age[0] == 'NULL' 
               else age[1] for age in 
               animals['age_upon_outcome'].str.split(' ')]
set(unit_values)
```




    {'NULL', 'days', 'month', 'months', 'weeks', 'year', 'years'}



Now, fill in the definition below to convert the ages to days


```python

def age_to_days(age):
    
    '''
    params: age upon outcome of shelter animal. 
    A number followed by a unit of time 
    'NULL', 'days', 'month', 'months', 'week', 'weeks', 'year', 'years'
    
    returns: days old at outcome
    '''
    
    age_split = age.split(' ')
    
    if len(age_split) == 1:
        return np.nan
    
    elif ... :
        return
    
    elif ... :
         pass
    
    elif ... :
         pass
    
    else:
         pass
    
    
animals['age_upon_outcome'].apply(age_to_days)

```




    0     NaN
    1     NaN
    2     NaN
    3     NaN
    4     NaN
           ..
    995   NaN
    996   NaN
    997   NaN
    998   NaN
    999   NaN
    Name: age_upon_outcome, Length: 1000, dtype: float64




```python
# Import solution to age todays
from src.sol import age_to_days
animals['age_upon_outcome']= animals['age_upon_outcome'].apply(age_to_days)

```


```python
# Let's look at the average age upon outcome of Adopted animals

```

### Anonymous Functions (Lambda Abstraction)

Simple functions can be defined right in the function call. This is called 'lambda abstraction'; the function thus defined has no name and hence is "anonymous".


```python
student_list
```




    ['Anj',
     'Guy',
     'Ivan',
     'DarigaSilverman',
     'Raf',
     'Emily',
     'Alex',
     'Christa',
     'Saad']




```python
list(map(lambda x: x + ' is '  + 
                    np.random.choice(['hungry', 'sleepy', 'hangry', 
                                      'super pumped about list comprehensions'],
                                     p=[.325,.325,.325,.025]), 
                 student_list))
```




    ['Anj is hungry',
     'Guy is sleepy',
     'Ivan is sleepy',
     'DarigaSilverman is hangry',
     'Raf is hungry',
     'Emily is sleepy',
     'Alex is super pumped about list comprehensions',
     'Christa is hangry',
     'Saad is hangry']



# Student Screen Share
Use another lambda function to convert days days upon outcome to weeks upon outcome <br>



```python
# Your code here
```

# Methods for Re-Organizing DataFrames: .groupby()

Those of you familiar with SQL have probably used the GROUP BY command. (And if you haven't, you'll see it very soon!) Pandas has this, too.

The .groupby() method is especially useful for aggregate functions applied to the data grouped in particular ways.


```python
animals.groupby('animal_type').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age_upon_outcome</th>
    </tr>
    <tr>
      <th>animal_type</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bird</th>
      <td>469.285714</td>
    </tr>
    <tr>
      <th>Cat</th>
      <td>397.442529</td>
    </tr>
    <tr>
      <th>Dog</th>
      <td>821.215569</td>
    </tr>
    <tr>
      <th>Livestock</th>
      <td>180.000000</td>
    </tr>
    <tr>
      <th>Other</th>
      <td>690.326087</td>
    </tr>
  </tbody>
</table>
</div>



Notice the object type [DataFrameGroupBy](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html) object. 

#### .groups and .get_group()


```python
animals.groupby(['animal_type', 'outcome_type'])
```




    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x10a130970>




```python
# This retuns each group indexed by the group name: I.E. 'Bird', along with the row indices of each value
animals.groupby('animal_type').groups
```




    {'Bird': [470, 668, 669, 670, 723, 796, 816], 'Cat': [2, 3, 4, 5, 17, 19, 24, 25, 35, 36, 37, 43, 44, 47, 48, 49, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 66, 67, 68, 69, 70, 73, 77, 78, 85, 87, 88, 92, 97, 98, 99, 100, 101, 102, 103, 109, 111, 115, 119, 123, 135, 136, 137, 141, 160, 161, 165, 166, 168, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 197, 210, 213, 216, 230, 233, 234, 235, 237, 238, 239, 242, 249, 250, 253, 254, 255, 256, 257, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, ...], 'Dog': [0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 26, 27, 28, 29, 30, 31, 32, 33, 34, 38, 40, 41, 42, 45, 46, 50, 51, 52, 63, 65, 71, 72, 74, 79, 80, 81, 82, 83, 84, 86, 89, 90, 91, 94, 95, 96, 104, 105, 106, 107, 108, 110, 112, 113, 114, 116, 117, 118, 120, 121, 122, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 138, 139, 140, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 156, 157, 158, 159, 162, 163, ...], 'Livestock': [76, 211, 337, 820, 834, 838, 841], 'Other': [18, 23, 39, 75, 93, 155, 231, 240, 300, 327, 348, 354, 356, 361, 362, 363, 364, 365, 440, 468, 476, 486, 487, 495, 537, 549, 573, 577, 589, 614, 636, 637, 648, 649, 650, 651, 741, 759, 761, 789, 886, 922, 923, 924, 957, 958]}



Once we know we are working with a type of object, it opens up a suite of attributes and methods. One attribute we can look at is groups.


```python
animals.groupby('animal_type').get_group('Dog')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>animal_id</th>
      <th>name</th>
      <th>datetime</th>
      <th>monthyear</th>
      <th>date_of_birth</th>
      <th>outcome_type</th>
      <th>animal_type</th>
      <th>sex_upon_outcome</th>
      <th>age_upon_outcome</th>
      <th>breed</th>
      <th>color</th>
      <th>outcome_subtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A821019</td>
      <td>Spot</td>
      <td>2020-12-08T12:37:00.000</td>
      <td>2020-12-08T12:37:00.000</td>
      <td>2017-04-03T00:00:00.000</td>
      <td>Adoption</td>
      <td>Dog</td>
      <td>Neutered Male</td>
      <td>1095.0</td>
      <td>Pit Bull</td>
      <td>White/Black</td>
      <td>no_type_or_subtype</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A824438</td>
      <td>*Rose</td>
      <td>2020-12-08T12:27:00.000</td>
      <td>2020-12-08T12:27:00.000</td>
      <td>2011-11-27T00:00:00.000</td>
      <td>Adoption</td>
      <td>Dog</td>
      <td>Spayed Female</td>
      <td>3285.0</td>
      <td>German Shepherd</td>
      <td>Tan/Black</td>
      <td>no_type_or_subtype</td>
    </tr>
    <tr>
      <th>6</th>
      <td>A825091</td>
      <td>*Darla</td>
      <td>2020-12-08T11:44:00.000</td>
      <td>2020-12-08T11:44:00.000</td>
      <td>2019-10-27T00:00:00.000</td>
      <td>Adoption</td>
      <td>Dog</td>
      <td>Spayed Female</td>
      <td>365.0</td>
      <td>Pit Bull</td>
      <td>White/Black</td>
      <td>Foster</td>
    </tr>
    <tr>
      <th>7</th>
      <td>A821660</td>
      <td>*Juice</td>
      <td>2020-12-08T11:41:00.000</td>
      <td>2020-12-08T11:41:00.000</td>
      <td>2015-08-20T00:00:00.000</td>
      <td>Transfer</td>
      <td>Dog</td>
      <td>Neutered Male</td>
      <td>1825.0</td>
      <td>Pit Bull</td>
      <td>White/Brown</td>
      <td>Partner</td>
    </tr>
    <tr>
      <th>8</th>
      <td>A826472</td>
      <td>*Electra</td>
      <td>2020-12-08T11:37:00.000</td>
      <td>2020-12-08T11:37:00.000</td>
      <td>2018-11-29T00:00:00.000</td>
      <td>Transfer</td>
      <td>Dog</td>
      <td>Intact Female</td>
      <td>730.0</td>
      <td>Pit Bull</td>
      <td>Black</td>
      <td>Partner</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>993</th>
      <td>A825181</td>
      <td>Bell</td>
      <td>2020-11-04T15:54:00.000</td>
      <td>2020-11-04T15:54:00.000</td>
      <td>2020-02-29T00:00:00.000</td>
      <td>Adoption</td>
      <td>Dog</td>
      <td>Spayed Female</td>
      <td>240.0</td>
      <td>Shetland Sheepdog Mix</td>
      <td>Brown/Black</td>
      <td>no_type_or_subtype</td>
    </tr>
    <tr>
      <th>994</th>
      <td>A695280</td>
      <td>Laika</td>
      <td>2020-11-04T15:37:00.000</td>
      <td>2020-11-04T15:37:00.000</td>
      <td>2013-01-12T00:00:00.000</td>
      <td>Euthanasia</td>
      <td>Dog</td>
      <td>Spayed Female</td>
      <td>2555.0</td>
      <td>German Shepherd</td>
      <td>Black/Brown</td>
      <td>Aggressive</td>
    </tr>
    <tr>
      <th>995</th>
      <td>A676602</td>
      <td>Farley</td>
      <td>2020-11-04T13:56:00.000</td>
      <td>2020-11-04T13:56:00.000</td>
      <td>2013-10-12T00:00:00.000</td>
      <td>Adoption</td>
      <td>Dog</td>
      <td>Neutered Male</td>
      <td>2555.0</td>
      <td>American Bulldog Mix</td>
      <td>White/Brown Brindle</td>
      <td>Foster</td>
    </tr>
    <tr>
      <th>997</th>
      <td>A825276</td>
      <td>unnamed</td>
      <td>2020-11-04T13:24:00.000</td>
      <td>2020-11-04T13:24:00.000</td>
      <td>2020-07-01T00:00:00.000</td>
      <td>Transfer</td>
      <td>Dog</td>
      <td>Intact Male</td>
      <td>120.0</td>
      <td>Doberman Pinsch</td>
      <td>Brown</td>
      <td>Partner</td>
    </tr>
    <tr>
      <th>998</th>
      <td>A825145</td>
      <td>unnamed</td>
      <td>2020-11-04T13:24:00.000</td>
      <td>2020-11-04T13:24:00.000</td>
      <td>2017-10-29T00:00:00.000</td>
      <td>Transfer</td>
      <td>Dog</td>
      <td>Neutered Male</td>
      <td>1095.0</td>
      <td>Great Pyrenees</td>
      <td>White</td>
      <td>Partner</td>
    </tr>
  </tbody>
</table>
<p>570 rows × 12 columns</p>
</div>



We can group by multiple columns, and also return a DataFrameGroupBy object


```python
animals.groupby(['animal_type', 'outcome_type'])
```




    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x119d35a60>




```python
animals.groupby(['animal_type', 'outcome_type']).groups.keys()
```




    dict_keys([('Bird', 'Adoption'), ('Bird', 'Died'), ('Bird', 'Disposal'), ('Bird', 'no_type_or_subtype'), ('Cat', 'Adoption'), ('Cat', 'Died'), ('Cat', 'Disposal'), ('Cat', 'Euthanasia'), ('Cat', 'Return to Owner'), ('Cat', 'Rto-Adopt'), ('Cat', 'Transfer'), ('Cat', 'no_type_or_subtype'), ('Dog', 'Adoption'), ('Dog', 'Died'), ('Dog', 'Disposal'), ('Dog', 'Euthanasia'), ('Dog', 'Missing'), ('Dog', 'Return to Owner'), ('Dog', 'Rto-Adopt'), ('Dog', 'Transfer'), ('Dog', 'no_type_or_subtype'), ('Livestock', 'Euthanasia'), ('Livestock', 'no_type_or_subtype'), ('Other', 'Adoption'), ('Other', 'Died'), ('Other', 'Disposal'), ('Other', 'Euthanasia'), ('Other', 'Transfer')])



#### Aggregating


```python
# Just like with single axis groups, we can aggregate on multiple axis
animals.groupby(['animal_type', 'outcome_type']).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>age_upon_outcome</th>
    </tr>
    <tr>
      <th>animal_type</th>
      <th>outcome_type</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Bird</th>
      <th>Adoption</th>
      <td>365.000000</td>
    </tr>
    <tr>
      <th>Died</th>
      <td>730.000000</td>
    </tr>
    <tr>
      <th>Disposal</th>
      <td>730.000000</td>
    </tr>
    <tr>
      <th>no_type_or_subtype</th>
      <td>365.000000</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Cat</th>
      <th>Adoption</th>
      <td>262.312253</td>
    </tr>
    <tr>
      <th>Died</th>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>Disposal</th>
      <td>730.000000</td>
    </tr>
    <tr>
      <th>Euthanasia</th>
      <td>1511.833333</td>
    </tr>
    <tr>
      <th>Return to Owner</th>
      <td>724.090909</td>
    </tr>
    <tr>
      <th>Rto-Adopt</th>
      <td>6205.000000</td>
    </tr>
    <tr>
      <th>Transfer</th>
      <td>573.276923</td>
    </tr>
    <tr>
      <th>no_type_or_subtype</th>
      <td>730.000000</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">Dog</th>
      <th>Adoption</th>
      <td>698.671171</td>
    </tr>
    <tr>
      <th>Died</th>
      <td>1832.000000</td>
    </tr>
    <tr>
      <th>Disposal</th>
      <td>797.000000</td>
    </tr>
    <tr>
      <th>Euthanasia</th>
      <td>1959.000000</td>
    </tr>
    <tr>
      <th>Missing</th>
      <td>21.000000</td>
    </tr>
    <tr>
      <th>Return to Owner</th>
      <td>1341.808511</td>
    </tr>
    <tr>
      <th>Rto-Adopt</th>
      <td>1216.666667</td>
    </tr>
    <tr>
      <th>Transfer</th>
      <td>576.830189</td>
    </tr>
    <tr>
      <th>no_type_or_subtype</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Livestock</th>
      <th>Euthanasia</th>
      <td>180.000000</td>
    </tr>
    <tr>
      <th>no_type_or_subtype</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Other</th>
      <th>Adoption</th>
      <td>973.333333</td>
    </tr>
    <tr>
      <th>Died</th>
      <td>730.000000</td>
    </tr>
    <tr>
      <th>Disposal</th>
      <td>730.000000</td>
    </tr>
    <tr>
      <th>Euthanasia</th>
      <td>646.571429</td>
    </tr>
    <tr>
      <th>Transfer</th>
      <td>608.333333</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We can then get a specific group, such as Cats that were adopted
animals.groupby(['animal_type', 'outcome_type']).get_group(('Cat', 'Adoption'))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>animal_id</th>
      <th>name</th>
      <th>datetime</th>
      <th>monthyear</th>
      <th>date_of_birth</th>
      <th>outcome_type</th>
      <th>animal_type</th>
      <th>sex_upon_outcome</th>
      <th>age_upon_outcome</th>
      <th>breed</th>
      <th>color</th>
      <th>outcome_subtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>A825587</td>
      <td>*Ludwig</td>
      <td>2020-12-08T12:22:00.000</td>
      <td>2020-12-08T12:22:00.000</td>
      <td>2011-11-06T00:00:00.000</td>
      <td>Adoption</td>
      <td>Cat</td>
      <td>Neutered Male</td>
      <td>3285.0</td>
      <td>Domestic Medium Hair</td>
      <td>Cream Tabby</td>
      <td>Foster</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A819626</td>
      <td>unnamed</td>
      <td>2020-12-08T11:53:00.000</td>
      <td>2020-12-08T11:53:00.000</td>
      <td>2020-06-25T00:00:00.000</td>
      <td>Adoption</td>
      <td>Cat</td>
      <td>Neutered Male</td>
      <td>150.0</td>
      <td>Domestic Shorthair</td>
      <td>White/Black</td>
      <td>Foster</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A819624</td>
      <td>unnamed</td>
      <td>2020-12-08T11:52:00.000</td>
      <td>2020-12-08T11:52:00.000</td>
      <td>2020-06-25T00:00:00.000</td>
      <td>Adoption</td>
      <td>Cat</td>
      <td>Neutered Male</td>
      <td>150.0</td>
      <td>Domestic Shorthair</td>
      <td>Black</td>
      <td>Foster</td>
    </tr>
    <tr>
      <th>17</th>
      <td>A825246</td>
      <td>unnamed</td>
      <td>2020-12-08T10:58:00.000</td>
      <td>2020-12-08T10:58:00.000</td>
      <td>2020-09-04T00:00:00.000</td>
      <td>Adoption</td>
      <td>Cat</td>
      <td>Neutered Male</td>
      <td>90.0</td>
      <td>Domestic Shorthair</td>
      <td>Gray Tabby/White</td>
      <td>Foster</td>
    </tr>
    <tr>
      <th>25</th>
      <td>A826721</td>
      <td>Sativa</td>
      <td>2020-12-07T18:11:00.000</td>
      <td>2020-12-07T18:11:00.000</td>
      <td>2019-12-04T00:00:00.000</td>
      <td>Adoption</td>
      <td>Cat</td>
      <td>Spayed Female</td>
      <td>365.0</td>
      <td>Domestic Shorthair Mix</td>
      <td>Black/White</td>
      <td>no_type_or_subtype</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>980</th>
      <td>A823832</td>
      <td>*Holden</td>
      <td>2020-11-05T12:16:00.000</td>
      <td>2020-11-05T12:16:00.000</td>
      <td>2020-08-17T00:00:00.000</td>
      <td>Adoption</td>
      <td>Cat</td>
      <td>Neutered Male</td>
      <td>60.0</td>
      <td>Domestic Shorthair</td>
      <td>Brown Tabby/White</td>
      <td>Foster</td>
    </tr>
    <tr>
      <th>981</th>
      <td>A823893</td>
      <td>*Phoebe</td>
      <td>2020-11-05T12:09:00.000</td>
      <td>2020-11-05T12:09:00.000</td>
      <td>2020-08-10T00:00:00.000</td>
      <td>Adoption</td>
      <td>Cat</td>
      <td>Spayed Female</td>
      <td>60.0</td>
      <td>Domestic Shorthair</td>
      <td>Silver Tabby</td>
      <td>Foster</td>
    </tr>
    <tr>
      <th>983</th>
      <td>A823911</td>
      <td>*Bunny</td>
      <td>2020-11-05T10:33:00.000</td>
      <td>2020-11-05T10:33:00.000</td>
      <td>2020-08-17T00:00:00.000</td>
      <td>Adoption</td>
      <td>Cat</td>
      <td>Spayed Female</td>
      <td>60.0</td>
      <td>Domestic Shorthair</td>
      <td>Brown Tabby</td>
      <td>Foster</td>
    </tr>
    <tr>
      <th>992</th>
      <td>A823026</td>
      <td>*Banana</td>
      <td>2020-11-04T16:01:00.000</td>
      <td>2020-11-04T16:01:00.000</td>
      <td>2020-08-21T00:00:00.000</td>
      <td>Adoption</td>
      <td>Cat</td>
      <td>Neutered Male</td>
      <td>60.0</td>
      <td>Domestic Shorthair Mix</td>
      <td>Orange Tabby</td>
      <td>Foster</td>
    </tr>
    <tr>
      <th>999</th>
      <td>A824293</td>
      <td>*Blueboy</td>
      <td>2020-11-04T12:35:00.000</td>
      <td>2020-11-04T12:35:00.000</td>
      <td>2020-09-04T00:00:00.000</td>
      <td>Adoption</td>
      <td>Cat</td>
      <td>Neutered Male</td>
      <td>30.0</td>
      <td>Domestic Shorthair</td>
      <td>Blue Tabby</td>
      <td>Foster</td>
    </tr>
  </tbody>
</table>
<p>253 rows × 12 columns</p>
</div>




```python
# Other methods
animals.groupby(['animal_type', 'outcome_type']).first()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>animal_id</th>
      <th>name</th>
      <th>datetime</th>
      <th>monthyear</th>
      <th>date_of_birth</th>
      <th>sex_upon_outcome</th>
      <th>age_upon_outcome</th>
      <th>breed</th>
      <th>color</th>
      <th>outcome_subtype</th>
    </tr>
    <tr>
      <th>animal_type</th>
      <th>outcome_type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">Bird</th>
      <th>Adoption</th>
      <td>A825549</td>
      <td>unnamed</td>
      <td>2020-11-14T13:24:00.000</td>
      <td>2020-11-14T13:24:00.000</td>
      <td>2019-11-05T00:00:00.000</td>
      <td>Unknown</td>
      <td>365.0</td>
      <td>Cockatiel</td>
      <td>White/Yellow</td>
      <td>no_type_or_subtype</td>
    </tr>
    <tr>
      <th>Died</th>
      <td>A825560</td>
      <td>unnamed</td>
      <td>2020-11-10T09:03:00.000</td>
      <td>2020-11-10T09:03:00.000</td>
      <td>2018-11-06T00:00:00.000</td>
      <td>Intact Male</td>
      <td>730.0</td>
      <td>Rhode Island</td>
      <td>Brown</td>
      <td>In Kennel</td>
    </tr>
    <tr>
      <th>Disposal</th>
      <td>A826140</td>
      <td>unnamed</td>
      <td>2020-11-23T09:00:00.000</td>
      <td>2020-11-23T09:00:00.000</td>
      <td>2018-11-19T00:00:00.000</td>
      <td>Unknown</td>
      <td>730.0</td>
      <td>Pigeon</td>
      <td>Gray/White</td>
      <td>no_type_or_subtype</td>
    </tr>
    <tr>
      <th>no_type_or_subtype</th>
      <td>A824708</td>
      <td>Loko</td>
      <td>2020-11-12T17:15:00.000</td>
      <td>2020-11-12T17:15:00.000</td>
      <td>2019-11-05T00:00:00.000</td>
      <td>Intact Female</td>
      <td>365.0</td>
      <td>Cockatiel</td>
      <td>Gray</td>
      <td>no_type_or_subtype</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Cat</th>
      <th>Adoption</th>
      <td>A825587</td>
      <td>*Ludwig</td>
      <td>2020-12-08T12:22:00.000</td>
      <td>2020-12-08T12:22:00.000</td>
      <td>2011-11-06T00:00:00.000</td>
      <td>Neutered Male</td>
      <td>3285.0</td>
      <td>Domestic Medium Hair</td>
      <td>Cream Tabby</td>
      <td>Foster</td>
    </tr>
    <tr>
      <th>Died</th>
      <td>A823214</td>
      <td>unnamed</td>
      <td>2020-11-25T17:12:00.000</td>
      <td>2020-11-25T17:12:00.000</td>
      <td>2020-08-03T00:00:00.000</td>
      <td>Neutered Male</td>
      <td>90.0</td>
      <td>Domestic Shorthair</td>
      <td>Brown Tabby</td>
      <td>In Foster</td>
    </tr>
    <tr>
      <th>Disposal</th>
      <td>A826207</td>
      <td>unnamed</td>
      <td>2020-11-24T09:00:00.000</td>
      <td>2020-11-24T09:00:00.000</td>
      <td>2018-11-21T00:00:00.000</td>
      <td>Intact Female</td>
      <td>730.0</td>
      <td>Domestic Longhair</td>
      <td>White/White</td>
      <td>no_type_or_subtype</td>
    </tr>
    <tr>
      <th>Euthanasia</th>
      <td>A826415</td>
      <td>unnamed</td>
      <td>2020-12-02T09:58:00.000</td>
      <td>2020-12-02T09:58:00.000</td>
      <td>2018-11-27T00:00:00.000</td>
      <td>Intact Male</td>
      <td>730.0</td>
      <td>Siamese</td>
      <td>Lynx Point</td>
      <td>Suffering</td>
    </tr>
    <tr>
      <th>Return to Owner</th>
      <td>A826884</td>
      <td>unnamed</td>
      <td>2020-12-08T11:49:00.000</td>
      <td>2020-12-08T11:49:00.000</td>
      <td>2020-07-08T00:00:00.000</td>
      <td>Intact Female</td>
      <td>150.0</td>
      <td>Domestic Shorthair</td>
      <td>Brown Tabby</td>
      <td>no_type_or_subtype</td>
    </tr>
    <tr>
      <th>Rto-Adopt</th>
      <td>A825478</td>
      <td>Ditto</td>
      <td>2020-11-10T12:01:00.000</td>
      <td>2020-11-10T12:01:00.000</td>
      <td>2003-11-10T00:00:00.000</td>
      <td>Spayed Female</td>
      <td>6205.0</td>
      <td>Domestic Shorthair</td>
      <td>Blue Tabby/White</td>
      <td>no_type_or_subtype</td>
    </tr>
    <tr>
      <th>Transfer</th>
      <td>A826564</td>
      <td>unnamed</td>
      <td>2020-12-07T18:13:00.000</td>
      <td>2020-12-07T18:13:00.000</td>
      <td>2020-06-06T00:00:00.000</td>
      <td>Spayed Female</td>
      <td>180.0</td>
      <td>Domestic Shorthair</td>
      <td>Blue Tabby/White</td>
      <td>Partner</td>
    </tr>
    <tr>
      <th>no_type_or_subtype</th>
      <td>A825923</td>
      <td>Harvey</td>
      <td>2020-12-08T00:00:00.000</td>
      <td>2020-12-08T00:00:00.000</td>
      <td>2020-12-08T00:00:00.000</td>
      <td>Intact Male</td>
      <td>730.0</td>
      <td>Domestic Shorthair</td>
      <td>Unknown</td>
      <td>no_type_or_subtype</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">Dog</th>
      <th>Adoption</th>
      <td>A821019</td>
      <td>Spot</td>
      <td>2020-12-08T12:37:00.000</td>
      <td>2020-12-08T12:37:00.000</td>
      <td>2017-04-03T00:00:00.000</td>
      <td>Neutered Male</td>
      <td>1095.0</td>
      <td>Pit Bull</td>
      <td>White/Black</td>
      <td>no_type_or_subtype</td>
    </tr>
    <tr>
      <th>Died</th>
      <td>A690421</td>
      <td>Tyson</td>
      <td>2020-11-16T10:25:00.000</td>
      <td>2020-11-16T10:25:00.000</td>
      <td>2010-10-20T00:00:00.000</td>
      <td>Neutered Male</td>
      <td>3650.0</td>
      <td>Beagle Mix</td>
      <td>Tan/White</td>
      <td>In Foster</td>
    </tr>
    <tr>
      <th>Disposal</th>
      <td>A826087</td>
      <td>Bertha</td>
      <td>2020-11-23T09:01:00.000</td>
      <td>2020-11-23T09:01:00.000</td>
      <td>2020-10-19T00:00:00.000</td>
      <td>Intact Female</td>
      <td>35.0</td>
      <td>Unknown</td>
      <td>Black</td>
      <td>no_type_or_subtype</td>
    </tr>
    <tr>
      <th>Euthanasia</th>
      <td>A825027</td>
      <td>*Moody</td>
      <td>2020-12-07T15:40:00.000</td>
      <td>2020-12-07T15:40:00.000</td>
      <td>2008-12-07T00:00:00.000</td>
      <td>Neutered Male</td>
      <td>4380.0</td>
      <td>Rottweiler/Labrador Retriever</td>
      <td>Black/Brown</td>
      <td>Suffering</td>
    </tr>
    <tr>
      <th>Missing</th>
      <td>A824954</td>
      <td>unnamed</td>
      <td>2020-11-09T10:35:00.000</td>
      <td>2020-11-09T10:35:00.000</td>
      <td>2020-10-16T00:00:00.000</td>
      <td>Intact Male</td>
      <td>21.0</td>
      <td>Labrador Retriever</td>
      <td>White/Black</td>
      <td>In Kennel</td>
    </tr>
    <tr>
      <th>Return to Owner</th>
      <td>A826860</td>
      <td>Chi Chi</td>
      <td>2020-12-08T11:20:00.000</td>
      <td>2020-12-08T11:20:00.000</td>
      <td>2015-12-07T00:00:00.000</td>
      <td>Intact Female</td>
      <td>1825.0</td>
      <td>German Shepherd Mix</td>
      <td>Tan/White</td>
      <td>no_type_or_subtype</td>
    </tr>
    <tr>
      <th>Rto-Adopt</th>
      <td>A826382</td>
      <td>Rocky</td>
      <td>2020-12-04T11:48:00.000</td>
      <td>2020-12-04T11:48:00.000</td>
      <td>2018-11-25T00:00:00.000</td>
      <td>Neutered Male</td>
      <td>730.0</td>
      <td>German Shepherd</td>
      <td>Brown/Black</td>
      <td>no_type_or_subtype</td>
    </tr>
    <tr>
      <th>Transfer</th>
      <td>A821660</td>
      <td>*Juice</td>
      <td>2020-12-08T11:41:00.000</td>
      <td>2020-12-08T11:41:00.000</td>
      <td>2015-08-20T00:00:00.000</td>
      <td>Neutered Male</td>
      <td>1825.0</td>
      <td>Pit Bull</td>
      <td>White/Brown</td>
      <td>Partner</td>
    </tr>
    <tr>
      <th>no_type_or_subtype</th>
      <td>A826701</td>
      <td>Chubby</td>
      <td>2020-12-04T14:47:00.000</td>
      <td>2020-12-04T14:47:00.000</td>
      <td>2020-12-04T14:47:00.000</td>
      <td>Intact Male</td>
      <td>NaN</td>
      <td>Chihuahua Shorthair</td>
      <td>Unknown</td>
      <td>no_type_or_subtype</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Livestock</th>
      <th>Euthanasia</th>
      <td>A826448</td>
      <td>unnamed</td>
      <td>2020-11-28T15:40:00.000</td>
      <td>2020-11-28T15:40:00.000</td>
      <td>2020-05-28T00:00:00.000</td>
      <td>Intact Male</td>
      <td>180.0</td>
      <td>Pig</td>
      <td>Black/White</td>
      <td>Suffering</td>
    </tr>
    <tr>
      <th>no_type_or_subtype</th>
      <td>A825584</td>
      <td>Daisy</td>
      <td>2020-12-06T00:00:00.000</td>
      <td>2020-12-06T00:00:00.000</td>
      <td>2020-12-06T00:00:00.000</td>
      <td>Intact Female</td>
      <td>NaN</td>
      <td>Fowl</td>
      <td>Unknown</td>
      <td>no_type_or_subtype</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Other</th>
      <th>Adoption</th>
      <td>A826601</td>
      <td>unnamed</td>
      <td>2020-12-06T11:22:00.000</td>
      <td>2020-12-06T11:22:00.000</td>
      <td>2019-12-02T00:00:00.000</td>
      <td>Unknown</td>
      <td>365.0</td>
      <td>Rabbit Sh</td>
      <td>Chocolate/White</td>
      <td>no_type_or_subtype</td>
    </tr>
    <tr>
      <th>Died</th>
      <td>A826413</td>
      <td>unnamed</td>
      <td>2020-11-28T08:05:00.000</td>
      <td>2020-11-28T08:05:00.000</td>
      <td>2018-11-27T00:00:00.000</td>
      <td>Unknown</td>
      <td>730.0</td>
      <td>Raccoon</td>
      <td>Black/Gray</td>
      <td>Enroute</td>
    </tr>
    <tr>
      <th>Disposal</th>
      <td>A826139</td>
      <td>unnamed</td>
      <td>2020-11-24T00:00:00.000</td>
      <td>2020-11-24T00:00:00.000</td>
      <td>2018-11-19T00:00:00.000</td>
      <td>Unknown</td>
      <td>730.0</td>
      <td>Raccoon</td>
      <td>Black/White</td>
      <td>no_type_or_subtype</td>
    </tr>
    <tr>
      <th>Euthanasia</th>
      <td>A826848</td>
      <td>unnamed</td>
      <td>2020-12-08T10:01:00.000</td>
      <td>2020-12-08T10:01:00.000</td>
      <td>2018-12-07T00:00:00.000</td>
      <td>Unknown</td>
      <td>730.0</td>
      <td>Bat</td>
      <td>Brown</td>
      <td>Rabies Risk</td>
    </tr>
    <tr>
      <th>Transfer</th>
      <td>A826355</td>
      <td>unnamed</td>
      <td>2020-12-03T18:20:00.000</td>
      <td>2020-12-03T18:20:00.000</td>
      <td>2019-11-25T00:00:00.000</td>
      <td>Unknown</td>
      <td>365.0</td>
      <td>Hedgehog</td>
      <td>Tan</td>
      <td>Partner</td>
    </tr>
  </tbody>
</table>
</div>


