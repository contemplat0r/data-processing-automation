
# coding: utf-8

# ##  <center> Airbnb booking data cleaning </center>
# This is the data that the Airbnb provided for the Kaggle competition in which it is necessary to predict the country of the user who booked a taxi. Data needs preliminary cleaning.
# 1. [Data loading](#dload)
# 2. [Initial exploration and cleaning user data](#uclean)
# 3. [Initial exploratioin and cleaning session data](#sclean)
# 4. [Description of the final result](#result)

# In[1]:


import os
import pathlib
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# #### <a name="dload">Data loading.</a>

# We have two tables that contains description of Airbnb users and sessions 

# In[3]:


users_df = pd.read_csv('../input/train_users_2.csv')


# In[4]:


sessions_df = pd.read_csv('../input/sessions.csv')


# #### <a name="uclean">Initial exploration and cleaning user data.</a>

# Let's look at the users data using standard Pandas methods.

# In[5]:


users_df.info()


# In[6]:


users_df.head()


# In[7]:


users_df.describe()


# In[8]:


sorted_age = users_df['age'].sort_values()


# In[9]:


sorted_age.head()


# In[10]:


sorted_age.tail()


# We can see that is significant number of NaNs in 'date_first_booking' and 'age' columns. And min age is 1 and max age is 2014
# that is terrible. And 'first_affiliate_tracked' column also contain NaNs.

# Let's look at 'age' column carefully, find the most common value, some of the most rare and most common values.

# In[11]:


users_df['age'].mode()


# In[12]:


count_age_values = users_df['age'].value_counts().sort_index()


# In[13]:


count_age_values.shape


# In[14]:


count_age_values.min()


# In[15]:


count_age_values.max()


# In[16]:


count_age_values.idxmax()


# In[17]:


count_age_values[count_age_values.idxmax()]


# In[18]:


count_age_values.head()


# In[19]:


count_age_values.tail()


# For clarity, let's make some graphs.

# In[20]:


cm = plt.cm.get_cmap('RdYlBu_r')
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111)
n, bins, patches = ax.hist(users_df['age'].dropna(), 100, color='green', alpha=0.5, histtype='bar', ec='black')
bin_centers = 0.5 * (bins[:-1] + bins[1:])

col = bin_centers - min(bin_centers)
col /= max(col)

for c, path in zip(col, patches):
    plt.setp(path, 'facecolor', cm(c))
ax.set_xlabel("Age")
ax.set_ylabel("Rate")
ax.set_title("Age rate histogram")
plt.show()


# Construct a histogram for anomalous values exceeding 90.

# In[21]:


cm = plt.cm.get_cmap('RdYlBu_r')
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111)
n, bins, patches = ax.hist(users_df.loc[users_df['age'] > 90, 'age'].dropna(), 100, color='green', alpha=0.5, histtype='bar', ec='black')
bin_centers = 0.5 * (bins[:-1] + bins[1:])

col = bin_centers - min(bin_centers)
col /= max(col)

for c, path in zip(col, patches):
    plt.setp(path, 'facecolor', cm(c))
ax.set_xlabel("Age")
ax.set_ylabel("Rate")
ax.set_title("Age rate histogram. Age > 90")
plt.show()


# Construct a histogram for anomalous values smaller than 16.

# In[22]:


cm = plt.cm.get_cmap('RdYlBu_r')
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111)
n, bins, patches = ax.hist(users_df.loc[users_df['age'] < 16, 'age'].dropna(), 100, color='green', alpha=0.5, histtype='bar', ec='black')
bin_centers = 0.5 * (bins[:-1] + bins[1:])

col = bin_centers - min(bin_centers)
col /= max(col)

for c, path in zip(col, patches):
    plt.setp(path, 'facecolor', cm(c))
ax.set_xlabel("Age")
ax.set_ylabel("Rate")
ax.set_title("Age rate histogram. Age < 16")
plt.show()


# Suppose that all values of age less than 16 and greater than 90 are outliers and erroneous
# (for example, the user made a mistake when entering and entered the current year, or only entered current month date that less then 16).
# Assign to such dates NaN:

# In[28]:


users_df.loc[users_df['age'] > 90, 'age'] = np.nan
users_df.loc[users_df['age'] < 16, 'age'] = np.nan


# And fill them by age mode

# In[44]:


#users_df.loc[users_df['age'].isna(), 'age'] = count_age_values.idxmax()
users_df.loc[users_df['age'].isna(), 'age'] = users_df['age'].mode()[0]


# In[45]:


any(users_df['age'].isna())


# Now we need to decide what to do with 'date_first_booking' column. Calculate what percentage of non NaN values:

# In[ ]:


print("{}%".format(users_df[users_df['date_first_booking'].notna()].shape[0] / users_df.shape[0] * 100))


# Since it contains only 42% not NaN values it seems appropriate to drop this column. Simple strategies - fill in the values with the most common value from the non NaN part, or the average is unlikely to fit here. Let's try a more complex approach.

# Find the number of unique values in non NaN part of 'date_first_booking'.

# In[46]:


len(users_df['date_first_booking'].unique())


# We see that the number of unique values is small compared to the number of rows in a dataset. Therefore, we can apply the frequency approach.  We can try to fill NaN values with values from the not NaN part with the same frequency (probabilities) distribution as in the not NaN part.  And further, when using for training the model, it will be possible to check both options - with the column droped and with the filled.

# Apparently in the future we will use this approach of filling NaNs. Therefore, it makes sense to write a function for this.

# In[47]:


def fill_nans(df, column):
    #calculate number of NaNs in column
    na_len = len(df.loc[df[column].isna(), column])
    #create a pandas series containing values and corresponding quantities from not NaNs part of the column
    count_notna = df.loc[df[column].notna(), column].value_counts()
    #calculate relative frequencies (probabilities) of each value
    frequencies = count_notna / count_notna.sum()
    #make array that contain fill values with the same relative frequencies as in 'not NaN' column part
    fill_values = np.array(
        [np.random.choice(frequencies.index, p=frequencies.values) for _ in range(na_len)]
    )
    #fill NaNs
    df.loc[df[column].isna(), column] = fill_values
    return df


# In[48]:


users_df = fill_nans(users_df, 'date_first_booking')


# In[49]:


any(users_df['date_first_booking'].isna())


# Now consider 'first_affiliate_tracked' column

# In[50]:


first_at_not_na_uniques_count = users_df.loc[
    users_df['first_affiliate_tracked'].notna(), 'first_affiliate_tracked'
].value_counts()


# In[51]:


first_at_not_na_uniques_count.shape


# In[52]:


first_at_not_na_uniques_count


# In[53]:


users_df[users_df['first_affiliate_tracked'].isna()].shape[0] / users_df.shape[0] * 100


# We can see that 'first_affiliate_tracked' column contains 2.84% NaNs. And fill NaNs by mode value - 'untracked' it's not exactly what we need. Other values make up a large proportion of all values. Therefore, we do the same as with 'date_first_booking' column

# In[54]:


users_df = fill_nans(users_df, 'first_affiliate_tracked')


# In[55]:


any(users_df['first_affiliate_tracked'].isna())


# Convert 'date_first_booking', 'date_account_created' and 'timestamp_first_active' to datetime, and add to users_df as new columns.

# In[56]:


users_df['date_first_active'] = pd.to_datetime(users_df['timestamp_first_active'] // 1000000, format = '%Y%m%d')
users_df['date_first_booking_dt'] = pd.to_datetime(users_df['date_first_booking'])
users_df['date_account_created_dt'] = pd.to_datetime(users_df['date_account_created'])


# In[57]:


users_df.head()


# #### <a name="sclean">Initial exploration and cleaning sessions data.</a>

# Now consider the sessions dataset

# In[58]:


sessions_df.info()


# In[59]:


sessions_df.head()


# In[60]:


sessions_df.describe()


# In[61]:


sessions_df.isna().sum() / len(sessions_df) * 100


# We can see that all columns, except 'device type', in sessions datasets have NaN values. 'action_type' and 'action_detail'
# have about 11% NaNs. It seems appropriate to fill such values. And NaN in other columns should also be filled,
# except 'user_id' column (NaNs in this column look at least strange.).
# In this case, it is most advisable to delete rows containing NaN values in 'user_id' column, the more so because they
# are few .
# But first, let's take a closer look at this column.

# In[62]:


users_df_unique_user_ids = set(users_df['id'].unique())
sessions_df_unique_user_ids = set(sessions_df['user_id'].unique())


# In[63]:


print(len(users_df_unique_user_ids))
print(len(sessions_df_unique_user_ids))


# We can see that the number of unique users identifiers is quite different in users dataframe and sessions dataframe.
# Calculate numbers of match ids, and and not match ids.

# In[64]:


print("len of intersection:", len(users_df_unique_user_ids.intersection(sessions_df_unique_user_ids)))
print(
    "len of difference unique user ids in users_df and sessions_df:",
    len(users_df_unique_user_ids.difference(sessions_df_unique_user_ids)))
print("len of difference unique user ids in sessions_df and users_df:",
      len(sessions_df_unique_user_ids.difference(users_df_unique_user_ids)))


# The most interesting is that in sessions dataset there are identifiers of which are not in users dataset. And number
# of intersected users ids not very large. But, all the same, it makes sense to use user dataset in combination
# with sessions dataset.

# Consider columns 'action', 'action_type', 'action_detail' in more detail.

# Find number of unique values in 'action' column:

# In[66]:


len(sessions_df['action'].unique())


# We see that the number of unique values is small compared to the number of rows in a dataset.

# In[67]:


#sessions_df['action'].value_counts()


# Select rows with NaN in 'action' column

# In[68]:


action_nan = sessions_df[sessions_df['action'].isna()]


# Find the number of unique values in 'action_type', 'action_detail', 'device_type' columns in those rows where there are NaNs in 'action' column

# In[69]:


print(len(action_nan['action_type'].unique()))
print(len(action_nan['action_detail'].unique()))
print(len(action_nan['device_type'].unique()))


# Show this values

# In[70]:


print(action_nan['action_type'].unique())
print(action_nan['action_detail'].unique())
print(action_nan['device_type'].unique())


# We can see that 'action_type' and 'action_detail' contains the same value: 'message_post' if 'action'
# contain NaN in that row

# Select rows with not NaN in 'action' column

# In[71]:


action_not_na = sessions_df[sessions_df['action'].notna()]


# In[72]:


action_not_na_uniques = action_not_na['action'].unique()
action_type_not_na_uniques = action_not_na['action_type'].unique()
action_detail_not_na_uniques = action_not_na['action_detail'].unique()


# In[73]:


print('message_post' in action_type_not_na_uniques)
print('message_post' in action_detail_not_na_uniques)


# We can see that  'action_type' and 'action_detail' contain 'message_post' also in rows where 'action' column not contain NaNs

# Let's see what values are contained in 'action' column in rows where  'action_type' and 'action_detail' columns contains 'messge_post' value

# In[90]:


print(action_not_na.loc[action_not_na['action_type'] == 'message_post', 'action'].unique())
print(action_not_na.loc[action_not_na['action_detail'] == 'message_post', 'action'].unique())


# We can see that values the same. Perhaps this will be useful during the in-depth study of this data. But here we will not do in-depth exploratory data analysis. 

# Fill NaNs in 'action' column as we did earlier for 'date_first_booking' and 'first_affiliate_tracked' columns.

# In[75]:


sessions_df = fill_nans(sessions_df, 'action')


# In[76]:


any(sessions_df.loc[sessions_df['action'].isna(), 'action'])


# Now consider columns 'action_type' and 'action_detail' itself. They have the same number of NaNs, and we can assume that the indices of the rows in which these values are located match up.

# In[77]:


all(sessions_df[sessions_df['action_type'].isna()].index == sessions_df[sessions_df['action_detail'].isna()].index)


# The rows are the same as we expected. Now we select rows that contain NaN in 'action_type' and 'action_detail'
# and rows that not contain. And we can use for selecting NaN and not NaN rows only one of them, 'action_type' for example

# Select separately rows in 'action_type' that contains NaN, and not contains NaN

# In[78]:


act_td_nans_df = sessions_df[sessions_df['action_type'].isna()]
act_td_not_nans_df = sessions_df[sessions_df['action_type'].notna()]


# In[79]:


act_td_nans_df.shape


# In[80]:


act_td_nans_df.head()


# In[81]:


act_td_not_nans_df.shape


# In[82]:


act_td_not_nans_df.head()


# Find number of unique values in not NaNs part of 'action_type' column

# In[83]:


action_t_td_not_nans_uniques_count = act_td_not_nans_df['action_type'].value_counts()


# In[84]:


print(len(action_t_td_not_nans_uniques_count))


# We can see that number of unique values in 'action_type' are few. Do the same for 'action_detail' column

# In[91]:


action_d_td_not_nans_uniques_count = act_td_not_nans_df['action_detail'].value_counts()


# In[92]:


print(len(action_d_td_not_nans_uniques_count))


# Let's look at these values.

# In[88]:


print(action_t_td_not_nans_uniques_count.index)


# In[89]:


print(action_d_td_not_nans_uniques_count.index)


# As we can see, the number of unique values in 'action_type' and 'action_detail' columns is small compared to the number of rows in these columns. Therefore, we can apply for filling NaNs in these columns the same frequency method as for 'action' column and 'date_first_booking' and 'first_affiliate_tracked' columns in users dataset.

# In[93]:


sessions_df = fill_nans(sessions_df, 'action_type')


# In[94]:


any(sessions_df.loc[sessions_df['action_type'].isna(), 'action_type'])


# In[95]:


sessions_df = fill_nans(sessions_df, 'action_detail')


# In[96]:


any(sessions_df.loc[sessions_df['action_detail'].isna(), 'action_detail'])


# Fill NaNs in 'secs_elaplsed' by mean of not NaN part of 'secs_elapsed' (as it is continuos number variable this approach looks prefered)

# In[97]:


sessions_df.loc[sessions_df['secs_elapsed'].isna(), 'secs_elapsed'] = sessions_df['secs_elapsed'].mean()


# Drop all rows where 'user_id' contain NaNs

# In[98]:


sessions_df =  sessions_df.drop(index=sessions_df[sessions_df['user_id'].isna()].index)


# In[99]:


any(sessions_df.loc[sessions_df['user_id'].isna(), 'user_id'])


# We also convert the float values that contain 'secs_elapsed' column to timedelta and add a column with these values to the dataframe.

# In[100]:


sessions_df['secs_elapsed_timedelta'] = pd.to_timedelta(sessions_df['secs_elapsed'], unit='s')


# #### <a name="result">Description of the final result.</a>

# On this dwell. We received a cosistant dataset that does not contain NaNs and outliers. The opposite approach is to remove all rows containing NaNs and columns containing a significant number of NaNs (for example, more than half). Thus, we also get consistent dataset.
# Now we can (preferably after in-depth exploratory data analysis), use the resulting dataset to build various predictive models (using only users datasets or combining it in various ways with sessions datasets) using different algorithms (logistic regression, gradient boosting, SVM etc ..) It is further possible to combine the results of these models (using bagging or/and stacking, or more complex methods) to obtain more accurate predictions.
