import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
#%matplotlib inline
import matplotlib.pyplot as plt
from collections import Counter
from random import randint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sys
import os
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = 'all'


NUM_FIGURE = 1

def my_pltsavefig(fig_title, mydir='./figure/', fig_extension='.png'):
    filename = mydir + fig_title.replace(' ', '-') + fig_extension
    if not os.path.exists(filename):
        plt.savefig(filename)
        return True
    else:
        return False


def convert_string_to_float(num_str):
    if type(num_str) == float: return num_str
    num_segments = num_str.split(',')
    num = ''
    for num_segment in num_segments:
        num += num_segment
    return float(num)

#####################
#### Import Data ####
#####################

# read full data set
def load_data(filename, nrows=None):
    bikeshare = pd.read_csv(filename, sep=',', \
            parse_dates=['start_time','end_time'], nrows=nrows)
    # convert tripduration string to float
    bikeshare['tripduration'] = \
            bikeshare['tripduration'].map(convert_string_to_float)
    # trip_id, bikeid, end_time can be dropped to save space
    bikeshare = bikeshare.drop(\
            columns=['trip_id', 'bikeid', 'end_time', \
                'from_station_name', 'to_station_name'])
    # convert float64 to int16/32
    bikeshare[['tripduration', 'from_station_id', 'to_station_id']] = \
    bikeshare[['tripduration', 'from_station_id', 'to_station_id']]\
        .astype({"tripduration": 'int32', \
            "from_station_id": 'int16', 'to_station_id': 'int16'})
    return bikeshare


def preprocess(df):

    def _set_age(usertype_birthyear):
        usertype = usertype_birthyear[0]
        birthyear = usertype_birthyear[1]
        if usertype == 'Customer':
            return 0
        else:
            return 2019 - birthyear

    def set_duration(tripduration):
        return float(tripduration.replace(',', '')) / 3600

    df['age'] = pd.DataFrame(\
            map(_set_age, df[['usertype', 'birthyear']].values), index=df.index)
    df['dayinweek'] = pd.Series(\
            list(df['start_time'].map(lambda x: x.dayofweek)), index=df.index)
    df['hourinday'] = pd.Series(\
            list(df['start_time'].map(lambda x: x.hour)), index=df.index)
    df['duration'] = pd.Series(\
            list(df['tripduration'].map(lambda x: x / 3600)), index=df.index)

    return df


def user_composition(df):
    global NUM_FIGURE
    df_subscriber = df[df['usertype'] == 'Subscriber']
    df_customer = df[df['usertype'] == 'Customer']
    subscriber_ratio = df_subscriber.shape[0] / df.shape[0]
    customer_ratio = df_customer.shape[0] / df.shape[0]
    male_ratio = np.sum(df_subscriber['gender'] == 'Male') / df.shape[0]
    female_ratio = np.sum(df_subscriber['gender'] == 'Female') / df.shape[0]

    df_male_age = df[df['gender'] == 'Male']['age']
    df_female_age = df[df['gender'] == 'Female']['age']
    young_male_ratio = np.sum(df_male_age < 30) /df.shape[0]
    mid_age_male_ratio = np.sum((df_male_age >= 30) & (df_male_age < 50)) /df.shape[0]
    senior_male_ratio = np.sum(df_male_age >= 50) /df.shape[0]
    young_female_ratio = np.sum(df_female_age < 30) /df.shape[0]
    mid_age_female_ratio = np.sum((df_female_age >= 30) & (df_female_age < 50)) /df.shape[0]
    senior_female_ratio = np.sum(df_female_age >= 50) /df.shape[0]
    unknown_gender_ratio = 1 - (young_male_ratio+mid_age_male_ratio+senior_male_ratio+\
            young_female_ratio+mid_age_female_ratio+senior_female_ratio)
    print(('subscriber ratio: %.3f, customer ratio: %.3f, \nmale ratio: %.3f, ' + \
            'female ratio: %.3f, \nyoung male ratio: %.3f, mid age male ratio: %.3f, ' +\
            'senior male ratio: %.3f, \nyoung female ratio: %.3f, ' + \
            'mid age ratio: %.3f, ' +\
            'senior female ratio: %.3f') % (subscriber_ratio, customer_ratio, \
            male_ratio, female_ratio, young_male_ratio, mid_age_male_ratio, \
            senior_male_ratio, young_female_ratio, mid_age_female_ratio, \
            senior_female_ratio))


    #plt.figure(NUM_FIGURE)
    fig1, ax1 = plt.subplots()
    NUM_FIGURE += 1
    ax1.pie([male_ratio, female_ratio, customer_ratio], \
            explode=(0.1, 0, 0), labels=['Subscriber-Male', \
            'Subscriber-Female', 'Customer'], shadow=True, startangle=90,\
            autopct='%1.1f%%')
    ax1.axis('equal')
    plt.title('male female customer pie')
    my_pltsavefig('male-female-customer-pie')


    fig2, ax2 = plt.subplots()
    NUM_FIGURE += 1
    ax2.pie([young_male_ratio, mid_age_male_ratio, senior_male_ratio,\
            young_female_ratio, mid_age_female_ratio, senior_female_ratio, \
            unknown_gender_ratio],\
            labels=['young male', 'mid-age male', 'senior male', \
            'young female', 'mid-age female', 'senior female', 'unknown gender'], \
            startangle=90,\
            autopct='%1.1f%%')
    plt.title('subscriber age pie')
    my_pltsavefig('subscriber-age-pie')
    return df


def temporal_travel_pattern(df):
    global NUM_FIGURE
    df['dateonly'] = df['start_time'].dt.date
    df_subscriber = df[df['usertype'] == 'Subscriber']
    df_customer = df[df['usertype'] == 'Customer']
    df_male = df[df['gender'] == 'Male']
    df_female = df[df['gender'] == 'Female']
    num_subscriber = df_subscriber.shape[0]
    num_customer = df_customer.shape[0]

    def groupby_and_count(inner_df, selected_columns=['dateonly', 'tripduration'], \
            groupby_name=None, rename_name=None):
        if groupby_name is None: groupby_name = selected_columns[0]
        if rename_name is None: rename_name = selected_columns[1]
        return inner_df[selected_columns].groupby(groupby_name)\
                .count().rename(columns={rename_name: 'count'})

    def date_pattern(df):
        global NUM_FIGURE
        date_male = groupby_and_count(df_male)
        date_female = groupby_and_count(df_female)
        date_customer = groupby_and_count(df_customer)
        date_subscriber = groupby_and_count(df_subscriber)
        plt.figure(NUM_FIGURE)
        NUM_FIGURE += 1
        plt.plot(date_male.index, date_male, label='date by male')
        plt.plot(date_female.index, date_female, label='date by female')
        plt.plot(date_customer.index, date_customer, label='date by customer')
        plt.plot(date_subscriber.index, date_subscriber, label='date by subscriber')
        plt.xlabel('date')
        plt.ylabel('frequency')
        plt.title('date pattern by user in different categories')
        plt.gcf().autofmt_xdate()
        leg = plt.legend(loc='best', ncol=2, mode="expand", 
                shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)
        my_pltsavefig('date pattern by user in different categories')

    def week_pattern(df):
        global NUM_FIGURE
        for i in range(7):
            sr = np.sum(df_subscriber['dayinweek'] == i) / num_subscriber
            cr = np.sum(df_customer['dayinweek']== i) / num_customer
            print('day %d in week: subscriber ratio: %.3f, customer ratio: %.3f' % (\
                    i+1, sr, cr))
        week_male = groupby_and_count(df_male, ['dayinweek', 'tripduration'])
        week_female = groupby_and_count(df_female, ['dayinweek', 'tripduration'])
        week_customer = groupby_and_count(df_customer, ['dayinweek', 'tripduration'])
    
        plt.figure(NUM_FIGURE, figsize=(9, 7))
        NUM_FIGURE += 1
        title = 'week pattern by user in different categories'
        plt.subplot(211)
        b1 = plt.bar(week_male.index.values+1, week_male.values.reshape(1, -1)[0], 0.35)
        b2 = plt.bar(week_female.index.values+1, week_female.values.reshape(1, -1)[0], \
                0.35, bottom=[int(i) for i in week_male.values])
        plt.ylabel('frequency')
        plt.legend((b1, b2), ('male', 'female'))
        plt.title(title)
        plt.xticks(week_male.index.values+1, ['Mon', 'Tue', 'Week', \
                'Thu', 'Fri', 'Sat', 'Sun'])

        plt.subplot(212)
        b3 = plt.bar(week_customer.index.values+1, \
            week_customer.values.reshape(1, -1)[0], 0.35)
        plt.ylabel('frequency')
        plt.title('week pattern by Customer')
        plt.xticks(week_male.index.values+1, ['Mon', 'Tue', 'Week', \
                'Thu', 'Fri', 'Sat', 'Sun'])

        my_pltsavefig(title)

    def hour_pattern(df):
        global NUM_FIGURE
        for i in range(24):
            sr = np.sum(df_subscriber['hourinday'] == i) / num_subscriber
            cr = np.sum(df_customer['hourinday']== i) / num_customer
            print('hour %d in day subscriber ratio: %.3f, customer ratio: %.3f' % (\
                    i+1, sr, cr))
        hour_male = groupby_and_count(df_male, ['hourinday', 'tripduration'])
        hour_female = groupby_and_count(df_female, ['hourinday', 'tripduration'])
        hour_customer = groupby_and_count(df_customer, ['hourinday', 'tripduration'])
    
        plt.figure(NUM_FIGURE, figsize=(9, 7))
        NUM_FIGURE += 1
        title = 'hour pattern by user in different categories'
        plt.subplot(211)
        b1 = plt.bar(hour_male.index.values+1, hour_male.values.reshape(1, -1)[0], 0.8)
        b2 = plt.bar(hour_female.index.values+1, hour_female.values.reshape(1, -1)[0], \
                0.8, bottom=[int(i) for i in hour_male.values])
        plt.ylabel('frequency')
        plt.legend((b1, b2), ('male', 'female'))
        plt.title(title)
        plt.xticks(hour_male.index.values+1, [(str(i) + ':00') for i in range(0, 24)],\
                fontsize='x-small')

        plt.subplot(212)
        b3 = plt.bar(hour_customer.index.values+1, \
            hour_customer.values.reshape(1, -1)[0], 0.8)
        plt.ylabel('frequency')
        plt.title('hour pattern by Customer')
        plt.xticks(hour_male.index.values+1, [(str(i) + ':00') for i in range(0, 24)],\
                fontsize='x-small')
        my_pltsavefig(title)

    def duration_pattern(df):
        global NUM_FIGURE
        df_sub_duration = df_subscriber['duration']
        df_cus_duration = df_customer['duration']
        sub_half_hour_ratio = np.sum(df_sub_duration < 0.5) / num_subscriber
        cus_half_hour_ratio = np.sum(df_cus_duration < 0.5) / num_customer
        sub_half21_ratio = \
                np.sum((df_sub_duration >= 0.5) & (df_sub_duration < 1)) / num_subscriber
        cus_half21_ratio = \
                np.sum((df_cus_duration >= 0.5) & (df_cus_duration < 1)) / num_customer
        sub_122_ratio = \
                np.sum((df_sub_duration >= 1) & (df_sub_duration < 2)) / num_subscriber
        cus_122_ratio = \
                np.sum((df_cus_duration >= 1) & (df_cus_duration < 2)) / num_customer
        sub_225_ratio = \
                np.sum((df_sub_duration >= 2) & (df_sub_duration < 5)) / num_subscriber
        cus_225_ratio = \
                np.sum((df_cus_duration >= 2) & (df_cus_duration < 5)) / num_customer
        sub_5210_ratio = \
                np.sum((df_sub_duration >= 5) & (df_sub_duration < 10)) / num_subscriber
        cus_5210_ratio = \
                np.sum((df_cus_duration >= 5) & (df_cus_duration < 10)) / num_customer
        sub_10_ratio = \
                np.sum(df_sub_duration >= 10) / num_subscriber
        cus_10_ratio = \
                np.sum(df_cus_duration >= 10) / num_customer
        print('duration < 0.5h: subscriber: %.3f, customer: %.3f' % \
                (sub_half_hour_ratio, cus_half_hour_ratio))
        print('duration [0.5h, 1h]: subscriber: %.3f, customer: %.3f' % \
                (sub_half21_ratio, cus_half21_ratio))
        print('duration [1h, 2h]: subscriber: %.3f, customer: %.3f' % \
                (sub_122_ratio, cus_122_ratio))
        print('duration [2h, 5h]: subscriber: %.3f, customer: %.3f' % \
                (sub_225_ratio, cus_225_ratio))
        print('duration [5h, 10h]: subscriber: %.3f, customer: %.3f' % \
                (sub_5210_ratio, cus_5210_ratio))
        print('duration > 10h: subscriber: %.3f, customer: %.3f' % \
                (sub_10_ratio, cus_10_ratio))
        duration_group_sub = [sub_half_hour_ratio, sub_half21_ratio, sub_122_ratio,\
                sub_225_ratio, sub_5210_ratio, sub_10_ratio]
        duration_group_cus = [cus_half_hour_ratio, cus_half21_ratio, cus_122_ratio,\
                cus_225_ratio, cus_5210_ratio, cus_10_ratio]
        plt.figure(NUM_FIGURE, figsize=(9, 7))
        NUM_FIGURE += 1
        title = 'tripduration pattern by user in different categories'
        plt.subplot(211)
        b1 = plt.bar(['<0.5h', '0.5-1h', '1-2h', '2-5h', '5-10h', '>10h'], \
                duration_group_sub)
        plt.ylabel('ratio')
        plt.title(title)

        plt.subplot(212)
        b3 = plt.bar(['<0.5h', '0.5-1h', '1-2h', '2-5h', '5-10h', '>10h'], \
            duration_group_cus)
        plt.ylabel('ratio')
        plt.title('tripduration pattern by Customer')
        my_pltsavefig(title)

    date_pattern(df)
    week_pattern(df)
    hour_pattern(df)
    duration_pattern(df)
    return df


def spatial_travel_pattern(df):
    df_subscriber = df[df['usertype'] == 'Subscriber']
    df_customer = df[df['usertype'] == 'Customer']
    df_subscriber_male = df_subscriber[df_subscriber['gender'] == 'Male']
    df_subscriber_female = df_subscriber[df_subscriber['gender'] == 'Female']

    def top_k_from_to_stations(df, k=10):
        df_from_to = df[['from_station_id', 'to_station_id']]
        top_from_station = \
                df_from_to.groupby(['from_station_id']).count()\
                .rename(columns={'to_station_id': 'count'})[['count']]\
                .sort_values(by=['count'], ascending=False).iloc[:k]
        top_to_station = \
                df_from_to.groupby(['to_station_id']).count()\
                .rename(columns={'from_station_id': 'count'})[['count']]\
                .sort_values(by=['count'], ascending=False).iloc[:k]
        station_pair = pd.DataFrame()
        station_pair['pair'] = tuple(\
                zip(df_from_to['from_station_id'], df_from_to['to_station_id']))
        station_pair['tripduration'] = df['tripduration']
        station_pair = \
                station_pair.groupby(['pair']).count()\
                .rename(columns={'tripduration': 'count'})\
                .sort_values(by=['count'], ascending=False).iloc[:k]
        return top_from_station, top_to_station, station_pair


    top_from, top_to, sp = top_k_from_to_stations(df_subscriber_male)
    top_from2, top_to2, sp2 = top_k_from_to_stations(df_subscriber_female)
    top_from3, top_to3, sp3 = top_k_from_to_stations(df_customer)
    print('subscriber male top from: \n{}\n top to: \n{}\n pair:\n{}'.format(\
            top_from, top_to, sp))
    print('subscriber female top from: \n{}\n top to: \n{}\n pair:\n{}'.format(
        top_from2, top_to2, sp2))
    print('customer top from: \n{}\n top to: \n{}\n pair:\n{}'.format(
        top_from3, top_to3, sp3))
    return df


def show_df_stat(bikeshare):
    stations_id_list = bikeshare['from_station_id'].unique()
    num_stations = len(stations_id_list)
    print("Num of Stations:", num_stations)
    # Dates
    start_date = bikeshare['start_time'].min()
    end_date = bikeshare['start_time'].max()
    print("Start Date:", start_date, "\nEnd Date:", end_date)
    # Num of Total Records
    num_records_before_process = bikeshare.shape[0]
    num_subscriber_records_before_process = \
            bikeshare[bikeshare['usertype']=='Subscriber'].shape[0]
    num_customer_records_before_process = \
            bikeshare[bikeshare['usertype']=='Customer'].shape[0]
    print("Num of Total Records:", num_records_before_process)
    print("Num of Subscriber Records:", num_subscriber_records_before_process, \
            ", Ratio:", num_subscriber_records_before_process/num_records_before_process)
    print("Num of Customer Records:", num_customer_records_before_process, \
            ", Ratio:", num_customer_records_before_process/num_records_before_process)
    num_male_before_process = bikeshare.loc[\
            (bikeshare['usertype']=='Subscriber') & (bikeshare['gender']=='Male')].shape[0]
    num_female_before_process = bikeshare.loc[\
            (bikeshare['usertype']=='Subscriber') & (bikeshare['gender']=='Female')].shape[0]
    print("Num of Male Records:", num_male_before_process, \
            "Ratio in Subscriber:", num_male_before_process/num_subscriber_records_before_process)
    print("Num of Female Records:", num_female_before_process, \
            "Ratio in Subscriber:", num_female_before_process/num_subscriber_records_before_process)
    print("-----\nNote: Total ratio of gender smaller than 1, some subscribers do not have gender filled!")

# show_df_stat(bikeshare)

# Age Distribution
def compute_age(birthyear):
    # did not check legal value of birthyear.
    return 2019 - birthyear

# Convert birthyear to age, drop birthyear
def set_age(bikeshare):
    bikeshare['age'] = pd.Series(\
            map(compute_age, bikeshare['birthyear']), index=bikeshare.index)
    bikeshare = bikeshare.drop(columns=['birthyear'])
    return bikeshare

# bikeshare = set_age(bikeshare)

def show_info(bikeshare):
    ages_list = bikeshare['age'].unique()
    ages_list = ages_list[~np.isnan(ages_list)]
    max_age = max(ages_list)
    min_age = min(ages_list)
    print("Max Age:", max_age)
    print("Min Age:", min_age)
    # Show distribution in <=30, 30-50, >=50
    num_young_before_process = bikeshare[(bikeshare['age']<=30)].shape[0]
    num_mid_before_process = bikeshare[(bikeshare['age']<50) & (bikeshare['age']>30)].shape[0]
    num_old_before_process = bikeshare[bikeshare['age']>=50].shape[0]
    num_subscriber_records_before_process = bikeshare[bikeshare['usertype']=='Subscriber'].shape[0]
    print("Num Of Young:", num_young_before_process, ", Ratio in Subscribers:", num_young_before_process/num_subscriber_records_before_process)
    print("Num Of Mid:", num_mid_before_process, ", Ratio in Subscribers:", num_mid_before_process/num_subscriber_records_before_process)
    print("Num Of Old:", num_old_before_process, ", Ratio in Subscribers:", num_old_before_process/num_subscriber_records_before_process)
    print("-----\nNOTE: Total ratio exceeds 1, some customer type user also have birthyear filled!")
    # Trip Duration
    tripdurations_list = bikeshare['tripduration'].unique()
    max_duration = max(tripdurations_list)
    min_duration = min(tripdurations_list)
    print("Max Duration:", max_duration)
    print("Min Duration:", min_duration)
    
# show_info(bikeshare)

# Compute Duration Distribution: try to make it evenly distribued
def compute_hours(sec):
    return sec / 3600

def convert_time_slice(hours):
    if hours < 0.1:
        return 0
    if hours < 0.2:
        return 1
    if hours >= 0.2 and hours < 0.4:
        return 2
    if hours >= 0.4 and hours < 1:
        return 3
    if hours >= 1 and hours < 3:
        return 4
    if hours >=3 and hours < 5:
        return 5
    if hours >= 5:
        return 6
    

def set_hours(bikeshare):
    bikeshare['hours'] = pd.Series(map(lambda s:convert_time_slice(compute_hours(s)), \
                                       bikeshare['tripduration']), index=bikeshare.index)
    #bikeshare.groupby(['hours'])['hours'].count()
    return bikeshare


def drop_outlier_cluster(bikeshare):
    # Method 1: drop value exceeds 5 hr
    #bikeshare = bikeshare[bikeshare['hours'] != 6]
    # Method2 (not adapted): for the same start and end id, compute average duration, remove outlier using statistical method (Gaussian Distribution Model)
    bikeshare_group_start_end_df = \
            bikeshare.groupby(['from_station_id', 'to_station_id'])[['tripduration']]
    bikeshare_group_stat_df = bikeshare_group_start_end_df.mean()
    bikeshare_group_stat_df['std'] = bikeshare_group_start_end_df.std()
    bikeshare_group_stat_df['count'] = bikeshare_group_start_end_df.count()
    bikeshare_group_stat_df.head(5)
    # result not very good due to very large std caused by greatly off-track outliers

    # Method 3: Clustered-based, use small amount of data to do clustering
    # a. cluester on smaller data set
    bikeshare_subset = bikeshare.sample(frac=0.1, replace=False) # random sample 1% without replacement
    X = [[bikeshare_subset['tripduration'].iloc[i]] for i in range(bikeshare_subset.shape[0])] # extract tripduration
    kmeans_subset = KMeans(n_clusters=35, random_state=0, algorithm="full").fit(X) # since we slice time into 7 segments, no need too many centers

    # b. count members of each centroid
    bikeshare_subset['centroid'] = kmeans_subset.labels_
    members_count = bikeshare_subset.groupby('centroid')['centroid'].count()
    threshold = 0.05*max(members_count)
    # c. keep those center has member count greater than threshold
    normal_center = [i for i,count in enumerate(members_count) if count > threshold]

    # d. scan through all data, drop those records which are clustered to abnormal center
    predict_data = [[bikeshare['tripduration'].iloc[i]] for i in range(bikeshare.shape[0])]
    predict_center = kmeans_subset.predict(predict_data)
    drop_result = [ind for ind, c in enumerate(predict_center) if c not in normal_center]
    bikeshare = bikeshare.drop(bikeshare.index[drop_result])

    # Check outlier removal result
    bikeshare_group_start_end_df = bikeshare.groupby(['from_station_id', 'to_station_id'])[['tripduration']]
    bikeshare_group_stat_df = bikeshare_group_start_end_df.mean()
    bikeshare_group_stat_df['std'] = bikeshare_group_start_end_df.std()
    bikeshare_group_stat_df['count'] = bikeshare_group_start_end_df.count()
    bikeshare_group_stat_df.head(5)
    return bikeshare
    # std better much better than before

# bikeshare = drop_outlier_cluster(bikeshare)

def map_user_type(t):
    if t == 'Subscriber':
        return [1, 0]
    if t == 'Customer':
        return [0, 1]
    return [0, 0] # nan

def map_gender(g):
    if g == 'Male':
        return [1, 0]
    if g == 'Female':
        return [0, 1]
    return [0, 0]

def map_age(a):
    # return a list of [young, mid, old]
    if a <= 30:
        return [1,0,0]
    if a <= 50:
        return [0,1,0]
    if a > 50:
        return [0,0,1]
    else:
        return [0,0,0] # nan

def map_datetime(datetime): # Mon-0, Sun-6
    # return a list of [Mon, .., Sun, 0am-1am, 1am-2am, ..., 11pm-12pm]
    weekday = datetime.dayofweek
    weekdays_list = [0 for i in range(7)]
    weekdays_list[weekday] = 1
    hour = datetime.hour
    hours_list = [0 for i in range(24)]
    hours_list[hour] = 1
    return weekdays_list+hours_list

def map_tripduration(sec):
    hours_list = [0 for i in range(7)]
    hours = sec / 3600
    if hours < 0.1:
        hours_list[0]=1
        return hours_list
    if hours < 0.2:
        hours_list[1]=1
        return hours_list
    if hours >= 0.2 and hours < 0.4:
        hours_list[2]=1
        return hours_list
    if hours >= 0.4 and hours < 1:
        hours_list[3]=1
        return hours_list
    if hours >= 1 and hours < 3:
        hours_list[4]=1
        return hours_list
    if hours >=3 and hours < 5:
        hours_list[5]=1
        return hours_list
    if hours >= 5: # this shall all be 0, since we drop this column during preprocessing
        hours_list[6]=1
        return hours_list
        
def convert_df(df):
    df[['subscriber', 'customer']] = pd.DataFrame(list(df['usertype'].map(map_user_type)), index=df.index)
    df[['male', 'female']] = pd.DataFrame(list(df['gender'].map(map_gender)), index=df.index)
    df[['young','mid','old']] = pd.DataFrame(list(df['age'].map(map_age)), index=df.index)
    df[['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',\
         '0','1','2','3','4','5','6','7','8','9','10','11','12',\
         '13','14','15','16','17','18','19','20','21','22','23']] = \
    pd.DataFrame(list(df['start_time'].map(map_datetime)), index=df.index)
    df[['d0','d1','d2','d3','d4','d5','d6']] = pd.DataFrame(\
        list(df['tripduration'].map(map_tripduration)), index=df.index)
    df = df.drop(columns=['usertype', 'gender', 'age', 'start_time','hours', 'tripduration'])
    return df

###### Method 2: extract as numerical result
# subscriber: 1, customer -1
# male subscriber: 1, female subscriber: -1, otherwise (male or female without subscribtion): 0
# subscriber age: keep value, otherwise 0
# weekday trip time: 1-7 for mon to sun
# hour of trip time: 0am-1am-0, 23pm-0am-23
# tripduration: remain the same
# from_station_id: remain the same
# to_station_id: remain the same

rush_hours = None
rush_hours_count = None


def map_user_type_numerical(t):
    if t == 'Subscriber':
        return 1
    if t == 'Customer':
        return -1
    return 0


def map_gender_numerical(record): # record = [usertype,gender]
    if record[0] != 1:
        return 0
    if record[1] == 'Male':
        return 1
    if record[1] == 'Female':
        return -1
    return 0 # although is subscriber, no record


def map_age_numerical(record): # [usertype, age]
    if record[0] != 1 or np.isnan(record[1]):
        return 0
    return record[1]


def map_week_numerical(datetime): # Mon-1, Sun-7
    return datetime.dayofweek+1


def map_hour_numerical(datetime): # 0am-1am, 1am-2am, ..., 11pm-12pm: 0, 1, .., 23
    return datetime.hour


def map_tripduration_numerical(sec):
    hours_list = [0 for i in range(7)]
    hours = sec / 3600
    if hours < 0.1:
        hours_list[0]=1
        return hours_list
    if hours < 0.2:
        hours_list[1]=1
        return hours_list
    if hours >= 0.2 and hours < 0.4:
        hours_list[2]=1
        return hours_list
    if hours >= 0.4 and hours < 1:
        hours_list[3]=1
        return hours_list
    if hours >= 1 and hours < 3:
        hours_list[4]=1
        return hours_list
    if hours >=3 and hours < 5:
        hours_list[5]=1
        return hours_list
    if hours >= 5: # this shall all be 0, since we drop this column during preprocessing
        hours_list[6]=1
        return hours_list


def map_weekend_numerical(w):
    if w == 6 or w == 7:
        return 1
    return 0


def map_rushhour_numerical(h):
    global rush_hours,rush_hours_count
    return rush_hours_count[list(rush_hours).index(h)]


def map_rush_from_station_numerical(from_station_id):
    global rush_from_station_count, rush_from_station
    return rush_from_station_count[list(rush_from_station).index(from_station_id)]


def convert_df_numerical(df):
    global rush_hours,rush_hours_count
    df[['usertype']] = pd.DataFrame(list(df['usertype'].map(map_user_type_numerical)), index=df.index)
    df[['gender']] = pd.DataFrame(map(map_gender_numerical,df[['usertype','gender']].values), index=df.index)
    df[['age']] = pd.DataFrame(map(map_age_numerical, df[['usertype','age']].values), index=df.index)
    df[['week']] = pd.DataFrame(list(df['start_time'].map(map_week_numerical)), index=df.index)
    df[['hour']] = pd.DataFrame(list(df['start_time'].map(map_hour_numerical)), index=df.index)
    rush_hours_flow = bikeshare.groupby('hour').count()\
        .rename(columns={'tripduration':'count'}).sort_values('count', ascending=False)
    rush_hours = rush_hours_flow.index.values
    rush_hours_count = rush_hours_flow['count'].values
    df[['weekend']] = pd.DataFrame(list(df['week'].map(map_weekend_numerical)), index=df.index)
    df[['rushhour']] = pd.DataFrame(list(df['hour'].map(map_rushhour_numerical)), index=df.index)
    df[['rushfrom']] = pd.DataFrame(list(df['from_station_id'].map(\
        map_rush_from_station_numerical)), index=df.index)
    df = df.drop(columns=['hours','start_time'])
    return df

# bikeshare = convert_df_numerical(bikeshare)


def inflate_df_numerical(df):
    df[['from_station_id']] = df['from_station_id'].map(lambda x: x*5)
    df[['weekend']] = df['weekend'].map(lambda x: x*5)
    df[['gender']] = df['gender'].map(lambda x: x*10)
    df[['usertype']] = df['usertype'].map(lambda x: x*10)
    df[['tripduration']] = df['tripduration'].map(lambda x: x/10)
    df[['rushfrom']] = df['rushfrom'].map(lambda x: x/100)
    df[['rushhour']] = df['rushhour'].map(lambda x: x/1000)
    return df


def inflate_df(df):
    df[['from_station_id']] = df['from_station_id'].map(lambda x: x*5)
    df[['male']] = df['male'].map(lambda x: x*10)
    df[['female']] = df['female'].map(lambda x: x*10)
    df[['subscriber']] = df['subscriber'].map(lambda x: x*10)
    df[['customer']] = df['customer'].map(lambda x: x*10)


# bikeshare = inflate_df_numerical(bikeshare)

########################
#### Model Training ####
########################

####### I. Unsupervised KNN: use to_station_id as label, the rest are attributes #######
knn_classifier = None
knn_y_predict = None

# I.a. Split vector x and label y
def flatten_2D_arr(array_2d):
    return [i for sublist in array_2d for i in sublist]


def unsupervised_attributes_label_arr(df): # convert dataframe into X and y array for unsupervised learning
    y = df[['to_station_id']].values
    X = df.drop(columns=['to_station_id']).values
    return X, y


# I.b. KNN model

def KNN_model(X_train, y_train, X_test, y_test, k=5):
    global knn_classifier, knn_accuracy
    if knn_classifier is None:
        knn_classifier = KNeighborsClassifier(n_neighbors=k, weights='distance')  
    knn_classifier.fit(X_train, y_train)
    knn_neighbors = knn_classifier.kneighbors(X_test)
    knn_y_predict = knn_classifier.predict(X_test)
    knn_accuracy = sum(knn_y_predict == y_test) / len(y_test)
    return knn_y_predict, knn_accuracy


rush_from_station = None
rush_from_station_count = None
bikeshare = None


def run(train_file_name, test_file_name):
    global bikeshare, rush_from_station, rush_from_station_count
    bikeshare = load_data(train_file_name)
    show_df_stat(bikeshare)
    bikeshare = set_age(bikeshare)
    show_info(bikeshare)
    bikeshare = set_hours(bikeshare)
    from_station_flow = bikeshare[['tripduration','from_station_id']]\
        .groupby(['from_station_id']).count().rename(columns={'tripduration':'count'})\
        .sort_values('count', ascending=False)
    # max_from_flow = max(from_station_flow.values)
    rush_from_station = from_station_flow.index.values
    rush_from_station_count = from_station_flow['count'].values
    bikeshare = drop_outlier_cluster(bikeshare)
    bikeshare = convert_df_numerical(bikeshare)
    bikeshare = inflate_df_numerical(bikeshare)

    bikeshare = bikeshare.replace([np.inf, -np.inf], np.nan)\
            .dropna(subset=bikeshare.columns.values.tolist())
    bikeshare_X, bikeshare_y = unsupervised_attributes_label_arr(bikeshare)
    bikeshare_train_X_un, bikeshare_test_X_un, \
        bikeshare_train_y_un, bikeshare_test_y_un = train_test_split(
        bikeshare_X, bikeshare_y, test_size=0.33, random_state=0)

    attributes_list = bikeshare.columns.values.tolist()
    attributes_list.remove('to_station_id')
    print("Attributes:", attributes_list)

    k_set = [i for i in range(4, 87, 7)]
    tmp_acc = []
    for _k in k_set:
        tmp_train_y, tmp_test_y = flatten_2D_arr(bikeshare_train_y_un), flatten_2D_arr(bikeshare_test_y_un)
        tmp_y_predict, tmpa = KNN_model(bikeshare_train_X_un, tmp_train_y, bikeshare_test_X_un, tmp_test_y, _k)
        print('k: %d acc: %.4f' % (_k, tmpa))
        tmp_acc.append(tmpa)
    print('mean acc:', np.mean(tmp_acc))
    return 




    fold_splits = 10
    K_accuracy = [0 for i in range(fold_splits)]
    kf = KFold(n_splits=fold_splits)
    k = 12
    counter = 0
    for train_index, test_index in kf.split(bikeshare_train_X_un, bikeshare_train_y_un):
        X_train, X_test = bikeshare_train_X_un[train_index], bikeshare_train_X_un[test_index]
        y_train, y_test = bikeshare_train_y_un[train_index], bikeshare_train_y_un[test_index]
        y_train, y_test = flatten_2D_arr(y_train), flatten_2D_arr(y_test)
        knn_y_predict, K_accuracy[counter] = KNN_model(X_train, y_train, X_test, y_test, k=k)
        print("Iteration", counter, "accuracy:", K_accuracy[counter])
        counter+=1

    print('avg acccuracy:', np.mean(K_accuracy))

    # I.d. plot K and Retrieve Best K
    # sometimes 5 yields better result than best-k, but why ?
    bikeshare_test = load_data(test_file_name, nrows=1000)
    bikeshare_test = set_age(bikeshare_test)
    bikeshare_test = set_hours(bikeshare_test)
    bikeshare_test = drop_outlier_cluster(bikeshare_test)
    bikeshare_test = convert_df_numerical(bikeshare_test)
    bikeshare_test = inflate_df_numerical(bikeshare_test)
    # bikeshare_test.info()
    # bikeshare_test.head(5)
    bikeshare_test_X, bikeshare_test_y = unsupervised_attributes_label_arr(bikeshare_test)
    knn_y_predict, knn_accuracy = KNN_model(bikeshare_X, flatten_2D_arr(bikeshare_y), \
                             bikeshare_test_X, flatten_2D_arr(bikeshare_test_y), k)
    print("KNN accuracy:", knn_accuracy,", with K chosen as:", k)
    return knn_y_predict


def main(argv):
    trainfile = './data_set1/training.csv'
    testfile = './data_set1/groundtruth1.csv'
    if len(argv) == 1:
        return
    elif argv[1] == 'train':
        my_y_predict = run(trainfile, testfile)
    elif argv[1] == 'show':
        df = load_data(trainfile)
        df = preprocess(df)
        user_composition(df)
        temporal_travel_pattern(df)
        spatial_travel_pattern(df)
        plt.show()
    else:
        print('unknown parameter.')


if __name__ == '__main__':
    main(sys.argv)



