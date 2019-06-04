import pandas as pd
from sklearn import linear_model, ensemble, preprocessing
import itertools
import datetime as dt
from patsy import dmatrices
import matplotlib.pyplot as plt

def mean_absolute_error(train, test):
    abs_errors = abs(train - test)
    return (sum(abs_errors)/abs_errors.count())

class LinearRegression:
    def __init__(self, file):


        self.data = pd.read_csv(file, '\t')

        # ljubljanski maraton
        marathon_date = dt.datetime.strptime("28.10.2012", "%d.%m.%Y").date()
        self.data = self.data[pd.to_datetime(self.data['Departure time']).dt.date != marathon_date]

        # prvi maj
        may = dt.datetime.strptime("1.5.2012", "%d.%m.%Y").date()
        self.data = self.data[pd.to_datetime(self.data['Departure time']).dt.date != may]


        # summer_start = dt.datetime.strptime("15.7.2012", "%d.%m.%Y").date()
        # summer_end = dt.datetime.strptime("25.8.2012", "%d.%m.%Y").date()
        # date = pd.to_datetime(self.data['Departure time']).dt.date
        # self.data = self.data[(date < summer_start) | (date > summer_end)]


        self.data['months'] = pd.to_datetime(self.data['Departure time']).dt.month
        self.data['year_quarters'] = pd.to_datetime(self.data['Departure time']).dt.quarter
        # self.data['year_quarters_sqared'] = self.data['year_quarters'] ** 2
        self.data['hour_whole'] = pd.to_datetime(self.data['Departure time']).dt.round('H')
        self.data['hour_whole'] = self.data['hour_whole'].dt.strftime('%H')
        self.data['hours'] = pd.to_datetime(self.data['Departure time']).dt.round('15min')
        self.data['hours'] = self.data['hours'].dt.strftime('%H:%M')
        self.data['weekdays'] = pd.to_datetime(self.data['Departure time']).dt.weekday
        self.data['weekend'] = ((self.data['weekdays'] == 5) | (self.data['weekdays'] == 6)).astype(int)

        # year seasons
        self.data['spring'] = ((self.data['months'] == 3) | (self.data['months'] == 4) | (self.data['months'] == 4))
        self.data['summer'] = ((self.data['months'] == 6) | (self.data['months'] == 7) | (self.data['months'] == 8))
        self.data['fall'] = ((self.data['months'] == 9) | (self.data['months'] == 10) | (self.data['months'] == 11))
        self.data['winter'] = ((self.data['months'] == 12) | (self.data['months'] == 1) | (self.data['months'] == 2))



        self.data['bus_freq'] = pd.to_datetime(self.data['Departure time']).apply(lambda x: len(self.data[pd.to_datetime(self.data['Departure time']).isin(
            pd.date_range(x - pd.Timedelta(2500, 's'), x + pd.Timedelta(2500, 's'), freq='s'))]))

        school_free_dates = ["1.1.2012", "2.1.2012", "20.2.2012", "26.2.2012", "27.4.2012", "2.5.2012", "25.6.2012",
                      "31.8.2012", "29.10.2012", "2.11.2012", "24.12.2012", "31.12.2012", "9.4.2012", "8.2.2012"]

        school_hol = [dt.datetime.strptime(x, "%d.%m.%Y").date() for x in school_free_dates]

        # college_holidays = ["10.7.2012", "30.9.2012", "24.12.2012", "31.12.2012"]
        #
        # college_hol = [dt.datetime.strptime(x, "%d.%m.%Y").date() for x in college_holidays]

        self.data['school_holidays'] = ((pd.to_datetime(self.data['Departure time']).dt.date >= school_hol[0]) & (pd.to_datetime(self.data['Departure time']).dt.date <= school_hol[1])) | \
                                       ((pd.to_datetime(self.data['Departure time']).dt.date >= school_hol[2]) & (pd.to_datetime(self.data['Departure time']).dt.date <= school_hol[3])) | \
                                       ((pd.to_datetime(self.data['Departure time']).dt.date >= school_hol[4]) & (pd.to_datetime(self.data['Departure time']).dt.date <= school_hol[5])) | \
                                       ((pd.to_datetime(self.data['Departure time']).dt.date >= school_hol[6]) & (pd.to_datetime(self.data['Departure time']).dt.date <= school_hol[7])) | \
                                       ((pd.to_datetime(self.data['Departure time']).dt.date >= school_hol[8]) & (pd.to_datetime(self.data['Departure time']).dt.date <= school_hol[9])) | \
                                       ((pd.to_datetime(self.data['Departure time']).dt.date >= school_hol[10]) & (pd.to_datetime(self.data['Departure time']).dt.date <= school_hol[11])) | \
                                       (pd.to_datetime(self.data['Departure time']).dt.date == school_hol[12]) | (pd.to_datetime(self.data['Departure time']).dt.date == school_hol[13])

        # self.data['college_holidays'] = ((pd.to_datetime(self.data['Departure time']).dt.date >= college_hol[0]) & (pd.to_datetime(self.data['Departure time']).dt.date <= college_hol[1])) | \
        #                                ((pd.to_datetime(self.data['Departure time']).dt.date >= college_hol[2]) & (pd.to_datetime(self.data['Departure time']).dt.date <= college_hol[3]))


        holidays = {
            dt.datetime.strptime("1.1.2012", "%d.%m.%Y").date(): True,
            dt.datetime.strptime("2.1.2012", "%d.%m.%Y").date(): True,
            dt.datetime.strptime("8.2.2012", "%d.%m.%Y").date(): True,
            dt.datetime.strptime("8.4.2012", "%d.%m.%Y").date(): True,
            dt.datetime.strptime("9.4.2012", "%d.%m.%Y").date(): True,
            dt.datetime.strptime("27.4.2012", "%d.%m.%Y").date(): True,
            dt.datetime.strptime("1.5.2012", "%d.%m.%Y").date(): True,
            dt.datetime.strptime("2.5.2012", "%d.%m.%Y").date(): True,
            dt.datetime.strptime("31.5.2012", "%d.%m.%Y").date(): False,
            dt.datetime.strptime("25.6.2012", "%d.%m.%Y").date(): True,
            dt.datetime.strptime("15.8.2012", "%d.%m.%Y").date(): True,
            dt.datetime.strptime("31.10.2012", "%d.%m.%Y").date(): True,
            dt.datetime.strptime("1.11.2012", "%d.%m.%Y").date(): True,
            dt.datetime.strptime("25.12.2012", "%d.%m.%Y").date(): True,
            dt.datetime.strptime("26.12.2012", "%d.%m.%Y").date(): True,
        }


        self.data['holidays'] = pd.to_datetime(self.data['Departure time']).dt.date.apply(lambda x: holidays.get(x,False))


        # self.data['rush_hours'] = (5 == self.data['hours']) | (6 == self.data['hours']) | (self.data['hours'] == 9) |\
        #                           ((11 < self.data['hours']) & (self.data['hours'] < 15))

        # self.data['hours2'] = self.data['hours'] ** 2
        # self.data['hours3'] = self.data['hours'] ** 3
        # self.data['hours4'] = self.data['hours'] ** 4
        # self.data['hours5'] = self.data['hours'] ** 5

        # self.data['weekdays2'] = self.data['weekdays'] ** 2


        if file=='train_pred.csv':
            self.data['time'] = (pd.to_datetime(self.data['Arrival time']) - pd.to_datetime(self.data['Departure time'])).dt.seconds


            # q1 = self.data['time'].quantile(.99)
            # # q2 = self.data['time'].quantile(.01)
            #
            # self.data = self.data[(self.data['time'] < q1)]

        # self.data.boxplot(column='time', by='holidays')
        # plt.show()

        unique_hours = self.data['hours'].unique().tolist()
        unique_weekdays = self.data['weekdays'].unique().tolist()
        unique_weekdays += ['weekend']
        unique_quarters = self.data['year_quarters'].unique().tolist()


        # self.data = pd.get_dummies(self.data, columns=['hours', 'weekdays', 'year_quarters'])  # one hot encoding

        # for h,w in itertools.product(unique_hours, unique_weekdays):
        #     if(w == 'weekend'):
        #         self.data["hours_{0}_{1}".format(h, w)] = self.data['hours_' + str(h)] * self.data[str(w)]
        #     else:
        #         self.data["hours_{0}_weekday_{1}".format(h,w)] = self.data['hours_'+str(h)] * self.data['weekdays_'+str(w)]
        #
        # for h in unique_hours:
        #     self.data["hours_{0}_holiday".format(h)] = self.data['hours_'+str(h)] * self.data['holidays']
        # #
        # # for h in unique_quarters:
        # #     self.data["year_quarters_{0}_weekend".format(h)] = self.data['year_quarters_' + str(h)] * self.data['weekend']
        #
        #
        #
        # self.data['weekend_and_holiday'] = self.data['weekend'] * self.data['holidays']
        #
        # self.data['weekend_and_spring'] = self.data['weekend'] * self.data['spring']
        # self.data['weekend_and_summer'] = self.data['weekend'] * self.data['summer']
        # self.data['weekend_and_fall'] = self.data['weekend'] * self.data['fall']
        # self.data['weekend_and_winter'] = self.data['weekend'] * self.data['winter']





        # to_intersect = [x for x in self.data.columns.values if 'hours_' in x]
        # to_intersect += [x for x in self.data.columns.values if 'weekdays_' in x]
        # to_intersect += ['weekend']
        #
        # inter = preprocessing.PolynomialFeatures(3,interaction_only=True, include_bias=False)
        #
        # intersected = inter.fit_transform(self.data[to_intersect].to_sparse())
        #
        # names = inter.get_feature_names(to_intersect)
        #
        # intersected = pd.DataFrame(intersected, columns=names)
        #
        # intersected.drop(to_intersect, axis=1, inplace=True)
        #
        #
        #
        # self.data = pd.concat([self.data, intersected.loc[:, (intersected==1).any()]], axis=1)


        # self.data[x] = [intersected[x] for x in names if (intersected[x] != 0)]


            # print(self.data[['Departure time', 'weekend', 'weekdays']].head())
        # print(self.data['time'].head())
        # print(self.data['weekdays'].unique())




train = LinearRegression('train_pred.csv')

# slow drivers and slow buses
g = train.data.groupby('Driver ID').time.median().reset_index()
slow_drivers = g[g.time > 2100]['Driver ID'].tolist()
g = train.data.groupby('Registration').time.median().reset_index()
slow_bus = g[g.time > 2100]['Registration'].tolist()

train.data['slow_driver'] = train.data['Driver ID'].isin(slow_drivers)
train.data['slow_bus'] = train.data['Registration'].isin(slow_bus)

# cross

# MAEs = []
#
# for x in range(1,12):
#     print(x)
#
#     lr = linear_model.Ridge(alpha=.001)
#
#     # training = train.data.loc[train.data['months'] != x]
#     # testing = train.data.loc[train.data['months'] == x]
#
#     joined = train.data
#     joined['foo'] = 0
#     _, X = dmatrices(
#         'foo ~ C(hours) * C(weekdays) + C(hours) * C(holidays) + C(weekend) * C(holidays) + C(slow_bus) + C(slow_driver) + C(year_quarters) + C(school_holidays) + C(bus_freq)',
#         joined, return_type='dataframe')
#
#     X_train = X.loc[joined['months'] != x]
#     y_train = train.data['time']
#
#     X_test = X.loc[joined['months'] == x]
#
#     lr.fit(X_train, y_train)
#
#     y_test = lr.predict(X_test)
#
#     X_test['Arrival time'] = pd.to_datetime(train.data.loc[train.data['months'] == x]['Departure time']) + pd.to_timedelta(y_test, 's')
#
#     mae = mean_absolute_error(y_train, y_test)
#     print(mae)
#     MAEs.append(mae)
#
# avg_mae = sum(MAEs)/len(MAEs)
#
# print(avg_mae)
#
# exit()




test = LinearRegression('test_pred.csv')


test.data['slow_driver'] = test.data['Driver ID'].isin(slow_drivers)
test.data['slow_bus'] = test.data['Registration'].isin(slow_bus)

joined = train.data.append(test.data)
joined['foo'] = 0
_,X = dmatrices('foo ~ C(hours) * C(weekdays) * C(holidays) * C(year_quarters) + C(hours) * C(holidays) + C(weekend) * C(holidays) + C(slow_bus) + C(slow_driver) + C(year_quarters) + C(school_holidays) + C(bus_freq)', joined, return_type='dataframe')


X_train = X.loc[joined['Arrival time'] != '?']
y_train = train.data['time']

X_test = X.loc[joined['Arrival time'] == '?']


# features = [x for x in train.data.columns.values if 'hours_' in x]
# features += [x for x in train.data.columns.values if 'weekdays_' in x]
# features += [x for x in train.data.columns.values if 'year_quarters_' in x]
# # features += [x for x in train.data.columns.values if 'bus_freq_' in x]
#
#
# features += ['weekend', 'holidays', 'school_holidays', 'weekend_and_holiday', 'slow_driver', 'slow_bus',
#              'weekend_and_spring', 'weekend_and_summer', 'weekend_and_fall', 'weekend_and_winter']

# MAEs = []
#
# for x in range(1,12):
#     # if ((x == 7) | (x==8)):
#     #     continue
#
#     # lr = ensemble.RandomForestRegressor(n_estimators=300, n_jobs=-1)
#     # lr = linear_model.ElasticNet(random_state=1, alpha=.01, l1_ratio=.8)
#     lr = linear_model.Ridge(alpha=.001)
#     training = train.data.loc[train.data['months'] != x]
#     testing = train.data.loc[train.data['months'] == x]
#
#     lr.fit(training[features].values, training['time'].values)
#     pred = lr.predict(testing[features].values)
#
#     mae = mean_absolute_error(testing['time'], pred)
#     MAEs.append(mae)
#
# avg_mae = sum(MAEs)/len(MAEs)
#
# print(avg_mae)
#
# exit()


# lr = linear_model.LinearRegression()
# lr = ensemble.RandomForestRegressor(n_estimators=1000, n_jobs=-1)
lr = linear_model.Ridge(alpha=.001)

# test.data['hours_03:00'] = 0
# test.data['hours_03:15'] = 0
# test.data['hours_03:30'] = 0
# test.data['hours_04:15'] = 0
# test.data['hours_04:45'] = 0
# test.data['hours_13:45'] = 0
# test.data['hours_14:00'] = 0
# test.data['hours_19:30'] = 0
# test.data['hours_20:00'] = 0
# test.data['hours_20:15'] = 0
#
# test.data['year_quarters_1'] = 0
# test.data['year_quarters_2'] = 0
# test.data['year_quarters_3'] = 0

# train_columns = train.data.columns
# test_columns = test.data.columns
#
# diff = list(set(train_columns) - set(test_columns))
#
# test.data = pd.concat([test.data, pd.DataFrame(columns=diff)], axis=1)
# test.data[diff] = 0
#
# train.data.sort_index( axis=1, inplace=True)
# test.data.sort_index(axis=1, inplace=True)

lr.fit(X_train, y_train)

print(lr.coef_)

y_test = lr.predict(X_test)

# print ("napoved ", pred)

X_test['Arrival time'] = pd.to_datetime(test.data['Departure time']) + pd.to_timedelta(y_test, 's')

f = open('results_pred.txt', 'w')
print(X_test['Arrival time'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').to_string(index=False), file=f)
f.close()
