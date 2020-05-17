def clean_drop(data):
    """Provide data frame in parenthesis and this function will
        drop nulls permanently, and reset the index."""
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

def reset_index(data):
    """Provide data frame in parenthesis and this function will
        reset your index."""
    data.reset_index(drop=True, inplace=True)
    return data

def make_value_chart(data, column, x_label, y_label, title, color='PRGn'):
    """Must fill out all arguments EXCEPT color. Color has a default setting."""
    fig, ax = plt.subplots(figsize=(14,8))
    ax = sns.barplot(x=list(data[column].value_counts().keys()),
     y=data[column].value_counts(), palette=color)    
    ax.set_xticklabels(ax.get_xticklabels(),
                      rotation=45, horizontalalignment='right')
    plt.xlabel(x_label, size=20)
    plt.ylabel(y_label, size=20)
    plt.title(title, size=26)
    return plt.show()

def rename_column(data, column, new_name):
    """Quickly renames columns"""
    data.rename(columns={column: new_name}, inplace=True)
    return data

def drop(data, columns=[]):
    """Quickly drops columns"""
    data.drop(columns=columns, inplace=True)
    return data

def dummy_up(data, column, prefix):
    """Quickly makes dummies for the selected column"""
    new = pd.get_dummies(data[column], prefix=prefix, drop_first=True)
    return new

def factorize(data, column):
    """Factorizes categorical columns to prepare for RandomForestClassifier.  

    Must apply to each column individually."""
    new = pd.factorize(data[column])[0] + 1
    return new

def makeMarker(coordinates, mapp):
    for i in coordinates:
        mark = folium.Marker(i)
        mark.add_to(mapp)

def dftest(data):
    test = adfuller(data['value'])
    test_output = pd.Series(test[0:4], index=['Test Stat', 'P-Value', '# Lags', '# Observations'])
    for key, value in test[4].items():
        test_output['Critical Value (%s)' %key]=value
    return(test_output)

def auto_corrs(data, metro):
    fig, ax = plt.subplots(figsize=(16,3))
    acf = plot_acf(data, ax=ax, lags=48, title=metro+' ACF')
    fig, ax = plt.subplots(figsize=(16,3))
    pacf = plot_pacf(data, ax=ax, lags=48, title=metro+' PACF')
    return acf, pacf

def arima_search(data, ps, ds, qs):
    aic_scores = {}
    rmse_scores ={}
    for p in ps:
        for d in ds:
            for q in qs:
                model = ARIMA(data['2008':'2015'], order=(p,d,q))
                fit = model.fit(disp=0, transparams=True)
                aic_scores[p,d,q] = fit.aic
                avg_vals = []
                for line in list(fit.forecast(37)[2]):
                    avg_vals.append((line[1]+line[0])/2)
                mse = mean_squared_error(data['2016':], avg_vals)
                rmse_scores[p,d,q] = math.sqrt(mse)
    return ('Best ARIMA Parameters for AIC: ', min(aic_scores, key=aic_scores.get)), ('Best ARIMA Parameters for RMSE: ', min(rmse_scores, key=rmse_scores.get))

def arima(data, order, keyword):
    model = ARIMA(data['2008':'2015'], order)
    fit = model.fit(disp=0)
    model_full = ARIMA(data['2008':], order)
    fit2 = model.fit(disp=0)
    print('ARIMA for',keyword)
    print(fit.summary())
    residuals = pd.DataFrame(fit.resid)
    residuals.plot(kind='kde', title=keyword)
    fig,ax=plt.subplots(figsize=(8,6))
    fit2.plot_predict(end='2019-01-01', ax=ax)
    ax = plt.xlabel('TIME')
    ax = plt.ylabel('GTREND SCORE')
    plt.title('In Sample Predictions for %s' % (keyword))
    avg_vals = []
    for line in list(fit.forecast(37)[2]):
        avg_vals.append((line[1]+line[0])/2)
    mse = mean_squared_error(data['2016':'2019-01-01'], avg_vals)
    rmse = math.sqrt(mse)
    print('This model can predict the remaining 37 data points with an RMSE of ',rmse)
    print('The following results are predicting values post-data set')
    print('These predictions can only be confirmed with new data')
    avg_2020 = ((fit.forecast(49)[2][-1][1]+fit.forecast(49)[2][-1][0])/2)
    print(f'Indoor Farming search trends in 2020 %s'%avg_2020)
    one_year = ((avg_2020-data['indoor farming']['2008-01-01'])/data['indoor farming']['2008-01-01'])*100
    print('Percent change from 2008-01-01 to 2020-01-01: ',one_year)
    fig2,ax2 = plt.subplots(figsize=(8,6))
    fit.plot_predict(end='2020-01-01', ax=ax2)
    ax2 = plt.xlabel('TIME')
    ax2 = plt.ylabel('GTREND SCORE')
    mons = (np.datetime64('2020-01-01','M') - np.datetime64('2019-01-01','M')).tolist()
    plt.title('%s Months Out of Sample Forecast for %s'% (mons,keyword))
    avg_2021 = ((fit.forecast(61)[2][-1][1]+fit.forecast(61)[2][-1][0])/2)
    two_year = ((avg_2021-data['indoor farming']['2008-01-01'])/data['indoor farming']['2008-01-01'])*100
    print(f'Indoor Farming search trends in 2021 %s'%avg_2021)
    print('Percent change from 2008-01-01 to 2021-01-01: ',two_year)
    fig3,ax3 = plt.subplots(figsize=(8,6))
    fit.plot_predict(end='2021-01-01', ax=ax3)
    ax3 = plt.xlabel('TIME')
    ax3 = plt.ylabel('GTREND SCORE')
    mons2 = (np.datetime64('2021-01-01','M') - np.datetime64('2019-01-01','M')).tolist()
    plt.title('%s Months Out of Sample Forecast for %s'%(mons2,keyword))
    avg_2022 = ((fit.forecast(73)[2][-1][1]+fit.forecast(73)[2][-1][0])/2)
    three_year = ((avg_2022-data['indoor farming']['2008-01-01'])/data['indoor farming']['2008-01-01'])*100
    print(print(f'Indoor Farming search trends in 2022 %s'%avg_2022))
    print('Percent change from 2008-01-01 to 2022-01-01: ',three_year)
    fig4, ax4 = plt.subplots(figsize=(8,6))
    fit.plot_predict(end='2022-01-01', ax=ax4)
    ax4 = plt.xlabel('TIME')
    ax4 = plt.ylabel('GTREND SCORE')
    mons3 = (np.datetime64('2022-01-01','M') - np.datetime64('2019-01-01','M')).tolist()
    plt.title('%s Months Out of Sample Forecast for %s'%(mons3,keyword))
    avg_2023 = ((fit.forecast(85)[2][-1][1]+fit.forecast(85)[2][-1][0])/2)
    four_year = ((avg_2023-data['indoor farming']['2008-01-01'])/data['indoor farming']['2008-01-01'])*100
    print(f'Indoor Farming search trends in 2023 %s'%avg_2023)
    print('Percent change from 2008-01-01 to 2023-01-01: ',four_year)
    fig5, ax5 = plt.subplots(figsize=(8,6))
    fit.plot_predict(end='2023-01-01', ax=ax5)
    ax5 = plt.xlabel('TIME')
    ax5 = plt.ylabel('GTREND SCORE')
    mons4 = (np.datetime64('2023-01-01','M') - np.datetime64('2019-01-01','M')).tolist()
    plt.title('%s Months Out of Sample Forecast for %s'%(mons4,keyword))

def rolling_stats(data, keyword, window):
    r_mean=data.rolling(window=window).mean()
    r_std=data.rolling(window=window).std()
    fig,ax = plt.subplots(figsize=(12,8))
    sns.lineplot(x=data.index,y=data[keyword])
    sns.lineplot(x=data.index,y=r_mean[keyword])
    sns.lineplot(x=data.index,y=r_std[keyword])
    plt.legend([keyword, 'ROLLING MEAN', 'ROLLING STDEV'], loc='best')
    plt.xlabel('TIME',size=18)
    plt.ylabel('GOOGLE TREND INTEREST INDEX', size=18)
    plt.title(f'ROLLING STATS FOR %s'%keyword, size=24)

def plot_ts(data,cities):
    fig, ax = plt.subplots(figsize=(12,8))
    for city in cities:
        sns.lineplot(x=data.index, y=data[city], ax=ax)
    plt.legend(cities, loc='best')
    plt.xlabel('TIME', size=18)
    plt.ylabel('HOUSE PRICE USD',size=18)
    plt.title('ALL SELECTED METROS',size=24)

def dftest2(data):
    test = adfuller(data['instances'])
    test_output = pd.Series(test[0:4], index=['Test Stat', 'P-Value', '# Lags', '# Observations'])
    for key, value in test[4].items():
        test_output['Critical Value (%s)' %key]=value
    return(test_output)

def process_tweets(data):
    """This function will receive a raw data set from the Twint scrapper
        and process the dataframe. It will reduce the dataframe to just dates
        and tweets, convert date to datetime, set the index, count how many
        instances are from each date, drop the tweets, and then eliminate the
        duplicate dates."""
    data = data[['date', 'tweet']]
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data['instances']=data.groupby('date').transform('count')
    data.drop(columns=['tweet'], axis=1, inplace=True)
    data = data.loc[~data.index.duplicated(keep='first')]
    return data

def get_int(keyword, timeframe):
    pytrends.build_payload([keyword], geo='US', timeframe=timeframe)
    ts = pytrends.interest_over_time()
    ts.drop(columns=['isPartial'], axis=1, inplace=True)
    return ts

def arima_logm(data, order, keyword):
    model = ARIMA(data['2009':'2015'], order)
    fit = model.fit(disp=0)
    model_full = ARIMA(data['2009':], order)
    fit2 = model.fit(disp=0)
    print('ARIMA for',keyword)
    print(fit.summary())
    residuals = pd.DataFrame(fit.resid)
    residuals.plot(kind='kde', title=keyword)
    fig,ax=plt.subplots(figsize=(8,6))
    fit2.plot_predict(end='2019-01-01', ax=ax)
    ax = plt.xlabel('TIME')
    ax = plt.ylabel('GTREND SCORE')
    plt.title('In Sample Predictions for %s' % (keyword))
    avg_vals = []
    for line in list(fit.forecast(37)[2]):
        avg_vals.append((line[1]+line[0])/2)
    mse = mean_squared_error(data['2016':'2019-01-01'], avg_vals)
    rmse = math.sqrt(mse)
    print('This model can predict the remaining 37 data points with an RMSE of ',rmse)
    print('The following results are predicting values post-data set')
    print('These predictions can only be confirmed with new data')
    avg_2020 = ((fit.forecast(49)[2][-1][1]+fit.forecast(49)[2][-1][0])/2)
    print(f'Indoor Farming search trends in 2020 %s'%avg_2020)
    one_year = ((avg_2020-data['indoor farming']['2009-01-01'])/data['indoor farming']['2009-01-01'])*100
    print('Percent change from 2008-01-01 to 2020-01-01: ',one_year)
    fig2,ax2 = plt.subplots(figsize=(8,6))
    fit.plot_predict(end='2020-01-01', ax=ax2)
    ax2 = plt.xlabel('TIME')
    ax2 = plt.ylabel('GTREND SCORE')
    mons = (np.datetime64('2020-01-01','M') - np.datetime64('2019-01-01','M')).tolist()
    plt.title('%s Months Out of Sample Forecast for %s'% (mons,keyword))
    avg_2021 = ((fit.forecast(61)[2][-1][1]+fit.forecast(61)[2][-1][0])/2)
    two_year = ((avg_2021-data['indoor farming']['2009-01-01'])/data['indoor farming']['2009-01-01'])*100
    print(f'Indoor Farming search trends in 2021 %s'%avg_2021)
    print('Percent change from 2008-01-01 to 2021-01-01: ',two_year)
    fig3,ax3 = plt.subplots(figsize=(8,6))
    fit.plot_predict(end='2021-01-01', ax=ax3)
    ax3 = plt.xlabel('TIME')
    ax3 = plt.ylabel('GTREND SCORE')
    mons2 = (np.datetime64('2021-01-01','M') - np.datetime64('2019-01-01','M')).tolist()
    plt.title('%s Months Out of Sample Forecast for %s'%(mons2,keyword))
    avg_2022 = ((fit.forecast(73)[2][-1][1]+fit.forecast(73)[2][-1][0])/2)
    three_year = ((avg_2022-data['indoor farming']['2009-01-01'])/data['indoor farming']['2009-01-01'])*100
    print(print(f'Indoor Farming search trends in 2022 %s'%avg_2022))
    print('Percent change from 2008-01-01 to 2022-01-01: ',three_year)
    fig4, ax4 = plt.subplots(figsize=(8,6))
    fit.plot_predict(end='2022-01-01', ax=ax4)
    ax4 = plt.xlabel('TIME')
    ax4 = plt.ylabel('GTREND SCORE')
    mons3 = (np.datetime64('2022-01-01','M') - np.datetime64('2019-01-01','M')).tolist()
    plt.title('%s Months Out of Sample Forecast for %s'%(mons3,keyword))
    avg_2023 = ((fit.forecast(85)[2][-1][1]+fit.forecast(85)[2][-1][0])/2)
    four_year = ((avg_2023-data['indoor farming']['2009-01-01'])/data['indoor farming']['2009-01-01'])*100
    print(f'Indoor Farming search trends in 2023 %s'%avg_2023)
    print('Percent change from 2008-01-01 to 2023-01-01: ',four_year)
    fig5, ax5 = plt.subplots(figsize=(8,6))
    fit.plot_predict(end='2023-01-01', ax=ax5)
    ax5 = plt.xlabel('TIME')
    ax5 = plt.ylabel('GTREND SCORE')
    mons4 = (np.datetime64('2023-01-01','M') - np.datetime64('2019-01-01','M')).tolist()
    plt.title('%s Months Out of Sample Forecast for %s'%(mons4,keyword))

def vader_process(data):
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data = data[['tweet']]
    scores = [analyzer.polarity_scores(tweet)['compound'] for tweet in data['tweet']]
    data['sentiment'] = scores
    data['score'] = pd.cut(data.sentiment,
                                    bins=[-1,-0.051,0.049,1],
                                    labels=['negative','neutral','positive'])
    return data

def dftest_tweets(data):
    test = adfuller(data['sentiment'])
    test_output = pd.Series(test[0:4], index=['Test Stat', 'P-Value', '# Lags', '# Observations'])
    for key, value in test[4].items():
        test_output['Critical Value (%s)' %key]=value
    return(test_output)

def nice_plot(data, keyword):
    """Provided a time series data set and a specified keyword for which
         interest data was pulled, the function will print a nice looking 
          series graph in Seaborn."""
    fig,ax=plt.subplots(figsize=(8,6))
    sns.lineplot(x=data.index, y=data[keyword])
    ax = plt.xlabel('TIME')
    ax = plt.ylabel('GOOGLE TREND INTEREST INDEX')
    plt.title('TIME SERIES VISUAL OF %s' % (keyword))
    plt.show()

def gen_dftest(data,keyword):
    test = adfuller(data[keyword])
    test_output = pd.Series(test[0:4], index=['Test Stat', 'P-Value', '# Lags', '# Observations'])
    for key, value in test[4].items():
        test_output['Critical Value (%s)' %key]=value
    return(test_output)

def szn_decomp(data, keyword, model=[], graphs=[]):
    """Will provide seasonal_decomposition information via
        `statsmodels seasonal_decompose`"""
    for mod in model:
        decomp = seasonal_decompose(data, model=mod)
        trend = decomp.trend
        seasonal = decomp.seasonal
        residual = decomp.resid
        fig,ax = plt.subplots(figsize=(12,8))
        if 'trend' in graphs:
            sns.lineplot(x=trend.index, y=trend, color='black')
        if 'szn' in graphs:
            sns.lineplot(x=seasonal.index, y=seasonal, color='green')
        if 'residual' in graphs:
            sns.lineplot(x=residual.index, y=residual)
        plt.legend(['TREND', 'SEASONALITY', 'RESIDUALS'], loc='best')
        plt.xlabel('TIME',size=18)
        plt.ylabel('GOOGLE TREND INTEREST INDEX', size=18)
        if mod == 'additive':
            plt.title(f'ADDITIVE MODEL FOR %s'%keyword, size=24)
        if mod == 'multiplicative':
            plt.title(f'MULTIPLICATIVE MODEL FOR %s'%keyword, size=24)

def sarima_gs(data_model, data_train, data_val, keyword, m=52):
    """Does an `auto_arima` search for the best parameters.
        Prints a graph showing the training, test, and actual values.
        train = training timeframe ['yyyy-mm-dd':'yyyy-mm-dd'] format
        forecast = forecast timeframe ['yyyy-mm-dd':] format
        m = can be changed to account for different types of seasonality
        ---->'m' default is 52 for weekly, 12 is for monthly
        Will print out some graphs and give model best params"""
    model = auto_arima(data_model,start_p=0,d=0,start_q=0,start_P=0,D=0,start_Q=0,
            trace=True, m=52, seasonal=True, error_action='ignore', n_jobs=-1,
             suppress_warnings=True, random_state=42)
    model.fit(data_train)
    print(model.summary())
    forecast = model.predict(n_periods=len(data_val))
    forecast = pd.DataFrame(forecast,index = data_val.index,columns=['prediction'])
    # Plot the nice graphs
    fig,ax = plt.subplots(figsize=(12,8))
    sns.lineplot(x=data_train.index, y=data_train[keyword], color='black')
    sns.lineplot(x=forecast.index, y=forecast['prediction'], color='green')
    sns.lineplot(x=data_val.index, y=data_val[keyword], color='orange')
    plt.legend(['TRAINING DATA', 'PREDICTIONS', 'ACTUAL'], loc='best')
    plt.xlabel('TIME',size=18)
    plt.ylabel('GOOGLE TREND INTEREST INDEX', size=18)
    plt.title(f'BEST SARIMA MODEL FOR %s'%keyword, size=24)

def sarima(data_train, data_val, keyword, order, sorder, m=52):
    """Does an `auto_arima` search for the best parameters.
        Prints a graph showing the training, test, and actual values.
        train = training timeframe ['yyyy-mm-dd':'yyyy-mm-dd'] format
        forecast = forecast timeframe ['yyyy-mm-dd':] format
        m = can be changed to account for different types of seasonality
        ---->'m' default is 52 for weekly, 12 is for monthly
        Will print out some graphs and give model best params"""
    model = SARIMAX(data_train, order=order, seasonal_order=sorder,
                      trend='t')
    fit = model.fit(data_train)
    # print(fit.summary())
    forecast = model.predict(60)
    print('Forecasting 60 months into the future from the\ntraining data (2016-2021).\n...\n...')
    forecast = pd.DataFrame(forecast)
    forecast = rename_column(forecast, 0, 'forecast')
    # Plot the nice graphs
    fig,ax = plt.subplots(figsize=(12,8))
    sns.lineplot(x=data_train.index, y=data_train, color='black')
    sns.lineplot(x=forecast.index, y=forecast['forecast'], color='green')
    sns.lineplot(x=data_val.index, y=data_val, color='orange')
    plt.legend(['TRAINING DATA', 'FORECAST', 'ACTUAL'], loc='best')
    plt.xlabel('TIME',size=18)
    plt.ylabel('GOOGLE TREND INTEREST INDEX', size=18)
    plt.title(f'BEST SARIMA MODEL FOR %s'%keyword, size=24)