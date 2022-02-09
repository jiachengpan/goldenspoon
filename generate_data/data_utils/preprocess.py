import datetime
from dateutil.relativedelta import relativedelta

def get_last_day_of_the_quarter(dt):
    current_quarter = int((dt.month - 1) / 3 + 1)
    return datetime.date(dt.year, 3 * current_quarter, 1) + \
        relativedelta(months=1) - relativedelta(days=1)

def get_last_day_of_the_month(dt):
    return datetime.date(dt.year, dt.month, 1) + \
        relativedelta(months=1) - relativedelta(days=1)

def preprocess_time(df):
    if 'time' not in df.index:
        return df

    if 'fund_id' in df.index:
        time_processor = get_last_day_of_the_quarter
    else:
        time_processor = get_last_day_of_the_month

    melt = df.reset_index().melt(id_vars=df.index).dropna()
    melt.time = melt.time.apply(time_processor)
    return melt.groupby(list(df.index)).last()

def preprocess_data(df):
    df = preprocess_time(df)

    return df