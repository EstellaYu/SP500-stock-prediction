import numpy as np
import pandas as pd
from datetime import datetime, timedelta


import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func

# ARIMA dependencies
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima.utils import nsdiffs
from pmdarima.arima.utils import ndiffs
import pmdarima as pm
from scipy.signal import find_peaks

from flask import Flask, jsonify, render_template

#################################################
# Database Setup
#################################################
engine = create_engine("sqlite:///data/CompanyData.sqlite")

# reflect an existing database into a new model
Base = automap_base()
# reflect the tables
Base.prepare(engine, reflect=True)

# Save reference to the table
MasterData = Base.classes.MasterData
QuintileMonthlyData = Base.classes.QuintileMonthlyData
QuintileAvgData = Base.classes.QuintileAvgData
CurrentData = Base.classes.CurrentData

# List of all criterias
criterias = ['price_earnings','price_book','ev_revenue','ev_ebit','net_debt_capital','market_cap']

#################################################
# Flask Setup
#################################################
app = Flask(__name__)

#################################################
# Flask Routes
#################################################

@app.route("/")
def welcome():
    # return Homepage
    return render_template("index.html")
    # """List all available api routes."""
    # Available links
    # f"Available Routes:<br/>"
    # f"/FiveLines/criteria/sector <br/>"
    # f"/BarChart/quintile/sector <br/>"
    # f"/ThreeDee/sector <br/>"
    # f"/CompanyData/ticker/criteria <br/>"
    # f"/CriteriaList <br/>"
    # f"/TickerList <br/>"
    # f"/SectorList <br/>"

def criteria_abbrev(criteria_dbname):
    if (criteria_dbname == "price_earnings"):
        criteria = "1 P-E"
    elif (criteria_dbname == "price_book"):
        criteria = "2 P-B"
    elif (criteria_dbname == "market_cap"):
        criteria = "6 Mkt Cap"
    elif (criteria_dbname == "net_debt_capital"):
        criteria = "5 Debt-Cap"
    elif (criteria_dbname == "ev_revenue"):
        criteria = "3 EV-Sales"
    elif (criteria_dbname == "ev_ebit"):
        criteria = "4 EV-EBIT"
    return criteria

def reverse_criteria(given_criteria):
    if (given_criteria == "1 P-E"):
        criteria = "price_earnings"
    elif (given_criteria == "2 P-B"):
        criteria = "price_book"
    elif (given_criteria == "6 Mkt Cap"):
        criteria = "market_cap"
    elif (given_criteria == "5 Debt-Cap"):
        criteria = "net_debt_capital"
    elif (given_criteria == "3 EV-Sales"):
        criteria = "ev_revenue"
    elif (given_criteria == "4 EV-EBIT"):
        criteria = "ev_ebit"
    return criteria

def abs2(x):
    return x.real**2 + x.imag**2

def next_month_end(time):
    if time.month == 12:
        y = time.year + 1
        m = 2
        return datetime(year = time.year + 1, month = 2, day = 1)- timedelta(days = 1)
    elif time.month == 11:
        y = time.year + 1
        m = 1
        return datetime(year = time.year + 1, month = 1, day = 1)- timedelta(days = 1)
    else:
        y = time.year
        m = time.month + 2
        
    next_mon_end = datetime(year = y, month = m, day = 1) - timedelta(days = 1)
    return next_mon_end

@app.route("/Quintile-Performance")
def fig1():
    return render_template("fig1.html")

@app.route("/Three-Dee-Criteria-Visuals")
def fig2():
    return render_template("fig2.html")
    
@app.route("/Criteria-Parallel-Comparison")
def fig3():
    return render_template("fig3.html")

@app.route("/Price-Predictor-ARIMA")
def fig4():
    return render_template("fig4.html")

@app.route("/ML-Portfolio-Index")
def fig5():
    return render_template("fig5.html")

@app.route("/CriteriaList")
def CriteriaList():
    """Return a list of all criteria"""
    # Get the unique list of criteria names
    session = Session(engine)
    results = session.query(QuintileAvgData.criteria).distinct().all()

    # Convert list of tuples into normal list, then run them through a name conversion
    all_criteria = list(np.ravel(results))
    all_items = [criteria_abbrev(x) for x in all_criteria]

    # Sort alphabetically
    all_items.sort()

    return jsonify(all_items)

@app.route("/SectorList")
def SectorList():
    """Return a list of all sectors"""
    # Get the unique list of sectors
    session = Session(engine)
    results = session.query(QuintileAvgData.sector).distinct().all()

    # Convert list of tuples into normal list, then run them through a name conversion
    all_sectors = list(np.ravel(results))

    # Sort alphabetically
    all_sectors.sort()

    return jsonify(all_sectors)

@app.route("/TickerList")
def TickerList():
    """Return a list of all tickers"""
    # Get the unique list of tickers
    session = Session(engine)
    results = session.query(MasterData.ticker).distinct().all()

    # Convert list of tuples into normal list, then run them through a name conversion
    all_tickers = list(np.ravel(results))

    # Sort alphabetically
    all_tickers.sort()

    return jsonify(all_tickers)

@app.route("/FiveLines/<selected_criteria>/<selected_sector>")
def FiveLines(selected_criteria, selected_sector):
    """Return a list of the data requested"""

    session = Session(engine)
    real_criteria = reverse_criteria(selected_criteria)
    results = session.query(QuintileMonthlyData).filter_by(criteria=real_criteria).filter_by(sector=selected_sector).all()

    # Create a dictionary from the row data and append to a list of all_rows
    all_rows = []
    for result in results:
        data_dict = {}
        data_dict["monthend_date"] = result.monthend_date.strftime("%Y-%m-%d")
        data_dict["quintile"] = result.quintile
        data_dict["wealth_index"] = result.wealth_index
        all_rows.append(data_dict)

    return jsonify(all_rows)

@app.route("/BarChart/<selected_quintile>/<selected_sector>")
def BarChart(selected_quintile, selected_sector):
    """Return a list of the data requested"""

    session = Session(engine)
    results = session.query(QuintileAvgData).filter_by(quintile=selected_quintile).filter_by(sector=selected_sector).all()

    # Create a dictionary from the row data and append to a list of all_rows
    all_rows = []
    for result in results:
        data_dict = {}
        data_dict["criteria"] = criteria_abbrev (result.criteria)
        data_dict["total_return"] = result.port_return
        all_rows.append(data_dict)

    # Sort the dictionaries by the value of "criteria"
    all_rows = sorted(all_rows, key = lambda i: i['criteria']) 
    return jsonify(all_rows)

@app.route("/ThreeDee/<selected_sector>")
def ThreeDee(selected_sector):
    """Return a list of the data requested"""

    session = Session(engine)
    results = session.query(QuintileAvgData).filter_by(sector=selected_sector).all()

    # Create a dictionary from the row data and append to a list of all_rows
    all_rows = []
    for result in results:
        data_dict = {}
        data_dict["criteria"] = criteria_abbrev (result.criteria)
        data_dict["quintile"] = result.quintile
        data_dict["total_return"] = result.port_return
        all_rows.append(data_dict)

    # Sort the dictionaries by the value of "criteria"
    all_rows = sorted(all_rows, key = lambda i: i['criteria']) 
    return jsonify(all_rows)

@app.route("/CompanyData/<selected_ticker>/<selected_criteria>")
def CompanyData(selected_ticker, selected_criteria):
    """Return a list of the data requested"""

    session = Session(engine)
    results = session.query(MasterData).filter_by(ticker=selected_ticker).all()

    # Create a dictionary from the row data and append to a list of all_rows
    all_rows = []
    for result in results:
        data_dict = {}
        data_dict["monthend_date"] = result.monthend_date.strftime("%Y-%m-%d")

        if (selected_criteria == "price"):
            item_value = result.wealth_index
        else: 
            real_criteria = reverse_criteria(selected_criteria)
            if (real_criteria == "price_earnings"):
                item_value = result.price_earnings
            elif (real_criteria == "price_book"):
                item_value = result.price_book
            elif (real_criteria == "market_cap"):
                item_value = result.market_cap
            elif (real_criteria == "net_debt_capital"):
                item_value = result.net_debt_capital
            elif (real_criteria == "ev_revenue"):
                item_value = result.ev_revenue
            elif (real_criteria == "ev_ebit"):
                item_value = result.ev_ebit

        data_dict["item_value"] = item_value
        all_rows.append(data_dict)

    # Sort the dictionaries by the value of "criteria"
    return jsonify(all_rows)

@app.route("/CompanyData/<selected_ticker>")
def CompanyPriceData(selected_ticker):
    """Return a list of the data requested"""
    session = Session(engine)
    results = session.query(MasterData).filter_by(ticker=selected_ticker).all()
    
    # Create a dictionary from the row data and append to a list of all_rows
    all_rows = []
    for result in results:
        data_dict = {}
        data_dict["monthend_date"] = result.monthend_date.strftime("%Y-%m-%d")

        item_value = result.wealth_index
        
        data_dict["item_value"] = item_value
        all_rows.append(data_dict)

    companyInfo = session.query(CurrentData).filter_by(ticker=selected_ticker).all()
    financials_dict = {}
    quintile_all={}
    quintile_sector = {}

    for info in companyInfo:
        for cri in criterias:
            fin_val = eval(f"info.{cri}")
            if (fin_val == np.Infinity) : 
                fin_val = 0
            financials_dict[cri] = fin_val
            
            quintile_all[cri] = eval(f"info.q_{cri}")
            quintile_sector[cri] = eval(f"info.qs_{cri}")
    companyData = {}
    companyData['ticker'] = selected_ticker
    companyData['name'] = results[0].name
    companyData['sector'] = results[0].sector
    companyData['price'] = all_rows
    companyData['info'] = financials_dict
    companyData['quintile'] = {"all": quintile_sector, "sector": quintile_sector}
    companyData['MLind'] = companyInfo[0].dl_prediction
    companyData['ARIMA'] = ARIMA(selected_ticker)
    # Sort the dictionaries by the value of "criteria"
    return jsonify([companyData])

# @app.route("/ARIMA/<selected_ticker>")
def ARIMA(selected_ticker):
    """Return a list of the data requested"""
    session = Session(engine)
    results = session.query(MasterData).filter_by(ticker=selected_ticker).all()

    """1. Create Dataframe, prepare for training"""
    date = [res.monthend_date for res in results]
    wealth_ind = [res.wealth_index for res in results]  
    data = pd.DataFrame({'date': date, 
                   'wealth_ind': wealth_ind})
    data = data.set_index('date')
    data.index = pd.to_datetime(data.index)
    
    """2. Find seasonality"""
    result = seasonal_decompose(data, model='multiplicative')
    # find periodicity in seasonal
    seasonal = result.seasonal['wealth_ind']
    fft = np.fft.rfft(seasonal, norm="ortho")
    selfconvol=np.fft.irfft(abs2(fft), norm="ortho")
    x, y = find_peaks(selfconvol, distance=6) 
    season = x[1]
    print(f'seasonality: {season}')

    """3. fit ARIMA peicewise, and get prediction"""
    period = season
    history = [data[i: i+period] for i in range(len(data) - period + 1)]

    predict = []

    for h in history:
        D = nsdiffs(h, m=season, max_D=12, test='ch') # determine seasonal D
        pre_dict = {}
        stepwise_model = pm.auto_arima(h, start_p=1, start_q=1,
                                    test='adf',
                            max_p=3, max_q=3, m=season,
                            start_P=0, seasonal=True,
                            d=None, D=D, trace=False,
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=True)
        print("ARIMA order: ", stepwise_model.order,stepwise_model.seasonal_order, end = '\r')
        model_fit = stepwise_model.fit(h)
        p = model_fit.predict(n_periods = 1)
        next_mon_end = next_month_end(h.index[-1])
        pre_dict['predict_time'] = datetime.date(next_mon_end).isoformat()
        pre_dict['price'] = p[0]
        predict.append(pre_dict)

    """4. predict forward for 6 months"""
    n_periods = 6

    D = nsdiffs(h, m=season, max_D=12, test='ch') # determine seasonal D
    stepwise_model = pm.auto_arima(data, start_p=1, start_q=1,
                                test='adf',
                                max_p=3, max_q=3, m=season,
                                start_P=0, seasonal=True,
                                d=None, D=D, trace=False,
                                error_action='ignore',  
                                suppress_warnings=True, 
                                stepwise=True)
    print("ARIMA order: ", stepwise_model.order,stepwise_model.seasonal_order, end = '\r')
    model_fit = stepwise_model.fit(data)

    fc, confint = stepwise_model.predict(n_periods=n_periods, return_conf_int=True)

    future_6_mon = []
    latest_time_stamp = data.index[-1]
    for i in range(len(fc)):
        pre_dict = {}
        latest_time_stamp = next_month_end(latest_time_stamp)
        pre_dict['predict_time'] = datetime.date(latest_time_stamp).isoformat()
        pre_dict['price'] = fc[i]
        pre_dict['conf_int_lo'] = confint[i, 0]
        pre_dict['conf_int_hi'] = confint[i, 1]
        future_6_mon.append(pre_dict)

    """5. assemble for output"""
    ARIMA = {}
    ARIMA['piecewise_arima'] = predict
    ARIMA['future_6_mon'] = future_6_mon

    return ARIMA


if __name__ == '__main__':
    app.run(debug=True)
