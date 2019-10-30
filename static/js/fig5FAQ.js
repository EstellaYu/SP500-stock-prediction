x = [
    {
      "header-html": "What is <em><b>Machine Learning Porfolio Index</b><em>?",
      "content-html":"<p><br>Stocks in S&P500 are grouped into <b>3 porfolios</b>:<ol><li>Outperformers</li><li>Market (Average) Performers</li><li>Under Performers</li></ol></p><p>This index is predicted from a <b>Deep Learning</b> model based on the monthly time series data in the past 24 months. </p>"
    }, 
    {
      "header-html": "Model Construction",
      "content-html":"<p><br><b>Training-test split</b>: In every experiment, 20% tickers are randomly selected as <em>test set</em>. <br><br><b>Feature Matrix</b>: Financial performance criteria data for 24 months. The 6 Criterias include<br> (1) Price/Earnings <br> (2) Price/Book <br> (3) Enterprise Value/Revenue <br> (4) Enterprise Value/EBIT <br> (5) Net Debt/Capital <br> (6) Market Capitalization <br><br> <b>Target</b>: Group Index (1, 2, or 3) within S&P500 in the following month<br><br> <b>Model Construction</b>: 2 Dense Layers, 100 nodes at each layer, 40 epoches.<br><br> <b>Monte Carlo Experiments</b>: Run experiments for 30 times, and the average test performance are reported. <br> </p>"
    },
    {
        "header-html": "Model Performance",
        "content-html":"<p><br>Porfolio average annual return<ul><li>S&P 500: 8.95% (Base case) </li><li>Group 1: 10.15%</li><li>Group 2: 8.64%</li><li>Group 3: 5.41%</li></ul> </p>"
    }
  ]