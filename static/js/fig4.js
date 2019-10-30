criterias = ['price_earnings','price_book','ev_revenue','ev_ebit','net_debt_capital','market_cap']

makeResponsive()
d3.select(window).on("resize", makeResponsive);

function makeResponsive(){

    input = "AMZN"
    if (tickers.includes(input.toUpperCase())) {
        ticker = input.toUpperCase();
    }
    else {
        ticker = "AMZN";
        console.warn("ticker not in S&P500")
    }
    data_path = "/CompanyData/" + ticker
    console.log(data_path)

    makeHomeButtons()
    inputTicker()

    // get lines data from file
    d3.json(data_path, function(error, data){
        if (error) return console.warn(error)
        // parse time
        var parseTime = d3.timeParse("%Y-%m-%d")

        data = data[0]
        d3.select("#fig4").select(".container").select(".row")
            .append('div').classed("col-10", true).attr("id", "stockinfo")
        stockInfo(data)
        // console.log(data.price)

        var chartGroup = resizeCanvas();
        var actual = data.price
        actual.forEach(function(d){
            d.item_value = +d.item_value
            d.monthend_date = parseTime(d.monthend_date)
        })
        
        piecewise_arima = data.ARIMA.piecewise_arima
        piecewise_arima.forEach(function(d){
            d.price = +d.price
            d.predict_time = parseTime(d.predict_time)
        })

        future_6_mon = data.ARIMA.future_6_mon
        future_6_mon.forEach(function(d){
            d.price = +d.price
            d.conf_int_hi = +d.conf_int_hi
            d.conf_int_lo = +d.conf_int_lo
            d.predict_time = parseTime(d.predict_time)
        })
        makeLines(actual, piecewise_arima, future_6_mon, chartGroup);
    })  

    d3.select('#formField').selectAll("button").on('click',function(){
        var value = document.getElementById("inputTicker").value
        if (value != input & tickers.includes(value.toUpperCase())){
            input = value; 
            ticker = input.toUpperCase();
            document.getElementById("inputTicker").value = ticker
            data_path = "/CompanyData/" + ticker
            var stockinfo = d3.select('#stockinfo').selectAll("div")
            if (!stockinfo.isEmpty){stockinfo.remove()} 
            update(data_path)
        }
        else {window.alert("Ticker '"+ value.toUpperCase() + "' not in S&P500")}  
    })

}

function makeHomeButtons(){
    var buttons = d3.select("#home")
    if (!buttons.isEmpty){buttons.remove()}
    var homeButton = d3.select("body").append("div").attr("id", "home")
                    .append("button").html("Home").attr("id", "homeButton")
    d3.select("#homeButton").on("click",function(){
        location.href = "/"
    })    
    var buttons = d3.select("#visual")
    if (!buttons.isEmpty){buttons.remove()}
    var visualButton = d3.select("body").append("div").attr("id", "visual")
                        .append("button").html("Visuals").attr("id", "visualButton")
    d3.select("#visualButton").on("click",function(){
        location.href = "/#page2"
    })
} 

function inputTicker(){
    var ALLdiv = d3.select("#fig4").selectAll("div")
    if (!ALLdiv.isEmpty){ALLdiv.remove()}
    var container = d3.select("#fig4").append("div").classed("container", true)
                        .append("div").classed("row", true)
    var submitButton = container.append("div")
                                    .classed("col-2", true)
                                    .attr("id", "selectTicker")
    var formField = submitButton.append('form')
                                .classed("form-inline", true)
                                .attr("id", 'formField')
    var inputForm = formField.append("div")
                            .classed("form-group", true)
                            .append("input")
                            .attr('type','text')
                            .attr("placeholder",'Input a Ticker')
                            .attr("id", "inputTicker")
    var inputButton = formField.append("button")
                                .classed("bnt", true)
                                .attr('type', 'button')
                                .attr("id", "submitTickerButton")
                                .html("Submit!")                       
}

function stockInfo(data){
    // preprocessing
    var stockinfo = d3.select('#stockinfo').selectAll("div")
    if (!stockinfo.isEmpty){stockinfo.remove()}

    var stockinfo = d3.select("#stockinfo").append("div")

    var name = data.name
    var sector = data.sector
    var finantials = data.info 
    var quintile_all = data.quintile.all
    var quintile_sector = data.quintile.sector
    var MLind = data.MLind

    stockinfo.append('p')
            .html("<b>" + name + "<b> (" + sector+ ")").attr('id','CompanyName')

    var infoTable = stockinfo.append("div")
                                .append('table').classed("table-bordered", true)
                                .attr("id","infoTable")
    
    head = infoTable.append('thead')
    header1 = head.append("tr")
    header1.append("td").attr("rowspan",'2').html("<b>ML Portolio Index")
    header1.append("td").attr("colspan",'7').html("<b>Quintile Information")
    header2 = head.append("tr")
    header2.append("td").html("")
    
    body = infoTable.append("tbody")
    line1 = body.append("tr")
    line2 = body.append("tr")
    line3 = body.append("tr")

    MLindex_td = line1.append("td").attr("rowspan",'3').attr("id", "MLindex")
                        .attr("bgcolor", markerColor("Q"+MLind))
    MLindex_td.append('font').attr("color",'white').html("Group " + MLind)
    .append("a").attr("id", "GroupDescription")
    .attr("href",'ML-Portfolio-Index')
                        .html("<em><b>ML Portfolio Index</em>?")
    line1.append("td").html("<b>current value")
    line2.append("td").html("<b>Quintile within S&P500")
    line3.append("td").html("<b>Quintile within sector")

    criterias.forEach(function(c){
        header2.append("td").html("<b>" + c.replace(/_/g, " "))
        if (finantials[c] < 1e3) line1.append("td").html(finantials[c].toFixed(1))
        else line1.append("td").html(finantials[c].toExponential(2))
        q = quintile_all[c].toFixed(0)
        line2.append("td").attr("bgcolor", markerColor("Q"+q))
            .append('font').attr("color",'white').html("Q<b>"+q)
        q = quintile_sector[c].toFixed(0)
        line3.append("td").attr("bgcolor", markerColor("Q"+q))
            .append('font').attr("color",'white').html("Q<b>"+q)
    })
    
}

function update(data_path){  
    document.getElementById("loader-word").style.display = "block"
    document.getElementById("loader-wraper").style.display = "flex"
    loaderSymbol() 
    d3.json(data_path, function(error, data){
        if (error) return console.warn(error)
        // parse time
        var parseTime = d3.timeParse("%Y-%m-%d")
       
        data = data[0]

        stockInfo(data)

        actual = data.price
        actual.forEach(function(d){
            d.item_value = +d.item_value
            d.monthend_date = parseTime(d.monthend_date)
        })

        piecewise_arima = data.ARIMA.piecewise_arima
        piecewise_arima.forEach(function(d){
            d.price = +d.price
            d.predict_time = parseTime(d.predict_time)
        })

        future_6_mon = data.ARIMA.future_6_mon
        future_6_mon.forEach(function(d){
            d.price = +d.price
            d.conf_int_hi = +d.conf_int_hi
            d.conf_int_lo = +d.conf_int_lo
            d.predict_time = parseTime(d.predict_time)
        })

        // console.log(data)
        var yLinearScale = make_yScale(actual,piecewise_arima, future_6_mon, chartHeight)
        yAxis = d3.select(".yAxis")
        yAxis = renderYAxis(yLinearScale, yAxis);
        var xTimeScale = d3.scaleTime()
                            .range([10, chartWidth])
                            .domain([d3.min(actual.map(actual => actual.monthend_date)),
                                     d3.max(future_6_mon.map(future_6_mon => future_6_mon.predict_time))])
        
        // update lines and cirlces
        var i = 3; // update actual price
        updateLineMarker(i, actual, xTimeScale, yLinearScale)
        var i = 5; // update piecewise arima price
        updateLineMarker(i, piecewise_arima, xTimeScale, yLinearScale)
        var i = 1; // update future_6_mon arima price
        updateLineMarker(i, future_6_mon, xTimeScale, yLinearScale)    
        // confidence_interval
              
    })  
}

function updateLineMarker(i, data, xTimeScale,  yLinearScale){
    if (i == 3) {// actual price
        xtoken = 'monthend_date' ; ytoken = 'item_value'
    }
    else {
        xtoken = 'predict_time' ; ytoken = 'price'
    }
    circles = d3.selectAll(`.Q${i}`)
                .data(data)
    line = d3.line()
            .x(d => xTimeScale(d[`${xtoken}`]))
            .y(d => yLinearScale(d[`${ytoken}`]))
    d3.selectAll(`.Q${i}_line`)
        .data(data)
        .transition()
        .duration(1000)
        .attr("d", line(data))
    if (i==1) {// future_6_mon arima
        indexies = d3.range(data.length);
        area = d3.area()
                .x( function(d) { return xTimeScale(data[d].predict_time) } )
                .y0( function(d) { return yLinearScale(data[d].conf_int_lo)  } )
                .y1(  function(d) { return yLinearScale(data[d].conf_int_hi) } );
                      
        d3.selectAll('#confidence_interval')
                    .datum(indexies)
                    .transition()
                    .duration(1000)
                    .attr('d', area)
    }
    renderMarkers(circles, xTimeScale, yLinearScale, xtoken, ytoken) 
}

// 1.  set up chart
// ================================
// function make responsieve to window size
function resizeCanvas() {
    // clear original chart in case window size update
    var svg = d3.select("#fig4").select("svg") 
    if (!svg.empty()){svg.remove();};

    svgHeight = window.innerHeight * 0.5;
    svgWidth = window.innerWidth * 0.8;

    margin = {
        left: 80,
        top: 10,
        right: 10,
        bottom: 100 
    };

    chartHeight = svgHeight - margin.top - margin.bottom
    chartWidth = svgWidth - margin.left - margin.right

    svg = d3.select("#fig4").append("svg")
            .attr("height", svgHeight)
            .attr("width", svgWidth)

    // create new canvas, shifting the origin to center canvas
    var chartGroup = svg.append("g")
                        .attr("transform", `translate(${margin.left}, ${margin.top})`)

    return chartGroup;
}

// 2. Main function to make lines
function makeLines(actual, piecewise_arima, future_6_mon, chartGroup){
    // 3. create scales, and axis (without data)
    // =================================
    // 3.1 create xScale and xAxis

    var xTimeScale = d3.scaleTime()
                       .range([10, chartWidth])
                       .domain([d3.min(actual.map(actual => actual.monthend_date)),
                                d3.max(future_6_mon.map(future_6_mon => future_6_mon.predict_time))])
    var bottomAxis = d3.axisBottom(xTimeScale).tickFormat(d3.timeFormat("%b/%y"));
    chartGroup.append("g")
                .attr("transform", `translate(0, ${chartHeight})`)
                .call(bottomAxis);

    // 3.2 create yScale and yAxis
    // =================================
    var yLinearScale = make_yScale(actual, piecewise_arima, future_6_mon, chartHeight)
    var leftAxis = d3.axisLeft(yLinearScale).tickFormat(d3.format("$.2f"))
    chartGroup.append("g")
            .attr("class","yAxis")
            .call(leftAxis)
    
    // add label
    var xLabel = chartGroup.append("g")
    var yLabel = chartGroup.append("g")
    xLabel.append("text")
            .attr("x", chartWidth/2 - 50)
            .attr('y', chartHeight + 70)
            .attr("font-size", "25px")
            .text('Month End')
    yLabel.append("text")
            .attr("x", - chartHeight/2 - 80)
            .attr('y', -60)
            .attr("font-size", "25px")
            .attr('transform', 'rotate(-90)')
            .text('Stock Wealth Index')

    // 5. Plot lines and circles
    // =================================
    // 5.1 plot actual price markers and lines
    var i = 3; // use the color black for historic price
    chartGroup.append("g").attr("id", `elementGroupQ${i}`)
    drawLine(actual, chartGroup, xTimeScale, yLinearScale, i, 0)
    drawMarkers(actual, chartGroup, xTimeScale, yLinearScale, i, 0);

    // 5.2 plot piecewise_arima markers and lines
    var i = 5; // use the color black for historic price
    chartGroup.append("g").attr("id", `elementGroupQ${i}`)
    drawLine(piecewise_arima, chartGroup, xTimeScale, yLinearScale, i, 1)
    drawMarkers(piecewise_arima, chartGroup, xTimeScale, yLinearScale, i, 1);

    // 5.3 plot future_6_mon confidence_interval
    var indexies = d3.range(future_6_mon.length );
    var area = d3.area()
                .x( function(d) { return xTimeScale(future_6_mon[d].predict_time) } )
                .y0( function(d) { return yLinearScale(future_6_mon[d].conf_int_lo)  } )
                .y1(  function(d) { return yLinearScale(future_6_mon[d].conf_int_hi) } );
                      
	chartGroup.append('path')
                .datum(indexies)
                .attr('class', 'area')
                .attr('fill', 'lightseagreen')
                .attr('opacity', '0.4')
                .attr('d', area)
                .attr("id", "confidence_interval")
    
    // 5.4 plot future_6_mon markers and lines
    var i = 1; // use the color black for historic price
    chartGroup.append("g").attr("id", `elementGroupQ${i}`)
    drawLine(future_6_mon, chartGroup, xTimeScale, yLinearScale, i, 2)
    drawMarkers(future_6_mon, chartGroup, xTimeScale, yLinearScale, i, 2);


    // 6. Add tooltips
    // ================================= 
    var circleGroup = d3.selectAll("circle")

    updateTooltip(circleGroup)
}

function updateTooltip(circleGroup){
    var formatTime = d3.timeFormat("%b-%Y")

    var tooltip = d3.select(".tooltip")
    if (!tooltip.empty()){tooltip.remove()}
    var tooltip = d3.select("#fig4")
                    .append("div")
                    .classed("tooltip", true)
    circleGroup
        .on("mouseover", function(d, i){
            d3.selectAll("circle")
                .transition()
                .duration(500)
                .attr("r", 2)
            d3.select(this)
                .transition()
                .duration(500)
                .attr("r", 5)

            var which_data_group = numQuintile(d3.select(this).attr("fill"));
            if (which_data_group == 3) {
                var currentVal = d.item_value;
                tooltip.style("display","block")
                        .html(`Actual Price<hr>${formatTime(d.monthend_date)} <hr>$${currentVal.toFixed(2)}<br>`)
                        .style("left", d3.event.pageX - 90 + "px")
                        .style("top", d3.event.pageY - 140 + "px")
                        .style("background", d3.select(this).attr("fill"))
            }
            else if (which_data_group == 5) {
                var currentVal = d.price;
                tooltip.style("display","block")
                        .html(`Stepwise ARIMA<hr>${formatTime(d.predict_time)} <hr>$${currentVal.toFixed(2)}<br>`)
                        .style("left", d3.event.pageX - 90 + "px")
                        .style("top", d3.event.pageY - 140 + "px")
                        .style("background", d3.select(this).attr("fill"))
            }
            else if (which_data_group == 1) {
                var currentVal = d.price;
                tooltip.style("display","block")
                        .html(`ARIMA Future Prediction<hr>${formatTime(d.predict_time)} <hr>$${currentVal.toFixed(2)}<br>`)
                        .style("left", d3.event.pageX - 90 + "px")
                        .style("top", d3.event.pageY - 140 + "px")
                        .style("background", d3.select(this).attr("fill"))
            }
        })
        .on("mouseout", function(){
            d3.selectAll("circle")
                .transition()
                .duration(500)
                .attr("r", 2)
            tooltip.style("display","none")

        })
}

// Function to draw lines based on Quintile number
// which_data_group: 0: actual; 1: piecewise_arima; 2: future_6_mon
function drawLine(data_Q, chartGroup, xTimeScale, yLinearScale, i, which_data_group){
    if ( which_data_group== 0){
        xtoken = 'monthend_date'
        ytoken = 'item_value'
    }
    else {
        xtoken = 'predict_time'
        ytoken = 'price'
    }
    var line = d3.line()
                    .x(d => xTimeScale(d[`${xtoken}`]))
                    .y(d => yLinearScale(d[`${ytoken}`]));
    var path = chartGroup.select(`#elementGroupQ${i}`)
                        .append("path")
                        .attr("d", line(data_Q))
                        .attr("stroke", markerColor(`Q${i}`))
                        .classed(`line Q${i}_line`, true);
    var totalLength = path.node().getTotalLength() + 2000;
    path
        .attr("stroke-dasharray", totalLength + " " + totalLength)
        .attr("stroke-dashoffset", totalLength)
        .transition().duration(3000)
        .attr("stroke-dashoffset", 0);
}

// Function to draw circle markers based on Quintile number
function drawMarkers(data_Q, chartGroup, xTimeScale, yLinearScale, i, which_data_group){
    var circleGroup = chartGroup
                    .selectAll(`#elementGroupQ${i}`)
                    .data(data_Q)
                    .enter()
                    .append("g")
    if ( which_data_group== 0){
        xtoken = 'monthend_date'
        ytoken = 'item_value'
    }
    else {
        xtoken = 'predict_time'
        ytoken = 'price'
    }
    circleGroup.append("circle")
                    .attr("cx", d => xTimeScale(d[`${xtoken}`]))
                    .attr("cy", d => yLinearScale(d[`${ytoken}`]))
                    .attr("r", 2)
                    .attr("stroke", markerColor(`Q${i}`))
                    .attr("fill", markerColor(`Q${i}`))
                    .classed(`circle Q${i} Group${which_data_group}`, true);
    circleGroup.transition().delay(1000)  
}


// function to update yScale
function make_yScale(actual, piecewise_arima, future_6_mon, chartHeight){

    var max_candidates = [d3.max(actual.map(actual => actual.item_value)),
                        d3.max(piecewise_arima.map(piecewise_arima => piecewise_arima.price)),
                        d3.max(future_6_mon.map(future_6_mon => future_6_mon.conf_int_hi))]
    var min_candidates = [d3.min(actual.map(actual => actual.item_value)),
                        d3.min(piecewise_arima.map(piecewise_arima => piecewise_arima.price)),
                        d3.min(future_6_mon.map(future_6_mon => future_6_mon.conf_int_lo))]
    var yMax = d3.max(max_candidates)
    var yMin = d3.min(min_candidates)
    var yLinearScale = d3.scaleLinear()
                        .range([chartHeight - 10, 0])
                        .domain([yMin * 0.95, yMax * 1.05])
    return yLinearScale;
}

function renderYAxis(newScale, yAxis){
    var LeftAxis = d3.axisLeft(newScale).tickFormat(d3.format("$.2f"));
    yAxis.transition().duration(1000).call(LeftAxis);
    return yAxis;
}

function renderMarkers(circles,XScale, YScale, xtoken, ytoken){
    circles.transition()
            .duration(1000)
            .attr("cx", d=> XScale(d[`${xtoken}`]))
            .attr("cy", d => YScale(d[`${ytoken}`]))
}

// Function to add "+" to persentChange
function sign(percentChange){
    if (percentChange>0) return "+"
    else return ""
}

// Function to select color based on Quintile
function markerColor(Q){
    switch (Q){
        case "Q1": return "#63A91F";
        case "Q2": return "#365542";
        case "Q3": return "black";
        case "Q4": return "#A40606";
        default: return "red";
    }
}

// Function to select color based on Quintile
function numQuintile(color){
    switch (color){
        case "#63A91F": return 1;
        case "#365542": return 2;
        case "black": return 3;
        case "#A40606": return 4;
        default: return 5;
    }
}