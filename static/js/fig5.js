// Run the program
var carousel_div = d3.select("#carousel-imgs")
var carousel_ind = d3.select("#carousel-indicators")
init();

function init() {
  makeHomeButtons()
  createCarousels()
  add_FAQ()
};

function createCarousels(){
  for (var i = 1; i<=10; i++){
    if (i == 1){
      carousel_ind.append("li").attr("data-target", '#myCarousel')
                  .attr("data-slide-to",`${i-1}`)
                  .attr('class', 'active')
      var img_container = carousel_div.append("div").attr("class","item active")
                                      .attr("margin", "auto !important")
    }
    else {
      carousel_ind.append("li").attr("data-target", '#myCarousel')
                  .attr("data-slide-to",`${i-1}`)
      var img_container = carousel_div.append("div").attr("class","item").attr("margin", "auto")
    }
    path = `../static/imgs/MLPorfolioIndex/Brad_slide-show-${i}.jpg`
    img_container.append("img")
                .attr("src", path)
                .attr("width", '75%')
                .classed("img-responsive center-block", true)

  }
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

function add_FAQ(){
  var parent = d3.select("#accordion-tab-1")
    for (var i = 0; i<x.length; i++){
      var card = parent.append("div").classed("card", true)
      var card_header = card.append("div").classed("card-header", true).attr("id", `accordion-tab-1-heading-${i}`)
                            .append("h5")
      button = card_header.append("button").attr("class","btn btn-link").html(x[i]['header-html'])
                            .attr("type","button")
                            .attr("data-toggle","collapse")
                            .attr("data-target",`#accordion-tab-1-content-${i}`)
                            .attr("aria-expanded","false")
                            .attr("aria-controls",`accordion-tab-1-content-${i}`)
                            
      var card_body = card.append("div").classed("collapse", true).attr("id",`accordion-tab-1-content-${i}`)
                          .attr("aria-labelledby", `accordion-tab-1-heading-${i}`)
                          .attr("aria-expanded","false")
                          .attr("data-parent","#accordion-tab-1")
                          .append("div").classed("card-body", true)
                          .html(x[i]['content-html'])
    }
}