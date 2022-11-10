###############################################################################
#####= EXAMPLE TWO: Causal inference with smart multivariate regression =######
###############################################################################

using DrWatson
@quickactivate "Causalties"
using CSV, DataFrames, DrWatson, GLM, GraphPlot, Graphs, StatsBase, UnicodePlots

# In this example, we'll use an example from Richard McElreath's 'Statistical 
# Rethinking (2nd edition)'. It goes as follows:
# "One of the most reliable sources of waffles in North America, if not the 
# entire world, is a Waffle House diner (https://en.wikipedia.org/wiki/Waffle_House). 
# Waffle house is nearly always open, even just after a hurricane. Most diners 
# invest in disaster preparedness, including having their own electrical 
# generators. As a consequence, the United States' disaster relief agency (FEMA)
# informally uses Waffle House as an index of disaster severity. If the Waffle 
# House is closed, that's a serious event."

### Load in the data
data = DataFrame(CSV.File("data/WaffleDivorce.csv"))

# This real-world data sets out some facts for each US state, including the number
# of Waffle House locations they have. Given what we just read above, it is 
# striking that Waffle House thriving in a state is associated with the US'
# highest divorce rates. McElreath asks: "Could always available waffles and hash
# brown potatoes put marriage at risk?" How would we go about answering this?

waffle_divorce_cor = round(corspearman(data.WaffleHouses, data.Divorce), digits=3)
println(
    """
    The Spearman-Rank correlation between the number of Waffle Houses and divorce 
    rate is $waffle_divorce_cor.
    """
    )

# As an exercise, take a minute to look at the data and try to think about how this
# phenomenon could arise. HINT: Waffle House had it's first location in Georgia, a 
# state in the nation's South.


### A naive solution
# Using the machine learning solutions we saw in the `ice_cream_burglars.jl` example,
# program a machine learning model to predict the divorce rate!
clf = ...

# And now use it to predict what would happen to the divorce rate if half the number
# of Waffle Houses in Alabama were shut down:
intervened_data = ...
predicted_divorce_rate = ...

# ... If we think about the mechanisms behind this system beforehand, we might be 
# able to come to a more accurate solution.

### A smart solution
# If you don't have a complete mechanistic model of your system, Directed Acyclic 
# Graphs (DAGs) can be amazing tools. They make assumptions transparant and easy to 
# critique. Given all the variables we see, we can somewhat safely assume the 
# following model: 

g = SimpleDiGraph(5)
add_edge!(g, 2, 1); add_edge!(g, 2, 3); add_edge!(g, 3, 1); add_edge!(g, 4, 2); 
add_edge!(g, 4, 3); add_edge!(g, 4, 5); add_edge!(g, 5, 1);
gplot(
    g, 
    layout=spring_layout,
    nodelabel=["Divorce", "MedianAgeMarriage", "Marriage", "South", "WaffleHouses"], 
    nodelabelc="white"
    )

# Now using what you know from do-calculus, to accurately estimate the effect of 
# Waffle Houses on Divorce rate, what variables should we adjust on in this model?
# In practise, you can also always use daggity.net or an implementation of it.
adjustment_set = ...

# Since 'South' is a binary variable, we can easily estimate the effect by 
# stratifying on it, i.e., looking at the effect WaffleHouse -> Divorce when we're
# (i) only looking at northern states and (ii) only looking at southern states and
# averaging the two.
northern_states = subset(data, :South => ByRow(==(0)))
southern_states = subset(data, :South => ByRow(==(1)))

fm = @formula(Divorce ~ WaffleHouses)
linearRegressor_north = lm(fm, northern_states)
println(linearRegressor_north)
linearRegressor_south = lm(fm, southern_states)
println(linearRegressor_south)

causal_estimate = coef(linearRegressor_north)[2]*(first(size(northern_states))/first(size(data))) + coef(linearRegressor_south)[2]*(first(size(southern_states))/first(size(data)))
