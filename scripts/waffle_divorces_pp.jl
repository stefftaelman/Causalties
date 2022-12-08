###############################################################################
######= EXAMPLE THREE: Causal inference with probabilistic programming =#######
###############################################################################

using DrWatson
@quickactivate "Causalties"
using CSV, DataFrames, GLM, GraphPlot, Graphs, LightGraphs, StatsBase, StatsPlots, StatisticalRethinkingPlots, Turing

# In this script, we'll use the Waffle House example once more, but this time 
# we'll try to solve it using bayesian regression in Turing, a probabilistic 
# programming language (PPL).

### Load in the data
data = DataFrame(CSV.File("data/WaffleDivorce.csv"))

# We'll once again ask the unlikely question: Does the number of Waffle House
# diners influence the divorce rate? We so far assumed the following system:

g = LightGraphs.SimpleDiGraph(5)
LightGraphs.add_edge!(g, 2, 1); LightGraphs.add_edge!(g, 2, 3); 
LightGraphs.add_edge!(g, 3, 1); LightGraphs.add_edge!(g, 4, 2); 
LightGraphs.add_edge!(g, 4, 3); LightGraphs.add_edge!(g, 4, 5); 
LightGraphs.add_edge!(g, 5, 1);
gplot(
    g, 
    layout=circular_layout,
    nodelabel=
        ["Divorce", "MedianAgeMarriage", "Marriage", "South", "WaffleHouses"], 
    nodelabelc="white"
    )

# using probabilistic programming means that we'll need to simulate new 
# datapoints to fit to the ones we've observed. To do this, we'll need more 
# than a DAG. We need a set of functions that tell us how each variable is 
# generated.

### probabilistic programming

# First, we'll standardize our variables of interest though, as this makes 
# everything downstream easier*.*Note: this is not only useful for probabilistic
# programming and is also a common practise in most traditional machine learning 
# models.

standardized_data = DataFrame(
    :D => standardize(ZScoreTransform, data.Divorce),
    :M => standardize(ZScoreTransform, data.Marriage),
    :A => standardize(ZScoreTransform, data.MedianAgeMarriage),
    :W => standardize(ZScoreTransform, Float64.(data.WaffleHouses)),
    :S => data.South
);

# For simplicity, we'll use Guassian (normal) distributions for each variable.
# The point here is to try to set out our assumptions for the data-generating
# process. The strategy to set this out is as follows:
#   (i)     Nominate the predictor variables you want in the linear model of 
#           the mean.
#   (ii)    For each predictor, make a paramater that will measure its 
#           conditional association to the outcome.
#   (iii)   Multiply the parameter by the variable and add that term to the 
#           linear model.
#   (from: Statistical Rethinking, 2nd edition - Richard McElreath)

@model function waffles(S, A, M, W, D)
    # A, M, and W cause D
    σ ~ Exponential(1)                          # prior for σ
    α ~ Normal(0, 0.2)                          # prior for α
    β_A ~ Normal(0, 0.5)                        # prior for β_A
    β_M ~ Normal(0, 0.5)                        # prior for β_M
    β_W ~ Normal(0, 0.5)                        # prior for β_W
    μ = @. α + β_A * A + β_M * M + β_W * W      # multivariate linear model
    D ~ MvNormal(μ, σ)                          # probability of the data

    # S -> M <- A
    σ_M ~ Exponential(1)
    α_M ~ Normal(0, 0.2)
    β_SM ~ Normal(0, 0.5)
    β_AM ~ Normal(0, 0.5)
    μ_M = @. α_M + β_SM * S + β_AM * A
    M ~ MvNormal(μ_M, σ_M)

    # S -> A
    σ_A ~ Exponential(1)
    α_A ~ Normal(0, 0.2)
    β_SA ~ Normal(0, 0.5)
    μ_A = @. α_A + β_SA * S
    A ~ MvNormal(μ_A, σ_A)

    # S -> W
    σ_W ~ Exponential(1)
    α_W ~ Normal(0, 0.2)
    β_SW ~ Normal(0, 0.5)
    μ_W = @. α_W + β_SW * S
    W ~ MvNormal(μ_W, σ_W)
end

model = waffles(
    standardized_data.S, 
    standardized_data.A,
    standardized_data.M, 
    standardized_data.W, 
    standardized_data.D
    );
chain = sample(model, NUTS(), 1000)

### interpretation
plot(chain) # this is kind of a messy plot, but take a look at it nonetheless
coeftab_plot(   # I like this one better
    DataFrame(chain); 
    pars=(:β_A, :β_M, :β_W, :β_AM, :β_SM, :β_SA, :β_SW)
    )


# Take a look at the variables' mean as estimated by our model. We see that,
# for instance, MedianAgeMarriage (A) and Marriage (M) are very negatively 
# associated and that the south and waffle house are very positively associated

# How would you make predictions with this model?

# How would you make interventions on this model?

# How would you do counterfactual analysis with this model?