using DataFrames, DrWatson, GLM, UnicodePlots
@quickactivate "Causalties"
include("../src/generate_data.jl")



### Simulate data
data = simulate_data(200)
print(first(data, 5))

# We can see that we have 1000 data points in total, each with information 
# on the temperature that week, the number of houses broken into, and the 
# number of ice cream cones sold.



### Exploration
UnicodePlots.histogram(data[!, "Temperature"])
UnicodePlots.histogram(data[!, "Houses_broken_into"])
UnicodePlots.histogram(data[!, "Ice_cream_sold"])

UnicodePlots.scatterplot(data[!, "Temperature"], data[!, "Ice_cream_sold"])
UnicodePlots.scatterplot(data[!, "Temperature"], data[!, "Houses_broken_into"])
UnicodePlots.scatterplot(data[!, "Houses_broken_into"], data[!, "Ice_cream_sold"])



### Modelling
# Let's say were a local ice cream shop looking to better manage our supply chain,
# can we predict how much ice cream we'll need given this data?
fm = @formula(Ice_cream_sold ~ Houses_broken_into + Temperature)
linearRegressor = lm(fm, data)
println(linearRegressor)

# Let's now use our machine learning model to predict the ice cream cones sold based
# a new dataset. As this dataset was not used to build the model with, we can call the
# earlier dataset the "training" data, and this one the "testing" data.
new_data = simulate_data(20)
test_data = new_data[!, [:Temperature, :Houses_broken_into]]
test_outputs = new_data[!, :Ice_cream_sold]
ypredicted_test = predict(linearRegressor, test_data)

# We can check how accurate these predictions are 
performance_testdf = DataFrame(y_actual = test_outputs, y_predicted = ypredicted_test)
performance_testdf.error = performance_testdf[!,:y_actual] - performance_testdf[!,:y_predicted]
UnicodePlots.histogram(performance_testdf.error)

# In most cases, our predictions are off by less than a 1000 cones. We can also check 
# another performance metric called the R^2, which gives a figure between 0 and 1 on how 
# accurate our machine learning model is.
println(r2(linearRegressor))

# given the simplicity of a linear model like this, that's quite good!



### Querying the model
# Traditional machine learning models are based on association and co-occurrence of feature
# values. We can start to see how their performance starts to break down when we try to 
# ask them a causal question. For instance, let's take the last observation from our test
# data, and ask the model what would have happened if there was a huge crime spree and there
# were 10 times as many houses robbed.
counterfactual_observation = last(test_data)
counterfactual_observation[:Houses_broken_into] *= 10
println(counterfactual_observation)

counterfactual_predicions = predict(linearRegressor, DataFrame(counterfactual_observation))[1]
true_amount_sold = last(test_outputs)[1]

println("The model predicts $counterfactual_predicions ice cream cones instead of $true_amount_sold when there are 10 times as many houses robbed that day.")

# The model thus seems to attach importance to the fact that there were more houses robbed to
# make a prediction for how many ice cream cones will be sold, even though we intuitively 
# know that both #ice cream cones sold and #houses robbed will be influenced by the temperature
# and not by each other.