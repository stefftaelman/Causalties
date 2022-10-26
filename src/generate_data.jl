using DataFrames, Distributions, DrWatson
@quickactivate "Causalties"

"""
function to generate some statistics for a city for a single month.   
"""
function simulate_monthly_stat()
    month_to_temp = Dict(:jan=>(15, 2.5), :feb=>(15.5, 2.5), :mar=>(16, 2.5), :apr=>(18, 2.5), 
                         :may=>(18.5, 2), :jun=>(19.5, 2), :jul=>(21, 3), :aug=>(21, 2.5), 
                         :sep=>(21, 3), :oct=>(19.5, 2.5), :nov=>(18, 3), :dec=>(15, 3))
    month = rand(keys(month_to_temp))
    T_outside = rand(Normal(month_to_temp[month]...))
    #AC_usage = T_outside >= 25
    #T_inside =
    #crime_rates = T_outside < 12 ? (20, 7) : T_outside < 20 ? (70, 20) : T_outside < 25 ? (150, 70) : (300, 70)
    crimes(temp) = round(temp * rand(Beta(temp, 200)) * 70)
    houses_broken_into = round(crimes(T_outside))
    cones(temp) = temp * rand(Normal(log10(temp)*log2(temp), 1)) / 300
    ice_cream_sold = rand(Binomial(10000, cones(T_outside)))
    return [T_outside houses_broken_into ice_cream_sold]
end

"""
function to generate N observations of some system.
"""
function simulate_data(N::Integer; f=simulate_monthly_stat::Function, colnames=["Temperature", "Houses_broken_into", "Ice_cream_sold"])
    data_mat = vcat([f() for i in 1:N]...)
    data = DataFrame(data_mat, colnames)
    return data
end
