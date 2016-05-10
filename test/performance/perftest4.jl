using Judyp


function perftest(n=1_000_000)
    s = Judyp.genvaluefunstate([7, 7, 10, 7], [0.5, 700., 0., 0.], [7., 2250., 1., 6.])
    state = [4., 801., 0.3, 4.5]
    c = ones(7*7*10*7)

    r = 0.

    for i=1:n
        r = r + Judyp.valuefun(state, c, s)
    end
    return r
end

perftest(2)

@time perftest()
