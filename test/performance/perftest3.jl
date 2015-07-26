using CompEcon
include("fchebtensor.jl")

function f1()
    vs1 = genvaluefunstate([10,10,10,10], [0.,0.,0.,0.],[20.,20.,20.,20.])
    c = ones(10^4)

    x = [3., 12., 45., 3.]

    @time for i=1:1000000
        y = valuefun(x, c, vs1)
    end
end

function f2()
    vs2 = fundefn(:cheb, [10,10,10,10], [0.,0.,0.,0.],[20.,20.,20.,20.])
    c = ones(10^4)

    x = [3., 12., 45., 3.]'

    @time for i=1:1000000
        y = funeval(c, vs2, x)[1][1]
    end
end

f1()
f2()




