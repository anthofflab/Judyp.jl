function our_status(m)
    try
        return status(mp)
    catch err
        @warn "The `status` function has thrown an exception, which it probably shouldn't."
        return :FAILURE
    end
end