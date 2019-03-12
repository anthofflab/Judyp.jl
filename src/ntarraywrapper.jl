struct NTArrayWrapper{T,NAMES,SIZES}
    data::T
end

@generated function _getproperty(ntaw::NTArrayWrapper{T,NAMES,SIZES}, name::Val{NAME}) where {T,NAMES,SIZES,NAME}
    name_index = findfirst(i->i==NAME, NAMES)
    array_index = 1
    for i=1:name_index-1
        if SIZES[i]===nothing
            array_index += 1
        else
            array_index += SIZES[i]
        end
    end

    if SIZES[name_index]===nothing
        return :(getfield(ntaw, :data)[$array_index])
    else
        return :(view(getfield(ntaw, :data), $array_index:$(array_index+SIZES[name_index]-1)))        
    end
end

function Base.getproperty(ntaw::NTArrayWrapper, name::Symbol)
    return _getproperty(ntaw, Val(name))
end
