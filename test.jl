
mutable struct Foo
    x::Float64
    y::Float64
end

Foo(x::Float64)=Foo(x,0.0)

foo=Foo(3.0)
