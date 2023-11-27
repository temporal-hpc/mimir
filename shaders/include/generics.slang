#pragma once
; // VSCode slang extension complains without this

struct VectorInput<T, let N : int>
{
    vector<T,N> data;
};

vector<float,N> convertInt<let N:int>(vector<int,N> vec)
{
    vector<float,N> res;
    [unroll]
    for (int i = 0; i < N; ++i) { res[i] = float(vec[i]); }
    return res;
}

vector<float,N> convertHalf<let N:int>(vector<half,N> vec)
{
    vector<float,N> res;
    [unroll]
    for (int i = 0; i < N; ++i) { res[i] = float(vec[i]); }
    return res;
}

vector<float,N> convertFloat<let N:int>(vector<float,N> vec)
{
    vector<float,N> res;
    [unroll]
    for (int i = 0; i < N; ++i) { res[i] = float(vec[i]); }
    return res;
}

vector<float,N> convertDouble<let N:int>(vector<double,N> vec)
{
    vector<float,N> res;
    [unroll]
    for (int i = 0; i < N; ++i) { res[i] = float(vec[i]); }
    return res;
}