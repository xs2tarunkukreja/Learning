# Relevance to ML
Linear Transformation are used in both abstract mathematics as well as computer science. Linear tranformation with calculus are used as way of tracking changes, also known as derivatives.

Linear Transformation are often used in ML applications. They are useful in modeling 2D and 3D animation, where an object size and shape need to be transformed from one viewing angle to next.

An object can rotated and scaled with a space using a type of linear transformation as geometric transformation

Linear transformation also used to project the data from one space to another. Sometimes a dataset is not linear separable in its original space therefore we need to transform (project) the data in another space with different dimensions. This can be done either using linear tranformation or kernel.

# Linear Transformation
Let V and W be vectors space over a field F of dimension m and n respectively.
LT is a mapping T: V(n) -> W(m) such that
    T(v1 + v2) = T(v1) + T(v2)
    T(αv) = αT(v)

Rⁿ -> Rⁿ where n is 2 both side
T(x1, x2) = (x1, x1+x2)

Remarks -
    A linear map from T to itself is called linear operator.
    A linear map from a vector space to underlying field is called a linear function.

## Proof
Rⁿ -> Rⁿ where n is 2 both side
T(x1, x2) = (x1, x1+x2)

Let v1 = (x1, x2) and v2 = (y1, y2). Both are part of V space.
T(v1) = T(x1, x2) = (x1, x1+x2)
T(v2) = T(y1, y2) = (y1, y1+y2)

L.H.S. T(v1+v2) = T((x1, x2) + (y1, y2)) = T(x1+y1, x2+y2) = (x1+y1, x1+x2+y1+y2)
R.H.S. T(v1) + T(v2) = (x1, x1+x2) + (y1, y1+y2) = (x1+y1, x1+x2+y1+y2)
L.H.S = R.H.S

L.H.S. T(αv) = T(α(x1, x2)) = T(αx1, αx2) = (αx1, αx1+αx2)
R.H.S. αT(v) = αT(x1, x2) = α(x1, x1+x2) = (αx1, αx1+αx2) = L.H.S

# Geomatrical Representation of  Tranformation
T(x1, x2) = T(2x1, 2x2) = Scaling of vectors.

We have a square = {(0,0), (1,0), (1,1), (0,1)}

T(V) = Another Square = {(0,0), (2,0), (2,2), (0,2)}

Another example
T(x1, x2) = (x1, 2x2)
T(x1, x2) = (x1, 0)
T(x1, x2) = (x1cosθ-x2sinθ, x1sinθ+x2cosθ)
            cosθ -sinθ    x1
            sinθ  cosθ    x2
            Rotation Matrix (R)

# Transformation Function is Matrix
A(m*n) change a vector from m-D Space to n-D space.
T(v) = Av

T(x1,x2) = (2x1-7x2, 4x1+3x2)

Vector Space = {(1,3),(2,5)}
T(1,3) = (-19,13) = a11(1,3) + a21(2,5)
T(2,5) = (-31,23) = a12(1,3) + a22(2,5)

A = 121   201
    -70  -116

# Nullspace and Range of Linear Map
T: V -> W be a linear map.
Nullspace of T = null(T) ={v ∈ V| T(v) = 0}
and range of T 
    range(T) = {w ∈ W | v ∈ V then T(v) = w}

Remarks -
Nullspace of T is also called kernel of T.
Null(T) is a subspace of V
Range(T) is a subspace of W.
Dimension of Null(T) is called nullity of T.
Dimension of Range(T) is called rank of T.

## Rank Nulity Theorem
dim(range(T)) + dim(null(T)) = dim(V)

T(x1, x2, x3) = (x1-x2+x3, x2-x3, x1, 2x1-5x2+5x3)
Range(T) = {(1,0,1,2), (1, -1, 0, 5)}
null(T) = {(0,1,1)}