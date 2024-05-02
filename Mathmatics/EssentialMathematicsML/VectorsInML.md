# Vectors
Vector is a mathematical object that encode a length and direction.

More formally they are elements of a vector space: a collection of objects that is closed under an additional rule and a rule for multiplication by scalars.

It is represented as 1-D array. Column form or Row form.

Represented Geometrically, vector represent a cordinates within n-dimension space.

A simple representation of a vector is a arrow in a vector space, with an origin, direction and magnitude (length).

---
2D Space - Vector V1 is (1,2) 1 is point in x axis and 2 in y-axis.
ND Space - v = [v1, v2, ...., vn]

# Vector Algebra
Vector should be of same dimension.
v1 = (1, 3)
v2 = (1, 1)

Addition - v = v1 + v2 = (1+1, 3+1) = (2, 4)
Subtraction - v = v1 - v2 = (1-1, 3-1) = (0, 2)

Dot Product -
v1 = (x1, x2, ...., xn)
v2 = (y1, y2, ...., yn)
Dot Product would be scaler.
v1 * v2 = x1y1 + x2y2 + ... + xnyn = Σxiyi

Length / Magnitude - SQRT(x1x1+x2x2+ .... + xnxn) = Scalar = |v|

Angle between 2 vectors -
cos-1 : cos inverse = cos power -1.
cos-1 ((v1*v2)/(|v1| * |v2|))

Rⁿ: n Dimension Space

# Linear Combination of Vector
A Set s = {v1, v2, ... , vk}
A new vector v = α1v1 + α2v2 + ... + αkvk = It is linear combination of vectors v1, v2, ... , vk
where α1,....αk are scalar.


# Linear Independent and Dependent Vector
A set of vectors, s = {v1, v2, ... , vk} is linearly independent if
     v = α1v1 + α2v2 + ... + αkvk = 0 vector; only if α1 = .... = αk = 0

LD and LI

Remarks - 
1. In Rⁿ, a set of more than n vectors is LD.
2. Any set contain zero vector is LD

# Orthogonal Vectors
A set of vector {v1, v2, ... , vk} are mutually(pairwise) orthogonal if vi * vj = 0 for i != j

Orthonormal vector - A set of orthogonal vectors is orthonormal if each vector has length 1.

Remarks - A set of orthogonal vectors is LI

# Example of Feature Vector - Use of vector in ML
Height; Weights; Employee Id
e1, h1, w1 are observations.
Heights and Weights are features or attributes.

Now (h20, w20) is feature vector of 20th employee.

# All in Python