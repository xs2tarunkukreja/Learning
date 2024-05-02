# Matrix
2 D Array of scalers with 1 or more columns and 1 or more rows.
R X C to tell the size of matrix

aij is an matrix element. 1 <= i <= R and 1 <= j <= C

a11 a12 a13
a21 a22 a23
a31 a32 a33

# Diagonal and Triangular Matrix
Diagonal Matrix - A square matrix whose off-diagnol elements are all zero.
2 0 0              0 0 0
0 3 0              0 0 0            
0 0 1              0 0 0
Triangular Matrix - A square matrix whose elements above or below the main diagonal are all zero.
5 8 7
0 3 8  = Upper Triangular Matrix [Other is lower.]
0 0 1 

# Identity Matrix
A diagonal matrix whose all diagonal entries are all 1.

# Matrix Algebra
Two matrices are equal if their dimensions are equal and all corresponding elements are equal.

## Addition and Subtraction
cij = aij +/- bij for all i, j
a and b should be same size and c will also have similar size.

Cumulative Law: A + B = B + A
Associative Law: A + (B + C) = (A + B) + C = A + B + C

Remarks = Matices of different size can't added or subtracted.

## Scalar Multiplication
Multiply contant with all elements.

α * aij

## Matrix Multiplication
Necessary condition for multiplication of 2 matrices A and B is that the number of column of A must be equal to number of row of B.

A(5*3) X B(3*4) = C(5*4)

m = row of A
k = column of A = row of B
n = column of B

cij = ai1 * b1j + ai2 * b2j + ... + aik * bkj = (ith row of A) * (jth column of B) = Dot product of 2 vector

### Rules
AI = IA = A
A(BC) = (AB)C = ABC = Associative
A(B+C) = AB + AC - First Distributive Rule
(A+B)C = AC + BC - Second Distributive Rule

AB != BA
If AB = 0; Neither A Nor B necessary = 0
IF AB = AC; B may be or may not be = C


# Transpose of a Matrix
(A+B)t [t is T in superscript]
(A+B)t = At + Bt
(AB)t = Bt At
(kA)t = k*At
(At)t = A

Transpose: A = {aij} = n * m
           At = {aji} = m * n

# Determinant of Matrix
Every square matrix have determinant.
Determinant of matrix is scalar number.
A = 1 -1
    2  3
|A| = 1 * 3 - (-1 * 2) = 3 + 2 = 5

# Inverse of a Matrix
Scalar k, inverse is reciprocal or division of 1 by k.
k = 7; k^-1 = 1/7

A * I(A) = I(A) * A = Identity Matrix

If |A| = 0 then I(A) doesn't exist or find.

I(AB) = I(B)*I(A)
I(I(A)) = A
I(At) = I(A)t
I(kA) = 1/k * I(A)

A square matrix that has an inverse are called a nonsingular matrix.
A matrix that doesn't have inverse is called singular. |A| = 0
