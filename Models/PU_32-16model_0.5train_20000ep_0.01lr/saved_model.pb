ΥΧ$
Φ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Expm1
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	

RandomUniformInt

shape"T
minval"Tout
maxval"Tout
output"Tout"
seedint "
seed2int "
Touttype:
2	"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
@
Softplus
features"T
activations"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
Α
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
^
StatelessRandomGetKeyCounter
seed"Tseed
key
counter"
Tseedtype0	:
2	
΅
StatelessRandomUniformIntV2
shape"Tshape
key
counter
alg
minval"dtype
maxval"dtype
output"dtype"
dtypetype:
2	"
Tshapetype0:
2	
@
StaticRegexFullMatch	
input

output
"
patternstring
φ
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68©Ε#

 dense_tfp_1/kernel_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape
:g *1
shared_name" dense_tfp_1/kernel_posterior_loc

4dense_tfp_1/kernel_posterior_loc/Read/ReadVariableOpReadVariableOp dense_tfp_1/kernel_posterior_loc*
_output_shapes

:g *
dtype0
Ό
0dense_tfp_1/kernel_posterior_untransformed_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape
:g *A
shared_name20dense_tfp_1/kernel_posterior_untransformed_scale
΅
Ddense_tfp_1/kernel_posterior_untransformed_scale/Read/ReadVariableOpReadVariableOp0dense_tfp_1/kernel_posterior_untransformed_scale*
_output_shapes

:g *
dtype0

dense_tfp_1/bias_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name dense_tfp_1/bias_posterior_loc

2dense_tfp_1/bias_posterior_loc/Read/ReadVariableOpReadVariableOpdense_tfp_1/bias_posterior_loc*
_output_shapes
: *
dtype0

 dense_tfp_2/kernel_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *1
shared_name" dense_tfp_2/kernel_posterior_loc

4dense_tfp_2/kernel_posterior_loc/Read/ReadVariableOpReadVariableOp dense_tfp_2/kernel_posterior_loc*
_output_shapes

: *
dtype0
Ό
0dense_tfp_2/kernel_posterior_untransformed_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *A
shared_name20dense_tfp_2/kernel_posterior_untransformed_scale
΅
Ddense_tfp_2/kernel_posterior_untransformed_scale/Read/ReadVariableOpReadVariableOp0dense_tfp_2/kernel_posterior_untransformed_scale*
_output_shapes

: *
dtype0

dense_tfp_2/bias_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name dense_tfp_2/bias_posterior_loc

2dense_tfp_2/bias_posterior_loc/Read/ReadVariableOpReadVariableOpdense_tfp_2/bias_posterior_loc*
_output_shapes
:*
dtype0

output/kernel_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*,
shared_nameoutput/kernel_posterior_loc

/output/kernel_posterior_loc/Read/ReadVariableOpReadVariableOpoutput/kernel_posterior_loc*
_output_shapes

:	*
dtype0
²
+output/kernel_posterior_untransformed_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*<
shared_name-+output/kernel_posterior_untransformed_scale
«
?output/kernel_posterior_untransformed_scale/Read/ReadVariableOpReadVariableOp+output/kernel_posterior_untransformed_scale*
_output_shapes

:	*
dtype0

output/bias_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape:	**
shared_nameoutput/bias_posterior_loc

-output/bias_posterior_loc/Read/ReadVariableOpReadVariableOpoutput/bias_posterior_loc*
_output_shapes
:	*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
ͺ
'Adam/dense_tfp_1/kernel_posterior_loc/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:g *8
shared_name)'Adam/dense_tfp_1/kernel_posterior_loc/m
£
;Adam/dense_tfp_1/kernel_posterior_loc/m/Read/ReadVariableOpReadVariableOp'Adam/dense_tfp_1/kernel_posterior_loc/m*
_output_shapes

:g *
dtype0
Κ
7Adam/dense_tfp_1/kernel_posterior_untransformed_scale/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:g *H
shared_name97Adam/dense_tfp_1/kernel_posterior_untransformed_scale/m
Γ
KAdam/dense_tfp_1/kernel_posterior_untransformed_scale/m/Read/ReadVariableOpReadVariableOp7Adam/dense_tfp_1/kernel_posterior_untransformed_scale/m*
_output_shapes

:g *
dtype0
’
%Adam/dense_tfp_1/bias_posterior_loc/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/dense_tfp_1/bias_posterior_loc/m

9Adam/dense_tfp_1/bias_posterior_loc/m/Read/ReadVariableOpReadVariableOp%Adam/dense_tfp_1/bias_posterior_loc/m*
_output_shapes
: *
dtype0
ͺ
'Adam/dense_tfp_2/kernel_posterior_loc/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *8
shared_name)'Adam/dense_tfp_2/kernel_posterior_loc/m
£
;Adam/dense_tfp_2/kernel_posterior_loc/m/Read/ReadVariableOpReadVariableOp'Adam/dense_tfp_2/kernel_posterior_loc/m*
_output_shapes

: *
dtype0
Κ
7Adam/dense_tfp_2/kernel_posterior_untransformed_scale/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *H
shared_name97Adam/dense_tfp_2/kernel_posterior_untransformed_scale/m
Γ
KAdam/dense_tfp_2/kernel_posterior_untransformed_scale/m/Read/ReadVariableOpReadVariableOp7Adam/dense_tfp_2/kernel_posterior_untransformed_scale/m*
_output_shapes

: *
dtype0
’
%Adam/dense_tfp_2/bias_posterior_loc/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/dense_tfp_2/bias_posterior_loc/m

9Adam/dense_tfp_2/bias_posterior_loc/m/Read/ReadVariableOpReadVariableOp%Adam/dense_tfp_2/bias_posterior_loc/m*
_output_shapes
:*
dtype0
 
"Adam/output/kernel_posterior_loc/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*3
shared_name$"Adam/output/kernel_posterior_loc/m

6Adam/output/kernel_posterior_loc/m/Read/ReadVariableOpReadVariableOp"Adam/output/kernel_posterior_loc/m*
_output_shapes

:	*
dtype0
ΐ
2Adam/output/kernel_posterior_untransformed_scale/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*C
shared_name42Adam/output/kernel_posterior_untransformed_scale/m
Ή
FAdam/output/kernel_posterior_untransformed_scale/m/Read/ReadVariableOpReadVariableOp2Adam/output/kernel_posterior_untransformed_scale/m*
_output_shapes

:	*
dtype0

 Adam/output/bias_posterior_loc/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" Adam/output/bias_posterior_loc/m

4Adam/output/bias_posterior_loc/m/Read/ReadVariableOpReadVariableOp Adam/output/bias_posterior_loc/m*
_output_shapes
:	*
dtype0
ͺ
'Adam/dense_tfp_1/kernel_posterior_loc/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:g *8
shared_name)'Adam/dense_tfp_1/kernel_posterior_loc/v
£
;Adam/dense_tfp_1/kernel_posterior_loc/v/Read/ReadVariableOpReadVariableOp'Adam/dense_tfp_1/kernel_posterior_loc/v*
_output_shapes

:g *
dtype0
Κ
7Adam/dense_tfp_1/kernel_posterior_untransformed_scale/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:g *H
shared_name97Adam/dense_tfp_1/kernel_posterior_untransformed_scale/v
Γ
KAdam/dense_tfp_1/kernel_posterior_untransformed_scale/v/Read/ReadVariableOpReadVariableOp7Adam/dense_tfp_1/kernel_posterior_untransformed_scale/v*
_output_shapes

:g *
dtype0
’
%Adam/dense_tfp_1/bias_posterior_loc/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/dense_tfp_1/bias_posterior_loc/v

9Adam/dense_tfp_1/bias_posterior_loc/v/Read/ReadVariableOpReadVariableOp%Adam/dense_tfp_1/bias_posterior_loc/v*
_output_shapes
: *
dtype0
ͺ
'Adam/dense_tfp_2/kernel_posterior_loc/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *8
shared_name)'Adam/dense_tfp_2/kernel_posterior_loc/v
£
;Adam/dense_tfp_2/kernel_posterior_loc/v/Read/ReadVariableOpReadVariableOp'Adam/dense_tfp_2/kernel_posterior_loc/v*
_output_shapes

: *
dtype0
Κ
7Adam/dense_tfp_2/kernel_posterior_untransformed_scale/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *H
shared_name97Adam/dense_tfp_2/kernel_posterior_untransformed_scale/v
Γ
KAdam/dense_tfp_2/kernel_posterior_untransformed_scale/v/Read/ReadVariableOpReadVariableOp7Adam/dense_tfp_2/kernel_posterior_untransformed_scale/v*
_output_shapes

: *
dtype0
’
%Adam/dense_tfp_2/bias_posterior_loc/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/dense_tfp_2/bias_posterior_loc/v

9Adam/dense_tfp_2/bias_posterior_loc/v/Read/ReadVariableOpReadVariableOp%Adam/dense_tfp_2/bias_posterior_loc/v*
_output_shapes
:*
dtype0
 
"Adam/output/kernel_posterior_loc/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*3
shared_name$"Adam/output/kernel_posterior_loc/v

6Adam/output/kernel_posterior_loc/v/Read/ReadVariableOpReadVariableOp"Adam/output/kernel_posterior_loc/v*
_output_shapes

:	*
dtype0
ΐ
2Adam/output/kernel_posterior_untransformed_scale/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*C
shared_name42Adam/output/kernel_posterior_untransformed_scale/v
Ή
FAdam/output/kernel_posterior_untransformed_scale/v/Read/ReadVariableOpReadVariableOp2Adam/output/kernel_posterior_untransformed_scale/v*
_output_shapes

:	*
dtype0

 Adam/output/bias_posterior_loc/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" Adam/output/bias_posterior_loc/v

4Adam/output/bias_posterior_loc/v/Read/ReadVariableOpReadVariableOp Adam/output/bias_posterior_loc/v*
_output_shapes
:	*
dtype0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
\
Const_1Const*
_output_shapes

:g *
dtype0*
valueBg *    
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
\
Const_3Const*
_output_shapes

: *
dtype0*
valueB *    
L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
\
Const_5Const*
_output_shapes

:	*
dtype0*
valueB	*    

NoOpNoOp
K
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*ΝJ
valueΓJBΐJ BΉJ
Α
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*
Ε
kernel_posterior_loc
($kernel_posterior_untransformed_scale
kernel_posterior
kernel_prior
bias_posterior_loc
bias_posterior
kernel_posterior_affine
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
Ε
kernel_posterior_loc
($kernel_posterior_untransformed_scale
kernel_posterior
kernel_prior
bias_posterior_loc
bias_posterior
 kernel_posterior_affine
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses*
Ε
'kernel_posterior_loc
(($kernel_posterior_untransformed_scale
)kernel_posterior
*kernel_prior
+bias_posterior_loc
,bias_posterior
-kernel_posterior_affine
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses*
ς
4iter

5beta_1

6beta_2
	7decay
8learning_ratemzm{m|m}m~m'm(m+mvvvvvv'v(v+v*
C
0
1
2
3
4
5
'6
(7
+8*
C
0
1
2
3
4
5
'6
(7
+8*
* 
°
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
* 
* 
* 

>serving_default* 
~x
VARIABLE_VALUE dense_tfp_1/kernel_posterior_locDlayer_with_weights-0/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE0dense_tfp_1/kernel_posterior_untransformed_scaleTlayer_with_weights-0/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUE*
+
?_distribution
@_graph_parents*
)
A_distribution
B_graph_parents* 
zt
VARIABLE_VALUEdense_tfp_1/bias_posterior_locBlayer_with_weights-0/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUE*
+
C_distribution
D_graph_parents*
$

E_scale
F_graph_parents*

0
1
2*

0
1
2*
* 

Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
~x
VARIABLE_VALUE dense_tfp_2/kernel_posterior_locDlayer_with_weights-1/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE0dense_tfp_2/kernel_posterior_untransformed_scaleTlayer_with_weights-1/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUE*
+
L_distribution
M_graph_parents*
)
N_distribution
O_graph_parents* 
zt
VARIABLE_VALUEdense_tfp_2/bias_posterior_locBlayer_with_weights-1/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUE*
+
P_distribution
Q_graph_parents*
$

R_scale
S_graph_parents*

0
1
2*

0
1
2*
* 

Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
* 
* 
ys
VARIABLE_VALUEoutput/kernel_posterior_locDlayer_with_weights-2/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+output/kernel_posterior_untransformed_scaleTlayer_with_weights-2/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUE*
+
Y_distribution
Z_graph_parents*
)
[_distribution
\_graph_parents* 
uo
VARIABLE_VALUEoutput/bias_posterior_locBlayer_with_weights-2/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUE*
+
]_distribution
^_graph_parents*
$

__scale
`_graph_parents*

'0
(1
+2*

'0
(1
+2*
* 

anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

f0
g1*
* 
* 
* 
.
_loc

E_scale
h_graph_parents*
* 

i_graph_parents* 
* 
"
_loc
j_graph_parents*
* 

_pretransformed_input*
* 
* 
* 
* 
* 
* 
.
_loc

R_scale
k_graph_parents*
* 

l_graph_parents* 
* 
"
_loc
m_graph_parents*
* 

_pretransformed_input*
* 
* 
* 
* 
* 
* 
.
'_loc

__scale
n_graph_parents*
* 

o_graph_parents* 
* 
"
+_loc
p_graph_parents*
* 

(_pretransformed_input*
* 
* 
* 
* 
* 
* 
8
	qtotal
	rcount
s	variables
t	keras_api*
H
	utotal
	vcount
w
_fn_kwargs
x	variables
y	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

q0
r1*

s	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

u0
v1*

x	variables*
’
VARIABLE_VALUE'Adam/dense_tfp_1/kernel_posterior_loc/m`layer_with_weights-0/kernel_posterior_loc/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Β»
VARIABLE_VALUE7Adam/dense_tfp_1/kernel_posterior_untransformed_scale/mplayer_with_weights-0/kernel_posterior_untransformed_scale/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE%Adam/dense_tfp_1/bias_posterior_loc/m^layer_with_weights-0/bias_posterior_loc/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
’
VARIABLE_VALUE'Adam/dense_tfp_2/kernel_posterior_loc/m`layer_with_weights-1/kernel_posterior_loc/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Β»
VARIABLE_VALUE7Adam/dense_tfp_2/kernel_posterior_untransformed_scale/mplayer_with_weights-1/kernel_posterior_untransformed_scale/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE%Adam/dense_tfp_2/bias_posterior_loc/m^layer_with_weights-1/bias_posterior_loc/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/output/kernel_posterior_loc/m`layer_with_weights-2/kernel_posterior_loc/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
½Ά
VARIABLE_VALUE2Adam/output/kernel_posterior_untransformed_scale/mplayer_with_weights-2/kernel_posterior_untransformed_scale/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/output/bias_posterior_loc/m^layer_with_weights-2/bias_posterior_loc/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
’
VARIABLE_VALUE'Adam/dense_tfp_1/kernel_posterior_loc/v`layer_with_weights-0/kernel_posterior_loc/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Β»
VARIABLE_VALUE7Adam/dense_tfp_1/kernel_posterior_untransformed_scale/vplayer_with_weights-0/kernel_posterior_untransformed_scale/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE%Adam/dense_tfp_1/bias_posterior_loc/v^layer_with_weights-0/bias_posterior_loc/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
’
VARIABLE_VALUE'Adam/dense_tfp_2/kernel_posterior_loc/v`layer_with_weights-1/kernel_posterior_loc/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Β»
VARIABLE_VALUE7Adam/dense_tfp_2/kernel_posterior_untransformed_scale/vplayer_with_weights-1/kernel_posterior_untransformed_scale/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE%Adam/dense_tfp_2/bias_posterior_loc/v^layer_with_weights-1/bias_posterior_loc/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/output/kernel_posterior_loc/v`layer_with_weights-2/kernel_posterior_loc/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
½Ά
VARIABLE_VALUE2Adam/output/kernel_posterior_untransformed_scale/vplayer_with_weights-2/kernel_posterior_untransformed_scale/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/output/bias_posterior_loc/v^layer_with_weights-2/bias_posterior_loc/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
x
serving_default_inputPlaceholder*'
_output_shapes
:?????????g*
dtype0*
shape:?????????g
Ξ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input0dense_tfp_1/kernel_posterior_untransformed_scale dense_tfp_1/kernel_posterior_locdense_tfp_1/bias_posterior_locConstConst_10dense_tfp_2/kernel_posterior_untransformed_scale dense_tfp_2/kernel_posterior_locdense_tfp_2/bias_posterior_locConst_2Const_3+output/kernel_posterior_untransformed_scaleoutput/kernel_posterior_locoutput/bias_posterior_locConst_4Const_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*+
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference_signature_wrapper_70056105
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ΰ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename4dense_tfp_1/kernel_posterior_loc/Read/ReadVariableOpDdense_tfp_1/kernel_posterior_untransformed_scale/Read/ReadVariableOp2dense_tfp_1/bias_posterior_loc/Read/ReadVariableOp4dense_tfp_2/kernel_posterior_loc/Read/ReadVariableOpDdense_tfp_2/kernel_posterior_untransformed_scale/Read/ReadVariableOp2dense_tfp_2/bias_posterior_loc/Read/ReadVariableOp/output/kernel_posterior_loc/Read/ReadVariableOp?output/kernel_posterior_untransformed_scale/Read/ReadVariableOp-output/bias_posterior_loc/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp;Adam/dense_tfp_1/kernel_posterior_loc/m/Read/ReadVariableOpKAdam/dense_tfp_1/kernel_posterior_untransformed_scale/m/Read/ReadVariableOp9Adam/dense_tfp_1/bias_posterior_loc/m/Read/ReadVariableOp;Adam/dense_tfp_2/kernel_posterior_loc/m/Read/ReadVariableOpKAdam/dense_tfp_2/kernel_posterior_untransformed_scale/m/Read/ReadVariableOp9Adam/dense_tfp_2/bias_posterior_loc/m/Read/ReadVariableOp6Adam/output/kernel_posterior_loc/m/Read/ReadVariableOpFAdam/output/kernel_posterior_untransformed_scale/m/Read/ReadVariableOp4Adam/output/bias_posterior_loc/m/Read/ReadVariableOp;Adam/dense_tfp_1/kernel_posterior_loc/v/Read/ReadVariableOpKAdam/dense_tfp_1/kernel_posterior_untransformed_scale/v/Read/ReadVariableOp9Adam/dense_tfp_1/bias_posterior_loc/v/Read/ReadVariableOp;Adam/dense_tfp_2/kernel_posterior_loc/v/Read/ReadVariableOpKAdam/dense_tfp_2/kernel_posterior_untransformed_scale/v/Read/ReadVariableOp9Adam/dense_tfp_2/bias_posterior_loc/v/Read/ReadVariableOp6Adam/output/kernel_posterior_loc/v/Read/ReadVariableOpFAdam/output/kernel_posterior_untransformed_scale/v/Read/ReadVariableOp4Adam/output/bias_posterior_loc/v/Read/ReadVariableOpConst_6*1
Tin*
(2&	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_save_70056712

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename dense_tfp_1/kernel_posterior_loc0dense_tfp_1/kernel_posterior_untransformed_scaledense_tfp_1/bias_posterior_loc dense_tfp_2/kernel_posterior_loc0dense_tfp_2/kernel_posterior_untransformed_scaledense_tfp_2/bias_posterior_locoutput/kernel_posterior_loc+output/kernel_posterior_untransformed_scaleoutput/bias_posterior_loc	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1'Adam/dense_tfp_1/kernel_posterior_loc/m7Adam/dense_tfp_1/kernel_posterior_untransformed_scale/m%Adam/dense_tfp_1/bias_posterior_loc/m'Adam/dense_tfp_2/kernel_posterior_loc/m7Adam/dense_tfp_2/kernel_posterior_untransformed_scale/m%Adam/dense_tfp_2/bias_posterior_loc/m"Adam/output/kernel_posterior_loc/m2Adam/output/kernel_posterior_untransformed_scale/m Adam/output/bias_posterior_loc/m'Adam/dense_tfp_1/kernel_posterior_loc/v7Adam/dense_tfp_1/kernel_posterior_untransformed_scale/v%Adam/dense_tfp_1/bias_posterior_loc/v'Adam/dense_tfp_2/kernel_posterior_loc/v7Adam/dense_tfp_2/kernel_posterior_untransformed_scale/v%Adam/dense_tfp_2/bias_posterior_loc/v"Adam/output/kernel_posterior_loc/v2Adam/output/kernel_posterior_untransformed_scale/v Adam/output/bias_posterior_loc/v*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference__traced_restore_70056830ϋ!
η
#
#__inference__wrapped_model_70054356	
inputW
Esequential_dense_tfp_1_normal_sample_softplus_readvariableop_resource:g I
7sequential_dense_tfp_1_matmul_1_readvariableop_resource:g 
~sequential_dense_tfp_1_independentdeterministic_constructed_at_dense_tfp_1_sample_deterministic_sample_readvariableop_resource: ι
δsequential_dense_tfp_1_kullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054057μ
ηsequential_dense_tfp_1_kullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xW
Esequential_dense_tfp_2_normal_sample_softplus_readvariableop_resource: I
7sequential_dense_tfp_2_matmul_1_readvariableop_resource: 
~sequential_dense_tfp_2_independentdeterministic_constructed_at_dense_tfp_2_sample_deterministic_sample_readvariableop_resource:ι
δsequential_dense_tfp_2_kullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054192μ
ηsequential_dense_tfp_2_kullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xR
@sequential_output_normal_sample_softplus_readvariableop_resource:	D
2sequential_output_matmul_1_readvariableop_resource:	
tsequential_output_independentdeterministic_constructed_at_output_sample_deterministic_sample_readvariableop_resource:	ί
Ϊsequential_output_kullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054327β
έsequential_output_kullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity’usequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ReadVariableOp’χsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp’κsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp’.sequential/dense_tfp_1/MatMul_1/ReadVariableOp’<sequential/dense_tfp_1/Normal/sample/Softplus/ReadVariableOp’usequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ReadVariableOp’χsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp’κsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp’.sequential/dense_tfp_2/MatMul_1/ReadVariableOp’<sequential/dense_tfp_2/Normal/sample/Softplus/ReadVariableOp’ksequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ReadVariableOp’νsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp’ΰsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp’)sequential/output/MatMul_1/ReadVariableOp’7sequential/output/Normal/sample/Softplus/ReadVariableOp
1sequential/dense_tfp_1/zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"g       l
'sequential/dense_tfp_1/zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ΐ
!sequential/dense_tfp_1/zeros_likeFill:sequential/dense_tfp_1/zeros_like/shape_as_tensor:output:00sequential/dense_tfp_1/zeros_like/Const:output:0*
T0*
_output_shapes

:g t
1sequential/dense_tfp_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB Β
<sequential/dense_tfp_1/Normal/sample/Softplus/ReadVariableOpReadVariableOpEsequential_dense_tfp_1_normal_sample_softplus_readvariableop_resource*
_output_shapes

:g *
dtype0¨
-sequential/dense_tfp_1/Normal/sample/SoftplusSoftplusDsequential/dense_tfp_1/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:g o
*sequential/dense_tfp_1/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4Μ
(sequential/dense_tfp_1/Normal/sample/addAddV23sequential/dense_tfp_1/Normal/sample/add/x:output:0;sequential/dense_tfp_1/Normal/sample/Softplus:activations:0*
T0*
_output_shapes

:g 
4sequential/dense_tfp_1/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"g       l
*sequential/dense_tfp_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
8sequential/dense_tfp_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:sequential/dense_tfp_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:sequential/dense_tfp_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2sequential/dense_tfp_1/Normal/sample/strided_sliceStridedSlice=sequential/dense_tfp_1/Normal/sample/shape_as_tensor:output:0Asequential/dense_tfp_1/Normal/sample/strided_slice/stack:output:0Csequential/dense_tfp_1/Normal/sample/strided_slice/stack_1:output:0Csequential/dense_tfp_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
6sequential/dense_tfp_1/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"g       n
,sequential/dense_tfp_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : 
:sequential/dense_tfp_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
<sequential/dense_tfp_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
<sequential/dense_tfp_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4sequential/dense_tfp_1/Normal/sample/strided_slice_1StridedSlice?sequential/dense_tfp_1/Normal/sample/shape_as_tensor_1:output:0Csequential/dense_tfp_1/Normal/sample/strided_slice_1/stack:output:0Esequential/dense_tfp_1/Normal/sample/strided_slice_1/stack_1:output:0Esequential/dense_tfp_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskx
5sequential/dense_tfp_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB z
7sequential/dense_tfp_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ή
2sequential/dense_tfp_1/Normal/sample/BroadcastArgsBroadcastArgs@sequential/dense_tfp_1/Normal/sample/BroadcastArgs/s0_1:output:0;sequential/dense_tfp_1/Normal/sample/strided_slice:output:0*
_output_shapes
:Ω
4sequential/dense_tfp_1/Normal/sample/BroadcastArgs_1BroadcastArgs7sequential/dense_tfp_1/Normal/sample/BroadcastArgs:r0:0=sequential/dense_tfp_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:~
4sequential/dense_tfp_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:r
0sequential/dense_tfp_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+sequential/dense_tfp_1/Normal/sample/concatConcatV2=sequential/dense_tfp_1/Normal/sample/concat/values_0:output:09sequential/dense_tfp_1/Normal/sample/BroadcastArgs_1:r0:09sequential/dense_tfp_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:
>sequential/dense_tfp_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
@sequential/dense_tfp_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Φ
Nsequential/dense_tfp_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal4sequential/dense_tfp_1/Normal/sample/concat:output:0*
T0*"
_output_shapes
:g *
dtype0
=sequential/dense_tfp_1/Normal/sample/normal/random_normal/mulMulWsequential/dense_tfp_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0Isequential/dense_tfp_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
:g ϋ
9sequential/dense_tfp_1/Normal/sample/normal/random_normalAddV2Asequential/dense_tfp_1/Normal/sample/normal/random_normal/mul:z:0Gsequential/dense_tfp_1/Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
:g Ι
(sequential/dense_tfp_1/Normal/sample/mulMul=sequential/dense_tfp_1/Normal/sample/normal/random_normal:z:0,sequential/dense_tfp_1/Normal/sample/add:z:0*
T0*"
_output_shapes
:g Ί
*sequential/dense_tfp_1/Normal/sample/add_1AddV2,sequential/dense_tfp_1/Normal/sample/mul:z:0*sequential/dense_tfp_1/zeros_like:output:0*
T0*"
_output_shapes
:g 
2sequential/dense_tfp_1/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"g       Ν
,sequential/dense_tfp_1/Normal/sample/ReshapeReshape.sequential/dense_tfp_1/Normal/sample/add_1:z:0;sequential/dense_tfp_1/Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

:g Q
sequential/dense_tfp_1/ShapeShapeinput*
T0*
_output_shapes
:t
*sequential/dense_tfp_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
,sequential/dense_tfp_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????v
,sequential/dense_tfp_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Β
$sequential/dense_tfp_1/strided_sliceStridedSlice%sequential/dense_tfp_1/Shape:output:03sequential/dense_tfp_1/strided_slice/stack:output:05sequential/dense_tfp_1/strided_slice/stack_1:output:05sequential/dense_tfp_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Bsequential/dense_tfp_1/rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:
@sequential/dense_tfp_1/rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????
@sequential/dense_tfp_1/rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????δ
<sequential/dense_tfp_1/rademacher/uniform/sanitize_seed/seedRandomUniformIntKsequential/dense_tfp_1/rademacher/uniform/sanitize_seed/seed/shape:output:0Isequential/dense_tfp_1/rademacher/uniform/sanitize_seed/seed/min:output:0Isequential/dense_tfp_1/rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:
Fsequential/dense_tfp_1/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R 
Fsequential/dense_tfp_1/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 Rυ
_sequential/dense_tfp_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterEsequential/dense_tfp_1/rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::
Fsequential/dense_tfp_1/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :
Bsequential/dense_tfp_1/rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2%sequential/dense_tfp_1/Shape:output:0esequential/dense_tfp_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0isequential/dense_tfp_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Osequential/dense_tfp_1/rademacher/uniform/stateless_random_uniform/alg:output:0Osequential/dense_tfp_1/rademacher/uniform/stateless_random_uniform/min:output:0Osequential/dense_tfp_1/rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????g*
dtype0	i
'sequential/dense_tfp_1/rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 Rέ
%sequential/dense_tfp_1/rademacher/mulMul0sequential/dense_tfp_1/rademacher/mul/x:output:0Ksequential/dense_tfp_1/rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????gi
'sequential/dense_tfp_1/rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R»
%sequential/dense_tfp_1/rademacher/subSub)sequential/dense_tfp_1/rademacher/mul:z:00sequential/dense_tfp_1/rademacher/sub/y:output:0*
T0	*'
_output_shapes
:?????????g
&sequential/dense_tfp_1/rademacher/CastCast)sequential/dense_tfp_1/rademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????gi
'sequential/dense_tfp_1/ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value	B : g
%sequential/dense_tfp_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ά
!sequential/dense_tfp_1/ExpandDims
ExpandDims0sequential/dense_tfp_1/ExpandDims/input:output:0.sequential/dense_tfp_1/ExpandDims/dim:output:0*
T0*
_output_shapes
:d
"sequential/dense_tfp_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ί
sequential/dense_tfp_1/concatConcatV2-sequential/dense_tfp_1/strided_slice:output:0*sequential/dense_tfp_1/ExpandDims:output:0+sequential/dense_tfp_1/concat/axis:output:0*
N*
T0*
_output_shapes
:
Dsequential/dense_tfp_1/rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:
Bsequential/dense_tfp_1/rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????
Bsequential/dense_tfp_1/rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????μ
>sequential/dense_tfp_1/rademacher_1/uniform/sanitize_seed/seedRandomUniformIntMsequential/dense_tfp_1/rademacher_1/uniform/sanitize_seed/seed/shape:output:0Ksequential/dense_tfp_1/rademacher_1/uniform/sanitize_seed/seed/min:output:0Ksequential/dense_tfp_1/rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:
Hsequential/dense_tfp_1/rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R 
Hsequential/dense_tfp_1/rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 Rω
asequential/dense_tfp_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterGsequential/dense_tfp_1/rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::
Hsequential/dense_tfp_1/rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :
Dsequential/dense_tfp_1/rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2&sequential/dense_tfp_1/concat:output:0gsequential/dense_tfp_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0ksequential/dense_tfp_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Qsequential/dense_tfp_1/rademacher_1/uniform/stateless_random_uniform/alg:output:0Qsequential/dense_tfp_1/rademacher_1/uniform/stateless_random_uniform/min:output:0Qsequential/dense_tfp_1/rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:????????? *
dtype0	k
)sequential/dense_tfp_1/rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 Rγ
'sequential/dense_tfp_1/rademacher_1/mulMul2sequential/dense_tfp_1/rademacher_1/mul/x:output:0Msequential/dense_tfp_1/rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:????????? k
)sequential/dense_tfp_1/rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RΑ
'sequential/dense_tfp_1/rademacher_1/subSub+sequential/dense_tfp_1/rademacher_1/mul:z:02sequential/dense_tfp_1/rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:????????? 
(sequential/dense_tfp_1/rademacher_1/CastCast+sequential/dense_tfp_1/rademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:????????? 
sequential/dense_tfp_1/mulMulinput*sequential/dense_tfp_1/rademacher/Cast:y:0*
T0*'
_output_shapes
:?????????g°
sequential/dense_tfp_1/MatMulMatMulsequential/dense_tfp_1/mul:z:05sequential/dense_tfp_1/Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:????????? ¬
sequential/dense_tfp_1/mul_1Mul'sequential/dense_tfp_1/MatMul:product:0,sequential/dense_tfp_1/rademacher_1/Cast:y:0*
T0*'
_output_shapes
:????????? ¦
.sequential/dense_tfp_1/MatMul_1/ReadVariableOpReadVariableOp7sequential_dense_tfp_1_matmul_1_readvariableop_resource*
_output_shapes

:g *
dtype0
sequential/dense_tfp_1/MatMul_1MatMulinput6sequential/dense_tfp_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ’
sequential/dense_tfp_1/addAddV2)sequential/dense_tfp_1/MatMul_1:product:0 sequential/dense_tfp_1/mul_1:z:0*
T0*'
_output_shapes
:????????? ‘
^sequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ΅
ssequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :Ο
sequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:°
usequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ReadVariableOpReadVariableOp~sequential_dense_tfp_1_independentdeterministic_constructed_at_dense_tfp_1_sample_deterministic_sample_readvariableop_resource*
_output_shapes
: *
dtype0ΐ
vsequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: ?
lsequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : Δ
zsequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ζ
|sequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ζ
|sequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ί
tsequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_sliceStridedSlicesequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/shape_as_tensor:output:0sequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_slice/stack:output:0sequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_slice/stack_1:output:0sequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskΊ
wsequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB Ό
ysequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ₯
tsequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/BroadcastArgsBroadcastArgssequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0}sequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:ΐ
vsequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:Ή
vsequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ΄
rsequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ‘
msequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concatConcatV2sequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concat/values_0:output:0ysequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/BroadcastArgs:r0:0sequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concat/values_2:output:0{sequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:‘
rsequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/BroadcastToBroadcastTo}sequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ReadVariableOp:value:0vsequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

: Ε
tsequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       
nsequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ReshapeReshape{sequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/BroadcastTo:output:0}sequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

: ©
_sequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: μ
Ysequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/ReshapeReshapewsequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/Reshape:output:0hsequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Reshape/shape:output:0*
T0*
_output_shapes
: ί
sequential/dense_tfp_1/BiasAddBiasAddsequential/dense_tfp_1/add:z:0bsequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Reshape:output:0*
T0*'
_output_shapes
:????????? ~
sequential/dense_tfp_1/ReluRelu'sequential/dense_tfp_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ώ
χsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOpEsequential_dense_tfp_1_normal_sample_softplus_readvariableop_resource*
_output_shapes

:g *
dtype0 
θsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus?sequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:g «
εsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4
γsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2ξsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0φsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes

:g ϊ
ίsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLogηsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes

:g ρ
αsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Logδsequential_dense_tfp_1_kullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054057*
T0*
_output_shapes
: ή
ίsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSubγsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0εsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes

:g γ
κsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp7sequential_dense_tfp_1_matmul_1_readvariableop_resource*
_output_shapes

:g *
dtype0τ
γsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDivςsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0δsequential_dense_tfp_1_kullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054057*
T0*
_output_shapes

:g λ
εsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDivηsequential_dense_tfp_1_kullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xδsequential_dense_tfp_1_kullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054057*
T0*
_output_shapes

:g 
νsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifferenceηsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0ιsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes

:g §
αsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ρ
ίsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMulκsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0ρsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes

:g ©
γsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @η
αsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mulμsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0γsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

:g ό
αsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1εsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes

:g ©
γsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ι
αsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mulμsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0εsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes

:g ΰ
ίsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2γsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0εsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes

:g ή
αsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Subγsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0γsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

:g ρ
sequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"????ώ???Ι
sequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSumεsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0¨sequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: e
 sequential/dense_tfp_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * 'G
sequential/dense_tfp_1/truedivRealDivsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0)sequential/dense_tfp_1/truediv/y:output:0*
T0*
_output_shapes
: y
(sequential/dense_tfp_1/divergence_kernelIdentity"sequential/dense_tfp_1/truediv:z:0*
T0*
_output_shapes
: v
!sequential/dense_tfp_2/zeros_likeConst*
_output_shapes

: *
dtype0*
valueB *    t
1sequential/dense_tfp_2/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB Β
<sequential/dense_tfp_2/Normal/sample/Softplus/ReadVariableOpReadVariableOpEsequential_dense_tfp_2_normal_sample_softplus_readvariableop_resource*
_output_shapes

: *
dtype0¨
-sequential/dense_tfp_2/Normal/sample/SoftplusSoftplusDsequential/dense_tfp_2/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

: o
*sequential/dense_tfp_2/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4Μ
(sequential/dense_tfp_2/Normal/sample/addAddV23sequential/dense_tfp_2/Normal/sample/add/x:output:0;sequential/dense_tfp_2/Normal/sample/Softplus:activations:0*
T0*
_output_shapes

: 
4sequential/dense_tfp_2/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"       l
*sequential/dense_tfp_2/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
8sequential/dense_tfp_2/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:sequential/dense_tfp_2/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:sequential/dense_tfp_2/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2sequential/dense_tfp_2/Normal/sample/strided_sliceStridedSlice=sequential/dense_tfp_2/Normal/sample/shape_as_tensor:output:0Asequential/dense_tfp_2/Normal/sample/strided_slice/stack:output:0Csequential/dense_tfp_2/Normal/sample/strided_slice/stack_1:output:0Csequential/dense_tfp_2/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
6sequential/dense_tfp_2/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"       n
,sequential/dense_tfp_2/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : 
:sequential/dense_tfp_2/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
<sequential/dense_tfp_2/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
<sequential/dense_tfp_2/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4sequential/dense_tfp_2/Normal/sample/strided_slice_1StridedSlice?sequential/dense_tfp_2/Normal/sample/shape_as_tensor_1:output:0Csequential/dense_tfp_2/Normal/sample/strided_slice_1/stack:output:0Esequential/dense_tfp_2/Normal/sample/strided_slice_1/stack_1:output:0Esequential/dense_tfp_2/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskx
5sequential/dense_tfp_2/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB z
7sequential/dense_tfp_2/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ή
2sequential/dense_tfp_2/Normal/sample/BroadcastArgsBroadcastArgs@sequential/dense_tfp_2/Normal/sample/BroadcastArgs/s0_1:output:0;sequential/dense_tfp_2/Normal/sample/strided_slice:output:0*
_output_shapes
:Ω
4sequential/dense_tfp_2/Normal/sample/BroadcastArgs_1BroadcastArgs7sequential/dense_tfp_2/Normal/sample/BroadcastArgs:r0:0=sequential/dense_tfp_2/Normal/sample/strided_slice_1:output:0*
_output_shapes
:~
4sequential/dense_tfp_2/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:r
0sequential/dense_tfp_2/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+sequential/dense_tfp_2/Normal/sample/concatConcatV2=sequential/dense_tfp_2/Normal/sample/concat/values_0:output:09sequential/dense_tfp_2/Normal/sample/BroadcastArgs_1:r0:09sequential/dense_tfp_2/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:
>sequential/dense_tfp_2/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
@sequential/dense_tfp_2/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Φ
Nsequential/dense_tfp_2/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal4sequential/dense_tfp_2/Normal/sample/concat:output:0*
T0*"
_output_shapes
: *
dtype0
=sequential/dense_tfp_2/Normal/sample/normal/random_normal/mulMulWsequential/dense_tfp_2/Normal/sample/normal/random_normal/RandomStandardNormal:output:0Isequential/dense_tfp_2/Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
: ϋ
9sequential/dense_tfp_2/Normal/sample/normal/random_normalAddV2Asequential/dense_tfp_2/Normal/sample/normal/random_normal/mul:z:0Gsequential/dense_tfp_2/Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
: Ι
(sequential/dense_tfp_2/Normal/sample/mulMul=sequential/dense_tfp_2/Normal/sample/normal/random_normal:z:0,sequential/dense_tfp_2/Normal/sample/add:z:0*
T0*"
_output_shapes
: Ί
*sequential/dense_tfp_2/Normal/sample/add_1AddV2,sequential/dense_tfp_2/Normal/sample/mul:z:0*sequential/dense_tfp_2/zeros_like:output:0*
T0*"
_output_shapes
: 
2sequential/dense_tfp_2/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ν
,sequential/dense_tfp_2/Normal/sample/ReshapeReshape.sequential/dense_tfp_2/Normal/sample/add_1:z:0;sequential/dense_tfp_2/Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

: u
sequential/dense_tfp_2/ShapeShape)sequential/dense_tfp_1/Relu:activations:0*
T0*
_output_shapes
:t
*sequential/dense_tfp_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
,sequential/dense_tfp_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????v
,sequential/dense_tfp_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Β
$sequential/dense_tfp_2/strided_sliceStridedSlice%sequential/dense_tfp_2/Shape:output:03sequential/dense_tfp_2/strided_slice/stack:output:05sequential/dense_tfp_2/strided_slice/stack_1:output:05sequential/dense_tfp_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Bsequential/dense_tfp_2/rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:
@sequential/dense_tfp_2/rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????
@sequential/dense_tfp_2/rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????δ
<sequential/dense_tfp_2/rademacher/uniform/sanitize_seed/seedRandomUniformIntKsequential/dense_tfp_2/rademacher/uniform/sanitize_seed/seed/shape:output:0Isequential/dense_tfp_2/rademacher/uniform/sanitize_seed/seed/min:output:0Isequential/dense_tfp_2/rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:
Fsequential/dense_tfp_2/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R 
Fsequential/dense_tfp_2/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 Rυ
_sequential/dense_tfp_2/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterEsequential/dense_tfp_2/rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::
Fsequential/dense_tfp_2/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :
Bsequential/dense_tfp_2/rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2%sequential/dense_tfp_2/Shape:output:0esequential/dense_tfp_2/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0isequential/dense_tfp_2/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Osequential/dense_tfp_2/rademacher/uniform/stateless_random_uniform/alg:output:0Osequential/dense_tfp_2/rademacher/uniform/stateless_random_uniform/min:output:0Osequential/dense_tfp_2/rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:????????? *
dtype0	i
'sequential/dense_tfp_2/rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 Rέ
%sequential/dense_tfp_2/rademacher/mulMul0sequential/dense_tfp_2/rademacher/mul/x:output:0Ksequential/dense_tfp_2/rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:????????? i
'sequential/dense_tfp_2/rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R»
%sequential/dense_tfp_2/rademacher/subSub)sequential/dense_tfp_2/rademacher/mul:z:00sequential/dense_tfp_2/rademacher/sub/y:output:0*
T0	*'
_output_shapes
:????????? 
&sequential/dense_tfp_2/rademacher/CastCast)sequential/dense_tfp_2/rademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:????????? i
'sequential/dense_tfp_2/ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value	B :g
%sequential/dense_tfp_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ά
!sequential/dense_tfp_2/ExpandDims
ExpandDims0sequential/dense_tfp_2/ExpandDims/input:output:0.sequential/dense_tfp_2/ExpandDims/dim:output:0*
T0*
_output_shapes
:d
"sequential/dense_tfp_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ί
sequential/dense_tfp_2/concatConcatV2-sequential/dense_tfp_2/strided_slice:output:0*sequential/dense_tfp_2/ExpandDims:output:0+sequential/dense_tfp_2/concat/axis:output:0*
N*
T0*
_output_shapes
:
Dsequential/dense_tfp_2/rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:
Bsequential/dense_tfp_2/rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????
Bsequential/dense_tfp_2/rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????μ
>sequential/dense_tfp_2/rademacher_1/uniform/sanitize_seed/seedRandomUniformIntMsequential/dense_tfp_2/rademacher_1/uniform/sanitize_seed/seed/shape:output:0Ksequential/dense_tfp_2/rademacher_1/uniform/sanitize_seed/seed/min:output:0Ksequential/dense_tfp_2/rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:
Hsequential/dense_tfp_2/rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R 
Hsequential/dense_tfp_2/rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 Rω
asequential/dense_tfp_2/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterGsequential/dense_tfp_2/rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::
Hsequential/dense_tfp_2/rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :
Dsequential/dense_tfp_2/rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2&sequential/dense_tfp_2/concat:output:0gsequential/dense_tfp_2/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0ksequential/dense_tfp_2/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Qsequential/dense_tfp_2/rademacher_1/uniform/stateless_random_uniform/alg:output:0Qsequential/dense_tfp_2/rademacher_1/uniform/stateless_random_uniform/min:output:0Qsequential/dense_tfp_2/rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????*
dtype0	k
)sequential/dense_tfp_2/rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 Rγ
'sequential/dense_tfp_2/rademacher_1/mulMul2sequential/dense_tfp_2/rademacher_1/mul/x:output:0Msequential/dense_tfp_2/rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????k
)sequential/dense_tfp_2/rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RΑ
'sequential/dense_tfp_2/rademacher_1/subSub+sequential/dense_tfp_2/rademacher_1/mul:z:02sequential/dense_tfp_2/rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:?????????
(sequential/dense_tfp_2/rademacher_1/CastCast+sequential/dense_tfp_2/rademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????ͺ
sequential/dense_tfp_2/mulMul)sequential/dense_tfp_1/Relu:activations:0*sequential/dense_tfp_2/rademacher/Cast:y:0*
T0*'
_output_shapes
:????????? °
sequential/dense_tfp_2/MatMulMatMulsequential/dense_tfp_2/mul:z:05sequential/dense_tfp_2/Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????¬
sequential/dense_tfp_2/mul_1Mul'sequential/dense_tfp_2/MatMul:product:0,sequential/dense_tfp_2/rademacher_1/Cast:y:0*
T0*'
_output_shapes
:?????????¦
.sequential/dense_tfp_2/MatMul_1/ReadVariableOpReadVariableOp7sequential_dense_tfp_2_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0Ύ
sequential/dense_tfp_2/MatMul_1MatMul)sequential/dense_tfp_1/Relu:activations:06sequential/dense_tfp_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????’
sequential/dense_tfp_2/addAddV2)sequential/dense_tfp_2/MatMul_1:product:0 sequential/dense_tfp_2/mul_1:z:0*
T0*'
_output_shapes
:?????????‘
^sequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ΅
ssequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :Ο
sequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:°
usequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ReadVariableOpReadVariableOp~sequential_dense_tfp_2_independentdeterministic_constructed_at_dense_tfp_2_sample_deterministic_sample_readvariableop_resource*
_output_shapes
:*
dtype0ΐ
vsequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:?
lsequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : Δ
zsequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ζ
|sequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ζ
|sequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ί
tsequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_sliceStridedSlicesequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/shape_as_tensor:output:0sequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_slice/stack:output:0sequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_slice/stack_1:output:0sequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskΊ
wsequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB Ό
ysequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ₯
tsequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/BroadcastArgsBroadcastArgssequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0}sequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:ΐ
vsequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:Ή
vsequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ΄
rsequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ‘
msequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concatConcatV2sequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concat/values_0:output:0ysequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/BroadcastArgs:r0:0sequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concat/values_2:output:0{sequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:‘
rsequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/BroadcastToBroadcastTo}sequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ReadVariableOp:value:0vsequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

:Ε
tsequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
nsequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ReshapeReshape{sequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/BroadcastTo:output:0}sequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

:©
_sequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:μ
Ysequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/ReshapeReshapewsequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/Reshape:output:0hsequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Reshape/shape:output:0*
T0*
_output_shapes
:ί
sequential/dense_tfp_2/BiasAddBiasAddsequential/dense_tfp_2/add:z:0bsequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????~
sequential/dense_tfp_2/ReluRelu'sequential/dense_tfp_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????ώ
χsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOpEsequential_dense_tfp_2_normal_sample_softplus_readvariableop_resource*
_output_shapes

: *
dtype0 
θsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus?sequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

: «
εsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4
γsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2ξsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0φsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes

: ϊ
ίsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLogηsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes

: ρ
αsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Logδsequential_dense_tfp_2_kullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054192*
T0*
_output_shapes
: ή
ίsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSubγsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0εsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes

: γ
κsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp7sequential_dense_tfp_2_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0τ
γsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDivςsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0δsequential_dense_tfp_2_kullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054192*
T0*
_output_shapes

: λ
εsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDivηsequential_dense_tfp_2_kullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xδsequential_dense_tfp_2_kullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054192*
T0*
_output_shapes

: 
νsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifferenceηsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0ιsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes

: §
αsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ρ
ίsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMulκsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0ρsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes

: ©
γsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @η
αsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mulμsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0γsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

: ό
αsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1εsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes

: ©
γsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ι
αsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mulμsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0εsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes

: ΰ
ίsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2γsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0εsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes

: ή
αsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Subγsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0γsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

: ρ
sequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"????ώ???Ι
sequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSumεsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0¨sequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: e
 sequential/dense_tfp_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * 'G
sequential/dense_tfp_2/truedivRealDivsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0)sequential/dense_tfp_2/truediv/y:output:0*
T0*
_output_shapes
: y
(sequential/dense_tfp_2/divergence_kernelIdentity"sequential/dense_tfp_2/truediv:z:0*
T0*
_output_shapes
: q
sequential/output/zeros_likeConst*
_output_shapes

:	*
dtype0*
valueB	*    o
,sequential/output/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB Έ
7sequential/output/Normal/sample/Softplus/ReadVariableOpReadVariableOp@sequential_output_normal_sample_softplus_readvariableop_resource*
_output_shapes

:	*
dtype0
(sequential/output/Normal/sample/SoftplusSoftplus?sequential/output/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:	j
%sequential/output/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4½
#sequential/output/Normal/sample/addAddV2.sequential/output/Normal/sample/add/x:output:06sequential/output/Normal/sample/Softplus:activations:0*
T0*
_output_shapes

:	
/sequential/output/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"   	   g
%sequential/output/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : }
3sequential/output/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5sequential/output/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential/output/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ω
-sequential/output/Normal/sample/strided_sliceStridedSlice8sequential/output/Normal/sample/shape_as_tensor:output:0<sequential/output/Normal/sample/strided_slice/stack:output:0>sequential/output/Normal/sample/strided_slice/stack_1:output:0>sequential/output/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
1sequential/output/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"   	   i
'sequential/output/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : 
5sequential/output/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7sequential/output/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7sequential/output/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
/sequential/output/Normal/sample/strided_slice_1StridedSlice:sequential/output/Normal/sample/shape_as_tensor_1:output:0>sequential/output/Normal/sample/strided_slice_1/stack:output:0@sequential/output/Normal/sample/strided_slice_1/stack_1:output:0@sequential/output/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masks
0sequential/output/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB u
2sequential/output/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB Ο
-sequential/output/Normal/sample/BroadcastArgsBroadcastArgs;sequential/output/Normal/sample/BroadcastArgs/s0_1:output:06sequential/output/Normal/sample/strided_slice:output:0*
_output_shapes
:Κ
/sequential/output/Normal/sample/BroadcastArgs_1BroadcastArgs2sequential/output/Normal/sample/BroadcastArgs:r0:08sequential/output/Normal/sample/strided_slice_1:output:0*
_output_shapes
:y
/sequential/output/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:m
+sequential/output/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
&sequential/output/Normal/sample/concatConcatV28sequential/output/Normal/sample/concat/values_0:output:04sequential/output/Normal/sample/BroadcastArgs_1:r0:04sequential/output/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:~
9sequential/output/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
;sequential/output/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Μ
Isequential/output/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal/sequential/output/Normal/sample/concat:output:0*
T0*"
_output_shapes
:	*
dtype0
8sequential/output/Normal/sample/normal/random_normal/mulMulRsequential/output/Normal/sample/normal/random_normal/RandomStandardNormal:output:0Dsequential/output/Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
:	μ
4sequential/output/Normal/sample/normal/random_normalAddV2<sequential/output/Normal/sample/normal/random_normal/mul:z:0Bsequential/output/Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
:	Ί
#sequential/output/Normal/sample/mulMul8sequential/output/Normal/sample/normal/random_normal:z:0'sequential/output/Normal/sample/add:z:0*
T0*"
_output_shapes
:	«
%sequential/output/Normal/sample/add_1AddV2'sequential/output/Normal/sample/mul:z:0%sequential/output/zeros_like:output:0*
T0*"
_output_shapes
:	~
-sequential/output/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   	   Ύ
'sequential/output/Normal/sample/ReshapeReshape)sequential/output/Normal/sample/add_1:z:06sequential/output/Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

:	p
sequential/output/ShapeShape)sequential/dense_tfp_2/Relu:activations:0*
T0*
_output_shapes
:o
%sequential/output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
'sequential/output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????q
'sequential/output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:©
sequential/output/strided_sliceStridedSlice sequential/output/Shape:output:0.sequential/output/strided_slice/stack:output:00sequential/output/strided_slice/stack_1:output:00sequential/output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
=sequential/output/rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:
;sequential/output/rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????
;sequential/output/rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????Π
7sequential/output/rademacher/uniform/sanitize_seed/seedRandomUniformIntFsequential/output/rademacher/uniform/sanitize_seed/seed/shape:output:0Dsequential/output/rademacher/uniform/sanitize_seed/seed/min:output:0Dsequential/output/rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:
Asequential/output/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R 
Asequential/output/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 Rλ
Zsequential/output/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter@sequential/output/rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::
Asequential/output/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :ΰ
=sequential/output/rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2 sequential/output/Shape:output:0`sequential/output/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0dsequential/output/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Jsequential/output/rademacher/uniform/stateless_random_uniform/alg:output:0Jsequential/output/rademacher/uniform/stateless_random_uniform/min:output:0Jsequential/output/rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????*
dtype0	d
"sequential/output/rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 RΞ
 sequential/output/rademacher/mulMul+sequential/output/rademacher/mul/x:output:0Fsequential/output/rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????d
"sequential/output/rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R¬
 sequential/output/rademacher/subSub$sequential/output/rademacher/mul:z:0+sequential/output/rademacher/sub/y:output:0*
T0	*'
_output_shapes
:?????????
!sequential/output/rademacher/CastCast$sequential/output/rademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????d
"sequential/output/ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value	B :	b
 sequential/output/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : §
sequential/output/ExpandDims
ExpandDims+sequential/output/ExpandDims/input:output:0)sequential/output/ExpandDims/dim:output:0*
T0*
_output_shapes
:_
sequential/output/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Λ
sequential/output/concatConcatV2(sequential/output/strided_slice:output:0%sequential/output/ExpandDims:output:0&sequential/output/concat/axis:output:0*
N*
T0*
_output_shapes
:
?sequential/output/rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:
=sequential/output/rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????
=sequential/output/rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????Ψ
9sequential/output/rademacher_1/uniform/sanitize_seed/seedRandomUniformIntHsequential/output/rademacher_1/uniform/sanitize_seed/seed/shape:output:0Fsequential/output/rademacher_1/uniform/sanitize_seed/seed/min:output:0Fsequential/output/rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:
Csequential/output/rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R 
Csequential/output/rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 Rο
\sequential/output/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterBsequential/output/rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::
Csequential/output/rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :ν
?sequential/output/rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2!sequential/output/concat:output:0bsequential/output/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0fsequential/output/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Lsequential/output/rademacher_1/uniform/stateless_random_uniform/alg:output:0Lsequential/output/rademacher_1/uniform/stateless_random_uniform/min:output:0Lsequential/output/rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????	*
dtype0	f
$sequential/output/rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 RΤ
"sequential/output/rademacher_1/mulMul-sequential/output/rademacher_1/mul/x:output:0Hsequential/output/rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????	f
$sequential/output/rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R²
"sequential/output/rademacher_1/subSub&sequential/output/rademacher_1/mul:z:0-sequential/output/rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:?????????	
#sequential/output/rademacher_1/CastCast&sequential/output/rademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????	 
sequential/output/mulMul)sequential/dense_tfp_2/Relu:activations:0%sequential/output/rademacher/Cast:y:0*
T0*'
_output_shapes
:?????????‘
sequential/output/MatMulMatMulsequential/output/mul:z:00sequential/output/Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????	
sequential/output/mul_1Mul"sequential/output/MatMul:product:0'sequential/output/rademacher_1/Cast:y:0*
T0*'
_output_shapes
:?????????	
)sequential/output/MatMul_1/ReadVariableOpReadVariableOp2sequential_output_matmul_1_readvariableop_resource*
_output_shapes

:	*
dtype0΄
sequential/output/MatMul_1MatMul)sequential/dense_tfp_2/Relu:activations:01sequential/output/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	
sequential/output/addAddV2$sequential/output/MatMul_1:product:0sequential/output/mul_1:z:0*
T0*'
_output_shapes
:?????????	
Tsequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB «
isequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :Δ
zsequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:
ksequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ReadVariableOpReadVariableOptsequential_output_independentdeterministic_constructed_at_output_sample_deterministic_sample_readvariableop_resource*
_output_shapes
:	*
dtype0Ά
lsequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:	€
bsequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : Ί
psequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ό
rsequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ό
rsequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ͺ
jsequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_sliceStridedSliceusequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/shape_as_tensor:output:0ysequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_slice/stack:output:0{sequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_slice/stack_1:output:0{sequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask°
msequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ²
osequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 
jsequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/BroadcastArgsBroadcastArgsxsequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0ssequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:Ά
lsequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:―
lsequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ͺ
hsequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ο
csequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concatConcatV2usequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concat/values_0:output:0osequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/BroadcastArgs:r0:0usequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concat/values_2:output:0qsequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:
hsequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/BroadcastToBroadcastTossequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ReadVariableOp:value:0lsequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

:	»
jsequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   	   
dsequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ReshapeReshapeqsequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/BroadcastTo:output:0ssequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

:	
Usequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:	Ξ
Osequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/ReshapeReshapemsequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/Reshape:output:0^sequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Reshape/shape:output:0*
T0*
_output_shapes
:	Λ
sequential/output/BiasAddBiasAddsequential/output/add:z:0Xsequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????	z
sequential/output/SoftmaxSoftmax"sequential/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	ο
νsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp@sequential_output_normal_sample_softplus_readvariableop_resource*
_output_shapes

:	*
dtype0
ήsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplusυsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:	‘
Ϋsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4β
Ωsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2δsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0μsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes

:	ζ
Υsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLogέsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes

:	έ
Χsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1LogΪsequential_output_kullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054327*
T0*
_output_shapes
: ΐ
Υsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSubΩsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0Ϋsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes

:	Τ
ΰsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp2sequential_output_matmul_1_readvariableop_resource*
_output_shapes

:	*
dtype0Φ
Ωsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDivθsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0Ϊsequential_output_kullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054327*
T0*
_output_shapes

:	Ν
Ϋsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDivέsequential_output_kullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xΪsequential_output_kullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054327*
T0*
_output_shapes

:	δ
γsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifferenceέsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0ίsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes

:	
Χsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Σ
Υsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMulΰsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0ηsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes

:	
Ωsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ι
Χsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mulβsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0Ωsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

:	θ
Χsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1Ϋsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes

:	
Ωsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Λ
Χsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mulβsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0Ϋsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes

:	Β
Υsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2Ωsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0Ϋsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes

:	ΐ
Χsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1SubΩsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0Ωsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

:	η
sequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"????ώ???«
sequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSumΫsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0sequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: `
sequential/output/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * 'Gϊ
sequential/output/truedivRealDivsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0$sequential/output/truediv/y:output:0*
T0*
_output_shapes
: o
#sequential/output/divergence_kernelIdentitysequential/output/truediv:z:0*
T0*
_output_shapes
: r
IdentityIdentity#sequential/output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????	
NoOpNoOpv^sequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ReadVariableOpψ^sequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpλ^sequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp/^sequential/dense_tfp_1/MatMul_1/ReadVariableOp=^sequential/dense_tfp_1/Normal/sample/Softplus/ReadVariableOpv^sequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ReadVariableOpψ^sequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpλ^sequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp/^sequential/dense_tfp_2/MatMul_1/ReadVariableOp=^sequential/dense_tfp_2/Normal/sample/Softplus/ReadVariableOpl^sequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ReadVariableOpξ^sequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpα^sequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp*^sequential/output/MatMul_1/ReadVariableOp8^sequential/output/Normal/sample/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????g: : : : :g : : : : : : : : : :	2ξ
usequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ReadVariableOpusequential/dense_tfp_1/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ReadVariableOp2τ
χsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpχsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2Ϊ
κsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpκsequential/dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2`
.sequential/dense_tfp_1/MatMul_1/ReadVariableOp.sequential/dense_tfp_1/MatMul_1/ReadVariableOp2|
<sequential/dense_tfp_1/Normal/sample/Softplus/ReadVariableOp<sequential/dense_tfp_1/Normal/sample/Softplus/ReadVariableOp2ξ
usequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ReadVariableOpusequential/dense_tfp_2/IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ReadVariableOp2τ
χsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpχsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2Ϊ
κsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpκsequential/dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2`
.sequential/dense_tfp_2/MatMul_1/ReadVariableOp.sequential/dense_tfp_2/MatMul_1/ReadVariableOp2|
<sequential/dense_tfp_2/Normal/sample/Softplus/ReadVariableOp<sequential/dense_tfp_2/Normal/sample/Softplus/ReadVariableOp2Ϊ
ksequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ReadVariableOpksequential/output/IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ReadVariableOp2ΰ
νsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpνsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2Ζ
ΰsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpΰsequential/output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2V
)sequential/output/MatMul_1/ReadVariableOp)sequential/output/MatMul_1/ReadVariableOp2r
7sequential/output/Normal/sample/Softplus/ReadVariableOp7sequential/output/Normal/sample/Softplus/ReadVariableOp:N J
'
_output_shapes
:?????????g

_user_specified_nameinput:

_output_shapes
: :$ 

_output_shapes

:g :	

_output_shapes
: :$
 

_output_shapes

: :

_output_shapes
: :$ 

_output_shapes

:	
ύ

$__inference__traced_restore_70056830
file_prefixC
1assignvariableop_dense_tfp_1_kernel_posterior_loc:g U
Cassignvariableop_1_dense_tfp_1_kernel_posterior_untransformed_scale:g ?
1assignvariableop_2_dense_tfp_1_bias_posterior_loc: E
3assignvariableop_3_dense_tfp_2_kernel_posterior_loc: U
Cassignvariableop_4_dense_tfp_2_kernel_posterior_untransformed_scale: ?
1assignvariableop_5_dense_tfp_2_bias_posterior_loc:@
.assignvariableop_6_output_kernel_posterior_loc:	P
>assignvariableop_7_output_kernel_posterior_untransformed_scale:	:
,assignvariableop_8_output_bias_posterior_loc:	&
assignvariableop_9_adam_iter:	 )
assignvariableop_10_adam_beta_1: )
assignvariableop_11_adam_beta_2: (
assignvariableop_12_adam_decay: 0
&assignvariableop_13_adam_learning_rate: #
assignvariableop_14_total: #
assignvariableop_15_count: %
assignvariableop_16_total_1: %
assignvariableop_17_count_1: M
;assignvariableop_18_adam_dense_tfp_1_kernel_posterior_loc_m:g ]
Kassignvariableop_19_adam_dense_tfp_1_kernel_posterior_untransformed_scale_m:g G
9assignvariableop_20_adam_dense_tfp_1_bias_posterior_loc_m: M
;assignvariableop_21_adam_dense_tfp_2_kernel_posterior_loc_m: ]
Kassignvariableop_22_adam_dense_tfp_2_kernel_posterior_untransformed_scale_m: G
9assignvariableop_23_adam_dense_tfp_2_bias_posterior_loc_m:H
6assignvariableop_24_adam_output_kernel_posterior_loc_m:	X
Fassignvariableop_25_adam_output_kernel_posterior_untransformed_scale_m:	B
4assignvariableop_26_adam_output_bias_posterior_loc_m:	M
;assignvariableop_27_adam_dense_tfp_1_kernel_posterior_loc_v:g ]
Kassignvariableop_28_adam_dense_tfp_1_kernel_posterior_untransformed_scale_v:g G
9assignvariableop_29_adam_dense_tfp_1_bias_posterior_loc_v: M
;assignvariableop_30_adam_dense_tfp_2_kernel_posterior_loc_v: ]
Kassignvariableop_31_adam_dense_tfp_2_kernel_posterior_untransformed_scale_v: G
9assignvariableop_32_adam_dense_tfp_2_bias_posterior_loc_v:H
6assignvariableop_33_adam_output_kernel_posterior_loc_v:	X
Fassignvariableop_34_adam_output_kernel_posterior_untransformed_scale_v:	B
4assignvariableop_35_adam_output_bias_posterior_loc_v:	
identity_37’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_31’AssignVariableOp_32’AssignVariableOp_33’AssignVariableOp_34’AssignVariableOp_35’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9’
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*Θ
valueΎB»%BDlayer_with_weights-0/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-1/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-1/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-2/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-2/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB`layer_with_weights-0/kernel_posterior_loc/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-0/kernel_posterior_untransformed_scale/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/bias_posterior_loc/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB`layer_with_weights-1/kernel_posterior_loc/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-1/kernel_posterior_untransformed_scale/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-1/bias_posterior_loc/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB`layer_with_weights-2/kernel_posterior_loc/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-2/kernel_posterior_untransformed_scale/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-2/bias_posterior_loc/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB`layer_with_weights-0/kernel_posterior_loc/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-0/kernel_posterior_untransformed_scale/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/bias_posterior_loc/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB`layer_with_weights-1/kernel_posterior_loc/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-1/kernel_posterior_untransformed_scale/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-1/bias_posterior_loc/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB`layer_with_weights-2/kernel_posterior_loc/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-2/kernel_posterior_untransformed_scale/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-2/bias_posterior_loc/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHΊ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ϊ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ͺ
_output_shapes
:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp1assignvariableop_dense_tfp_1_kernel_posterior_locIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_1AssignVariableOpCassignvariableop_1_dense_tfp_1_kernel_posterior_untransformed_scaleIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_2AssignVariableOp1assignvariableop_2_dense_tfp_1_bias_posterior_locIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:’
AssignVariableOp_3AssignVariableOp3assignvariableop_3_dense_tfp_2_kernel_posterior_locIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_4AssignVariableOpCassignvariableop_4_dense_tfp_2_kernel_posterior_untransformed_scaleIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_5AssignVariableOp1assignvariableop_5_dense_tfp_2_bias_posterior_locIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp.assignvariableop_6_output_kernel_posterior_locIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_7AssignVariableOp>assignvariableop_7_output_kernel_posterior_untransformed_scaleIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp,assignvariableop_8_output_bias_posterior_locIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_iterIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_decayIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp&assignvariableop_13_adam_learning_rateIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_18AssignVariableOp;assignvariableop_18_adam_dense_tfp_1_kernel_posterior_loc_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ό
AssignVariableOp_19AssignVariableOpKassignvariableop_19_adam_dense_tfp_1_kernel_posterior_untransformed_scale_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:ͺ
AssignVariableOp_20AssignVariableOp9assignvariableop_20_adam_dense_tfp_1_bias_posterior_loc_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_21AssignVariableOp;assignvariableop_21_adam_dense_tfp_2_kernel_posterior_loc_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ό
AssignVariableOp_22AssignVariableOpKassignvariableop_22_adam_dense_tfp_2_kernel_posterior_untransformed_scale_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:ͺ
AssignVariableOp_23AssignVariableOp9assignvariableop_23_adam_dense_tfp_2_bias_posterior_loc_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_24AssignVariableOp6assignvariableop_24_adam_output_kernel_posterior_loc_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_25AssignVariableOpFassignvariableop_25_adam_output_kernel_posterior_untransformed_scale_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:₯
AssignVariableOp_26AssignVariableOp4assignvariableop_26_adam_output_bias_posterior_loc_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_27AssignVariableOp;assignvariableop_27_adam_dense_tfp_1_kernel_posterior_loc_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ό
AssignVariableOp_28AssignVariableOpKassignvariableop_28_adam_dense_tfp_1_kernel_posterior_untransformed_scale_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:ͺ
AssignVariableOp_29AssignVariableOp9assignvariableop_29_adam_dense_tfp_1_bias_posterior_loc_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_30AssignVariableOp;assignvariableop_30_adam_dense_tfp_2_kernel_posterior_loc_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ό
AssignVariableOp_31AssignVariableOpKassignvariableop_31_adam_dense_tfp_2_kernel_posterior_untransformed_scale_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:ͺ
AssignVariableOp_32AssignVariableOp9assignvariableop_32_adam_dense_tfp_2_bias_posterior_loc_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_33AssignVariableOp6assignvariableop_33_adam_output_kernel_posterior_loc_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_34AssignVariableOpFassignvariableop_34_adam_output_kernel_posterior_untransformed_scale_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:₯
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_output_bias_posterior_loc_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 η
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_37IdentityIdentity_36:output:0^NoOp_1*
T0*
_output_shapes
: Τ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_37Identity_37:output:0*]
_input_shapesL
J: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Οψ
π

I__inference_dense_tfp_1_layer_call_and_return_conditional_losses_70056263

inputs@
.normal_sample_softplus_readvariableop_resource:g 2
 matmul_1_readvariableop_resource:g u
gindependentdeterministic_constructed_at_dense_tfp_1_sample_deterministic_sample_readvariableop_resource: ?
Νkullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70056233Υ
Πkullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1’^IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ReadVariableOp’ΰKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp’ΣKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp’MatMul_1/ReadVariableOp’%Normal/sample/Softplus/ReadVariableOpk
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"g       U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes

:g ]
Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 
%Normal/sample/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*
_output_shapes

:g *
dtype0z
Normal/sample/SoftplusSoftplus-Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:g X
Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4
Normal/sample/addAddV2Normal/sample/add/x:output:0$Normal/sample/Softplus:activations:0*
T0*
_output_shapes

:g n
Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"g       U
Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Normal/sample/strided_sliceStridedSlice&Normal/sample/shape_as_tensor:output:0*Normal/sample/strided_slice/stack:output:0,Normal/sample/strided_slice/stack_1:output:0,Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"g       W
Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : m
#Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:©
Normal/sample/strided_slice_1StridedSlice(Normal/sample/shape_as_tensor_1:output:0,Normal/sample/strided_slice_1/stack:output:0.Normal/sample/strided_slice_1/stack_1:output:0.Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maska
Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB c
 Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 
Normal/sample/BroadcastArgsBroadcastArgs)Normal/sample/BroadcastArgs/s0_1:output:0$Normal/sample/strided_slice:output:0*
_output_shapes
:
Normal/sample/BroadcastArgs_1BroadcastArgs Normal/sample/BroadcastArgs:r0:0&Normal/sample/strided_slice_1:output:0*
_output_shapes
:g
Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:[
Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ύ
Normal/sample/concatConcatV2&Normal/sample/concat/values_0:output:0"Normal/sample/BroadcastArgs_1:r0:0"Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:l
'Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¨
7Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat:output:0*
T0*"
_output_shapes
:g *
dtype0Π
&Normal/sample/normal/random_normal/mulMul@Normal/sample/normal/random_normal/RandomStandardNormal:output:02Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
:g Ά
"Normal/sample/normal/random_normalAddV2*Normal/sample/normal/random_normal/mul:z:00Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
:g 
Normal/sample/mulMul&Normal/sample/normal/random_normal:z:0Normal/sample/add:z:0*
T0*"
_output_shapes
:g u
Normal/sample/add_1AddV2Normal/sample/mul:z:0zeros_like:output:0*
T0*"
_output_shapes
:g l
Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"g       
Normal/sample/ReshapeReshapeNormal/sample/add_1:z:0$Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

:g ;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ο
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masku
+rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:t
)rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????o
)rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????
%rademacher/uniform/sanitize_seed/seedRandomUniformInt4rademacher/uniform/sanitize_seed/seed/shape:output:02rademacher/uniform/sanitize_seed/seed/min:output:02rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:q
/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R q
/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 RΗ
Hrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter.rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::q
/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :β
+rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2Shape:output:0Nrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Rrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:08rademacher/uniform/stateless_random_uniform/alg:output:08rademacher/uniform/stateless_random_uniform/min:output:08rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????g*
dtype0	R
rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R
rademacher/mulMulrademacher/mul/x:output:04rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????gR
rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rv
rademacher/subSubrademacher/mul:z:0rademacher/sub/y:output:0*
T0	*'
_output_shapes
:?????????gl
rademacher/CastCastrademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????gR
ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value	B : P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsExpandDims/input:output:0ExpandDims/dim:output:0*
T0*
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2strided_slice:output:0ExpandDims:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:w
-rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:v
+rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????q
+rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????
'rademacher_1/uniform/sanitize_seed/seedRandomUniformInt6rademacher_1/uniform/sanitize_seed/seed/shape:output:04rademacher_1/uniform/sanitize_seed/seed/min:output:04rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:s
1rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R s
1rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 RΛ
Jrademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter0rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::s
1rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :ο
-rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2concat:output:0Prademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Trademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0:rademacher_1/uniform/stateless_random_uniform/alg:output:0:rademacher_1/uniform/stateless_random_uniform/min:output:0:rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:????????? *
dtype0	T
rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R
rademacher_1/mulMulrademacher_1/mul/x:output:06rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:????????? T
rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R|
rademacher_1/subSubrademacher_1/mul:z:0rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:????????? p
rademacher_1/CastCastrademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:????????? Y
mulMulinputsrademacher/Cast:y:0*
T0*'
_output_shapes
:?????????gk
MatMulMatMulmul:z:0Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:????????? g
mul_1MulMatMul:product:0rademacher_1/Cast:y:0*
T0*'
_output_shapes
:????????? x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:g *
dtype0m
MatMul_1MatMulinputsMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ]
addAddV2MatMul_1:product:0	mul_1:z:0*
T0*'
_output_shapes
:????????? 
GIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 
\IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :·
mIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:
^IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ReadVariableOpReadVariableOpgindependentdeterministic_constructed_at_dense_tfp_1_sample_deterministic_sample_readvariableop_resource*
_output_shapes
: *
dtype0©
_IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: 
UIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ­
cIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ―
eIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:―
eIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ι
]IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_sliceStridedSlicehIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/shape_as_tensor:output:0lIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_slice/stack:output:0nIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_slice/stack_1:output:0nIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask£
`IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ₯
bIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ί
]IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/BroadcastArgsBroadcastArgskIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0fIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:©
_IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:’
_IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 
[IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
VIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concatConcatV2hIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concat/values_0:output:0bIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/BroadcastArgs:r0:0hIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concat/values_2:output:0dIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:ά
[IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/BroadcastToBroadcastTofIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ReadVariableOp:value:0_IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

: ?
]IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ω
WIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ReshapeReshapedIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/BroadcastTo:output:0fIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

: 
HIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: §
BIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/ReshapeReshape`IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/Reshape:output:0QIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Reshape/shape:output:0*
T0*
_output_shapes
: 
BiasAddBiasAddadd:z:0KIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Reshape:output:0*
T0*'
_output_shapes
:????????? P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? Π
ΰKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*
_output_shapes

:g *
dtype0ς
ΡKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplusθKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:g 
ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4»
ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2ΧKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0ίKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes

:g Μ
ΘKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLogΠKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes

:g Γ
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1LogΝkullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70056233*
T0*
_output_shapes
: 
ΘKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSubΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes

:g ΅
ΣKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:g *
dtype0―
ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDivΫKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0Νkullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70056233*
T0*
_output_shapes

:g ¦
ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDivΠkullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xΝkullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70056233*
T0*
_output_shapes

:g ½
ΦKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifferenceΠKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes

:g 
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¬
ΘKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMulΣKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0ΪKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes

:g 
ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @’
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1MulΥKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

:g Ξ
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes

:g 
ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?€
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2MulΥKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes

:g 
ΘKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes

:g 
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1SubΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

:g Ϊ
KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"????ώ???
vKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSumΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * 'GΘ
truedivRealDivKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0truediv/y:output:0*
T0*
_output_shapes
: K
divergence_kernelIdentitytruediv:z:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? Z

Identity_1Identitydivergence_kernel:output:0^NoOp*
T0*
_output_shapes
: €
NoOpNoOp_^IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ReadVariableOpα^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpΤ^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp^MatMul_1/ReadVariableOp&^Normal/sample/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????g: : : : :g 2ΐ
^IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ReadVariableOp^IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ReadVariableOp2Ζ
ΰKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpΰKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2¬
ΣKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpΣKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2N
%Normal/sample/Softplus/ReadVariableOp%Normal/sample/Softplus/ReadVariableOp:O K
'
_output_shapes
:?????????g
 
_user_specified_nameinputs:

_output_shapes
: :$ 

_output_shapes

:g 
	
Φ
.__inference_dense_tfp_2_layer_call_fn_70056279

inputs
unknown: 
	unknown_0: 
	unknown_1:
	unknown_2
	unknown_3
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????: *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dense_tfp_2_layer_call_and_return_conditional_losses_70054658o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:????????? : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :$ 

_output_shapes

: 
ΑX

!__inference__traced_save_70056712
file_prefix?
;savev2_dense_tfp_1_kernel_posterior_loc_read_readvariableopO
Ksavev2_dense_tfp_1_kernel_posterior_untransformed_scale_read_readvariableop=
9savev2_dense_tfp_1_bias_posterior_loc_read_readvariableop?
;savev2_dense_tfp_2_kernel_posterior_loc_read_readvariableopO
Ksavev2_dense_tfp_2_kernel_posterior_untransformed_scale_read_readvariableop=
9savev2_dense_tfp_2_bias_posterior_loc_read_readvariableop:
6savev2_output_kernel_posterior_loc_read_readvariableopJ
Fsavev2_output_kernel_posterior_untransformed_scale_read_readvariableop8
4savev2_output_bias_posterior_loc_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopF
Bsavev2_adam_dense_tfp_1_kernel_posterior_loc_m_read_readvariableopV
Rsavev2_adam_dense_tfp_1_kernel_posterior_untransformed_scale_m_read_readvariableopD
@savev2_adam_dense_tfp_1_bias_posterior_loc_m_read_readvariableopF
Bsavev2_adam_dense_tfp_2_kernel_posterior_loc_m_read_readvariableopV
Rsavev2_adam_dense_tfp_2_kernel_posterior_untransformed_scale_m_read_readvariableopD
@savev2_adam_dense_tfp_2_bias_posterior_loc_m_read_readvariableopA
=savev2_adam_output_kernel_posterior_loc_m_read_readvariableopQ
Msavev2_adam_output_kernel_posterior_untransformed_scale_m_read_readvariableop?
;savev2_adam_output_bias_posterior_loc_m_read_readvariableopF
Bsavev2_adam_dense_tfp_1_kernel_posterior_loc_v_read_readvariableopV
Rsavev2_adam_dense_tfp_1_kernel_posterior_untransformed_scale_v_read_readvariableopD
@savev2_adam_dense_tfp_1_bias_posterior_loc_v_read_readvariableopF
Bsavev2_adam_dense_tfp_2_kernel_posterior_loc_v_read_readvariableopV
Rsavev2_adam_dense_tfp_2_kernel_posterior_untransformed_scale_v_read_readvariableopD
@savev2_adam_dense_tfp_2_bias_posterior_loc_v_read_readvariableopA
=savev2_adam_output_kernel_posterior_loc_v_read_readvariableopQ
Msavev2_adam_output_kernel_posterior_untransformed_scale_v_read_readvariableop?
;savev2_adam_output_bias_posterior_loc_v_read_readvariableop
savev2_const_6

identity_1’MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*Θ
valueΎB»%BDlayer_with_weights-0/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-1/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-1/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-2/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-2/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB`layer_with_weights-0/kernel_posterior_loc/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-0/kernel_posterior_untransformed_scale/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/bias_posterior_loc/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB`layer_with_weights-1/kernel_posterior_loc/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-1/kernel_posterior_untransformed_scale/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-1/bias_posterior_loc/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB`layer_with_weights-2/kernel_posterior_loc/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-2/kernel_posterior_untransformed_scale/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-2/bias_posterior_loc/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB`layer_with_weights-0/kernel_posterior_loc/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-0/kernel_posterior_untransformed_scale/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/bias_posterior_loc/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB`layer_with_weights-1/kernel_posterior_loc/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-1/kernel_posterior_untransformed_scale/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-1/bias_posterior_loc/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB`layer_with_weights-2/kernel_posterior_loc/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-2/kernel_posterior_untransformed_scale/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-2/bias_posterior_loc/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH·
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B μ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0;savev2_dense_tfp_1_kernel_posterior_loc_read_readvariableopKsavev2_dense_tfp_1_kernel_posterior_untransformed_scale_read_readvariableop9savev2_dense_tfp_1_bias_posterior_loc_read_readvariableop;savev2_dense_tfp_2_kernel_posterior_loc_read_readvariableopKsavev2_dense_tfp_2_kernel_posterior_untransformed_scale_read_readvariableop9savev2_dense_tfp_2_bias_posterior_loc_read_readvariableop6savev2_output_kernel_posterior_loc_read_readvariableopFsavev2_output_kernel_posterior_untransformed_scale_read_readvariableop4savev2_output_bias_posterior_loc_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopBsavev2_adam_dense_tfp_1_kernel_posterior_loc_m_read_readvariableopRsavev2_adam_dense_tfp_1_kernel_posterior_untransformed_scale_m_read_readvariableop@savev2_adam_dense_tfp_1_bias_posterior_loc_m_read_readvariableopBsavev2_adam_dense_tfp_2_kernel_posterior_loc_m_read_readvariableopRsavev2_adam_dense_tfp_2_kernel_posterior_untransformed_scale_m_read_readvariableop@savev2_adam_dense_tfp_2_bias_posterior_loc_m_read_readvariableop=savev2_adam_output_kernel_posterior_loc_m_read_readvariableopMsavev2_adam_output_kernel_posterior_untransformed_scale_m_read_readvariableop;savev2_adam_output_bias_posterior_loc_m_read_readvariableopBsavev2_adam_dense_tfp_1_kernel_posterior_loc_v_read_readvariableopRsavev2_adam_dense_tfp_1_kernel_posterior_untransformed_scale_v_read_readvariableop@savev2_adam_dense_tfp_1_bias_posterior_loc_v_read_readvariableopBsavev2_adam_dense_tfp_2_kernel_posterior_loc_v_read_readvariableopRsavev2_adam_dense_tfp_2_kernel_posterior_untransformed_scale_v_read_readvariableop@savev2_adam_dense_tfp_2_bias_posterior_loc_v_read_readvariableop=savev2_adam_output_kernel_posterior_loc_v_read_readvariableopMsavev2_adam_output_kernel_posterior_untransformed_scale_v_read_readvariableop;savev2_adam_output_bias_posterior_loc_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: :g :g : : : ::	:	:	: : : : : : : : : :g :g : : : ::	:	:	:g :g : : : ::	:	:	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:g :$ 

_output_shapes

:g : 

_output_shapes
: :$ 

_output_shapes

: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:	:$ 

_output_shapes

:	: 	

_output_shapes
:	:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:g :$ 

_output_shapes

:g : 

_output_shapes
: :$ 

_output_shapes

: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:	:$ 

_output_shapes

:	: 

_output_shapes
:	:$ 

_output_shapes

:g :$ 

_output_shapes

:g : 

_output_shapes
: :$ 

_output_shapes

: :$  

_output_shapes

: : !

_output_shapes
::$" 

_output_shapes

:	:$# 

_output_shapes

:	: $

_output_shapes
:	:%

_output_shapes
: 
	
Φ
.__inference_dense_tfp_1_layer_call_fn_70056121

inputs
unknown:g 
	unknown_0:g 
	unknown_1: 
	unknown_2
	unknown_3
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dense_tfp_1_layer_call_and_return_conditional_losses_70054505o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????g: : : : :g 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????g
 
_user_specified_nameinputs:

_output_shapes
: :$ 

_output_shapes

:g 

Ί
-__inference_sequential_layer_call_fn_70054864	
input
unknown:g 
	unknown_0:g 
	unknown_1: 
	unknown_2
	unknown_3
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7
	unknown_8
	unknown_9:	

unknown_10:	

unknown_11:	

unknown_12

unknown_13
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????	: : : *+
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_70054828o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????g: : : : :g : : : : : : : : : :	22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????g

_user_specified_nameinput:

_output_shapes
: :$ 

_output_shapes

:g :	

_output_shapes
: :$
 

_output_shapes

: :

_output_shapes
: :$ 

_output_shapes

:	
ό
γ
H__inference_sequential_layer_call_and_return_conditional_losses_70054998

inputs&
dense_tfp_1_70054958:g &
dense_tfp_1_70054960:g "
dense_tfp_1_70054962: 
dense_tfp_1_70054964
dense_tfp_1_70054966&
dense_tfp_2_70054970: &
dense_tfp_2_70054972: "
dense_tfp_2_70054974:
dense_tfp_2_70054976
dense_tfp_2_70054978!
output_70054982:	!
output_70054984:	
output_70054986:	
output_70054988
output_70054990
identity

identity_1

identity_2

identity_3’#dense_tfp_1/StatefulPartitionedCall’#dense_tfp_2/StatefulPartitionedCall’output/StatefulPartitionedCallΞ
#dense_tfp_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_tfp_1_70054958dense_tfp_1_70054960dense_tfp_1_70054962dense_tfp_1_70054964dense_tfp_1_70054966*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dense_tfp_1_layer_call_and_return_conditional_losses_70054505τ
#dense_tfp_2/StatefulPartitionedCallStatefulPartitionedCall,dense_tfp_1/StatefulPartitionedCall:output:0dense_tfp_2_70054970dense_tfp_2_70054972dense_tfp_2_70054974dense_tfp_2_70054976dense_tfp_2_70054978*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????: *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dense_tfp_2_layer_call_and_return_conditional_losses_70054658Ρ
output/StatefulPartitionedCallStatefulPartitionedCall,dense_tfp_2/StatefulPartitionedCall:output:0output_70054982output_70054984output_70054986output_70054988output_70054990*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????	: *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_70054811v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	l

Identity_1Identity,dense_tfp_1/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: l

Identity_2Identity,dense_tfp_2/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: g

Identity_3Identity'output/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: ³
NoOpNoOp$^dense_tfp_1/StatefulPartitionedCall$^dense_tfp_2/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????g: : : : :g : : : : : : : : : :	2J
#dense_tfp_1/StatefulPartitionedCall#dense_tfp_1/StatefulPartitionedCall2J
#dense_tfp_2/StatefulPartitionedCall#dense_tfp_2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:?????????g
 
_user_specified_nameinputs:

_output_shapes
: :$ 

_output_shapes

:g :	

_output_shapes
: :$
 

_output_shapes

: :

_output_shapes
: :$ 

_output_shapes

:	
ϊ
Ρ
)__inference_output_layer_call_fn_70056435

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2
	unknown_3
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????	: *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_70054811o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : :	22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :$ 

_output_shapes

:	
ης
Ν

D__inference_output_layer_call_and_return_conditional_losses_70056575

inputs@
.normal_sample_softplus_readvariableop_resource:	2
 matmul_1_readvariableop_resource:	p
bindependentdeterministic_constructed_at_output_sample_deterministic_sample_readvariableop_resource:	Ν
Θkullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70056545Π
Λkullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1’YIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ReadVariableOp’ΫKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp’ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp’MatMul_1/ReadVariableOp’%Normal/sample/Softplus/ReadVariableOp_

zeros_likeConst*
_output_shapes

:	*
dtype0*
valueB	*    ]
Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 
%Normal/sample/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*
_output_shapes

:	*
dtype0z
Normal/sample/SoftplusSoftplus-Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:	X
Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4
Normal/sample/addAddV2Normal/sample/add/x:output:0$Normal/sample/Softplus:activations:0*
T0*
_output_shapes

:	n
Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"   	   U
Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Normal/sample/strided_sliceStridedSlice&Normal/sample/shape_as_tensor:output:0*Normal/sample/strided_slice/stack:output:0,Normal/sample/strided_slice/stack_1:output:0,Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"   	   W
Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : m
#Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:©
Normal/sample/strided_slice_1StridedSlice(Normal/sample/shape_as_tensor_1:output:0,Normal/sample/strided_slice_1/stack:output:0.Normal/sample/strided_slice_1/stack_1:output:0.Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maska
Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB c
 Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 
Normal/sample/BroadcastArgsBroadcastArgs)Normal/sample/BroadcastArgs/s0_1:output:0$Normal/sample/strided_slice:output:0*
_output_shapes
:
Normal/sample/BroadcastArgs_1BroadcastArgs Normal/sample/BroadcastArgs:r0:0&Normal/sample/strided_slice_1:output:0*
_output_shapes
:g
Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:[
Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ύ
Normal/sample/concatConcatV2&Normal/sample/concat/values_0:output:0"Normal/sample/BroadcastArgs_1:r0:0"Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:l
'Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¨
7Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat:output:0*
T0*"
_output_shapes
:	*
dtype0Π
&Normal/sample/normal/random_normal/mulMul@Normal/sample/normal/random_normal/RandomStandardNormal:output:02Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
:	Ά
"Normal/sample/normal/random_normalAddV2*Normal/sample/normal/random_normal/mul:z:00Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
:	
Normal/sample/mulMul&Normal/sample/normal/random_normal:z:0Normal/sample/add:z:0*
T0*"
_output_shapes
:	u
Normal/sample/add_1AddV2Normal/sample/mul:z:0zeros_like:output:0*
T0*"
_output_shapes
:	l
Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   	   
Normal/sample/ReshapeReshapeNormal/sample/add_1:z:0$Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

:	;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ο
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masku
+rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:t
)rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????o
)rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????
%rademacher/uniform/sanitize_seed/seedRandomUniformInt4rademacher/uniform/sanitize_seed/seed/shape:output:02rademacher/uniform/sanitize_seed/seed/min:output:02rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:q
/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R q
/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 RΗ
Hrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter.rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::q
/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :β
+rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2Shape:output:0Nrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Rrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:08rademacher/uniform/stateless_random_uniform/alg:output:08rademacher/uniform/stateless_random_uniform/min:output:08rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????*
dtype0	R
rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R
rademacher/mulMulrademacher/mul/x:output:04rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????R
rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rv
rademacher/subSubrademacher/mul:z:0rademacher/sub/y:output:0*
T0	*'
_output_shapes
:?????????l
rademacher/CastCastrademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????R
ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value	B :	P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsExpandDims/input:output:0ExpandDims/dim:output:0*
T0*
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2strided_slice:output:0ExpandDims:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:w
-rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:v
+rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????q
+rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????
'rademacher_1/uniform/sanitize_seed/seedRandomUniformInt6rademacher_1/uniform/sanitize_seed/seed/shape:output:04rademacher_1/uniform/sanitize_seed/seed/min:output:04rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:s
1rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R s
1rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 RΛ
Jrademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter0rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::s
1rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :ο
-rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2concat:output:0Prademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Trademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0:rademacher_1/uniform/stateless_random_uniform/alg:output:0:rademacher_1/uniform/stateless_random_uniform/min:output:0:rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????	*
dtype0	T
rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R
rademacher_1/mulMulrademacher_1/mul/x:output:06rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????	T
rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R|
rademacher_1/subSubrademacher_1/mul:z:0rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:?????????	p
rademacher_1/CastCastrademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????	Y
mulMulinputsrademacher/Cast:y:0*
T0*'
_output_shapes
:?????????k
MatMulMatMulmul:z:0Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????	g
mul_1MulMatMul:product:0rademacher_1/Cast:y:0*
T0*'
_output_shapes
:?????????	x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:	*
dtype0m
MatMul_1MatMulinputsMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	]
addAddV2MatMul_1:product:0	mul_1:z:0*
T0*'
_output_shapes
:?????????	
BIndependentDeterministic_CONSTRUCTED_AT_output/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 
WIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :²
hIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:ψ
YIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ReadVariableOpReadVariableOpbindependentdeterministic_constructed_at_output_sample_deterministic_sample_readvariableop_resource*
_output_shapes
:	*
dtype0€
ZIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:	
PIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ¨
^IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ͺ
`IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ͺ
`IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Π
XIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_sliceStridedSlicecIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/shape_as_tensor:output:0gIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_slice/stack:output:0iIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_slice/stack_1:output:0iIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
[IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB  
]IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB Π
XIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/BroadcastArgsBroadcastArgsfIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0aIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:€
ZIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
ZIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 
VIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
QIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concatConcatV2cIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concat/values_0:output:0]IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/BroadcastArgs:r0:0cIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concat/values_2:output:0_IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:Ν
VIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/BroadcastToBroadcastToaIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ReadVariableOp:value:0ZIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

:	©
XIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   	   Κ
RIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ReshapeReshape_IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/BroadcastTo:output:0aIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

:	
CIndependentDeterministic_CONSTRUCTED_AT_output/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:	
=IndependentDeterministic_CONSTRUCTED_AT_output/sample/ReshapeReshape[IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/Reshape:output:0LIndependentDeterministic_CONSTRUCTED_AT_output/sample/Reshape/shape:output:0*
T0*
_output_shapes
:	
BiasAddBiasAddadd:z:0FIndependentDeterministic_CONSTRUCTED_AT_output/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????	V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????	Λ
ΫKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*
_output_shapes

:	*
dtype0θ
ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplusγKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:	
ΙKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4¬
ΗKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0ΪKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes

:	Β
ΓKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLogΛKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes

:	Ή
ΕKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1LogΘkullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70056545*
T0*
_output_shapes
: 
ΓKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSubΗKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0ΙKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes

:	°
ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:	*
dtype0 
ΗKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDivΦKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0Θkullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70056545*
T0*
_output_shapes

:	
ΙKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDivΛkullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xΘkullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70056545*
T0*
_output_shapes

:	?
ΡKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifferenceΛKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0ΝKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes

:	
ΕKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
ΓKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMulΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0ΥKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes

:	
ΗKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
ΕKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1MulΠKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0ΗKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

:	Δ
ΕKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1ΙKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes

:	
ΗKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
ΕKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2MulΠKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0ΙKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes

:	
ΓKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2ΗKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0ΙKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes

:	
ΕKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1SubΗKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0ΗKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

:	Υ
KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"????ώ???τ
qKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSumΙKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * 'GΓ
truedivRealDivzKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0truediv/y:output:0*
T0*
_output_shapes
: K
divergence_kernelIdentitytruediv:z:0*
T0*
_output_shapes
: `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????	Z

Identity_1Identitydivergence_kernel:output:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOpZ^IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ReadVariableOpά^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpΟ^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp^MatMul_1/ReadVariableOp&^Normal/sample/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : :	2Ά
YIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ReadVariableOpYIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ReadVariableOp2Ό
ΫKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpΫKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2’
ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2N
%Normal/sample/Softplus/ReadVariableOp%Normal/sample/Softplus/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :$ 

_output_shapes

:	
οφ
π

I__inference_dense_tfp_2_layer_call_and_return_conditional_losses_70056419

inputs@
.normal_sample_softplus_readvariableop_resource: 2
 matmul_1_readvariableop_resource: u
gindependentdeterministic_constructed_at_dense_tfp_2_sample_deterministic_sample_readvariableop_resource:?
Νkullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70056389Υ
Πkullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1’^IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ReadVariableOp’ΰKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp’ΣKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp’MatMul_1/ReadVariableOp’%Normal/sample/Softplus/ReadVariableOp_

zeros_likeConst*
_output_shapes

: *
dtype0*
valueB *    ]
Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 
%Normal/sample/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*
_output_shapes

: *
dtype0z
Normal/sample/SoftplusSoftplus-Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

: X
Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4
Normal/sample/addAddV2Normal/sample/add/x:output:0$Normal/sample/Softplus:activations:0*
T0*
_output_shapes

: n
Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"       U
Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Normal/sample/strided_sliceStridedSlice&Normal/sample/shape_as_tensor:output:0*Normal/sample/strided_slice/stack:output:0,Normal/sample/strided_slice/stack_1:output:0,Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"       W
Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : m
#Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:©
Normal/sample/strided_slice_1StridedSlice(Normal/sample/shape_as_tensor_1:output:0,Normal/sample/strided_slice_1/stack:output:0.Normal/sample/strided_slice_1/stack_1:output:0.Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maska
Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB c
 Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 
Normal/sample/BroadcastArgsBroadcastArgs)Normal/sample/BroadcastArgs/s0_1:output:0$Normal/sample/strided_slice:output:0*
_output_shapes
:
Normal/sample/BroadcastArgs_1BroadcastArgs Normal/sample/BroadcastArgs:r0:0&Normal/sample/strided_slice_1:output:0*
_output_shapes
:g
Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:[
Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ύ
Normal/sample/concatConcatV2&Normal/sample/concat/values_0:output:0"Normal/sample/BroadcastArgs_1:r0:0"Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:l
'Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¨
7Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat:output:0*
T0*"
_output_shapes
: *
dtype0Π
&Normal/sample/normal/random_normal/mulMul@Normal/sample/normal/random_normal/RandomStandardNormal:output:02Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
: Ά
"Normal/sample/normal/random_normalAddV2*Normal/sample/normal/random_normal/mul:z:00Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
: 
Normal/sample/mulMul&Normal/sample/normal/random_normal:z:0Normal/sample/add:z:0*
T0*"
_output_shapes
: u
Normal/sample/add_1AddV2Normal/sample/mul:z:0zeros_like:output:0*
T0*"
_output_shapes
: l
Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       
Normal/sample/ReshapeReshapeNormal/sample/add_1:z:0$Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

: ;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ο
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masku
+rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:t
)rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????o
)rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????
%rademacher/uniform/sanitize_seed/seedRandomUniformInt4rademacher/uniform/sanitize_seed/seed/shape:output:02rademacher/uniform/sanitize_seed/seed/min:output:02rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:q
/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R q
/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 RΗ
Hrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter.rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::q
/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :β
+rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2Shape:output:0Nrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Rrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:08rademacher/uniform/stateless_random_uniform/alg:output:08rademacher/uniform/stateless_random_uniform/min:output:08rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:????????? *
dtype0	R
rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R
rademacher/mulMulrademacher/mul/x:output:04rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:????????? R
rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rv
rademacher/subSubrademacher/mul:z:0rademacher/sub/y:output:0*
T0	*'
_output_shapes
:????????? l
rademacher/CastCastrademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:????????? R
ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value	B :P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsExpandDims/input:output:0ExpandDims/dim:output:0*
T0*
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2strided_slice:output:0ExpandDims:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:w
-rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:v
+rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????q
+rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????
'rademacher_1/uniform/sanitize_seed/seedRandomUniformInt6rademacher_1/uniform/sanitize_seed/seed/shape:output:04rademacher_1/uniform/sanitize_seed/seed/min:output:04rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:s
1rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R s
1rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 RΛ
Jrademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter0rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::s
1rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :ο
-rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2concat:output:0Prademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Trademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0:rademacher_1/uniform/stateless_random_uniform/alg:output:0:rademacher_1/uniform/stateless_random_uniform/min:output:0:rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????*
dtype0	T
rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R
rademacher_1/mulMulrademacher_1/mul/x:output:06rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????T
rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R|
rademacher_1/subSubrademacher_1/mul:z:0rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:?????????p
rademacher_1/CastCastrademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????Y
mulMulinputsrademacher/Cast:y:0*
T0*'
_output_shapes
:????????? k
MatMulMatMulmul:z:0Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????g
mul_1MulMatMul:product:0rademacher_1/Cast:y:0*
T0*'
_output_shapes
:?????????x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0m
MatMul_1MatMulinputsMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????]
addAddV2MatMul_1:product:0	mul_1:z:0*
T0*'
_output_shapes
:?????????
GIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 
\IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :·
mIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:
^IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ReadVariableOpReadVariableOpgindependentdeterministic_constructed_at_dense_tfp_2_sample_deterministic_sample_readvariableop_resource*
_output_shapes
:*
dtype0©
_IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:
UIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ­
cIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ―
eIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:―
eIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ι
]IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_sliceStridedSlicehIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/shape_as_tensor:output:0lIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_slice/stack:output:0nIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_slice/stack_1:output:0nIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask£
`IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ₯
bIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ί
]IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/BroadcastArgsBroadcastArgskIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0fIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:©
_IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:’
_IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 
[IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
VIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concatConcatV2hIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concat/values_0:output:0bIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/BroadcastArgs:r0:0hIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concat/values_2:output:0dIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:ά
[IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/BroadcastToBroadcastTofIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ReadVariableOp:value:0_IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

:?
]IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ω
WIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ReshapeReshapedIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/BroadcastTo:output:0fIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

:
HIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:§
BIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/ReshapeReshape`IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/Reshape:output:0QIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Reshape/shape:output:0*
T0*
_output_shapes
:
BiasAddBiasAddadd:z:0KIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????Π
ΰKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*
_output_shapes

: *
dtype0ς
ΡKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplusθKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

: 
ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4»
ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2ΧKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0ίKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes

: Μ
ΘKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLogΠKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes

: Γ
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1LogΝkullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70056389*
T0*
_output_shapes
: 
ΘKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSubΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes

: ΅
ΣKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0―
ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDivΫKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0Νkullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70056389*
T0*
_output_shapes

: ¦
ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDivΠkullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xΝkullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70056389*
T0*
_output_shapes

: ½
ΦKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifferenceΠKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes

: 
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¬
ΘKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMulΣKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0ΪKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes

: 
ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @’
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1MulΥKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

: Ξ
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes

: 
ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?€
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2MulΥKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes

: 
ΘKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes

: 
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1SubΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

: Ϊ
KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"????ώ???
vKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSumΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * 'GΘ
truedivRealDivKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0truediv/y:output:0*
T0*
_output_shapes
: K
divergence_kernelIdentitytruediv:z:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????Z

Identity_1Identitydivergence_kernel:output:0^NoOp*
T0*
_output_shapes
: €
NoOpNoOp_^IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ReadVariableOpα^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpΤ^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp^MatMul_1/ReadVariableOp&^Normal/sample/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:????????? : : : : : 2ΐ
^IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ReadVariableOp^IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ReadVariableOp2Ζ
ΰKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpΰKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2¬
ΣKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpΣKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2N
%Normal/sample/Softplus/ReadVariableOp%Normal/sample/Softplus/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :$ 

_output_shapes

: 

»
-__inference_sequential_layer_call_fn_70055202

inputs
unknown:g 
	unknown_0:g 
	unknown_1: 
	unknown_2
	unknown_3
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7
	unknown_8
	unknown_9:	

unknown_10:	

unknown_11:	

unknown_12

unknown_13
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????	: : : *+
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_70054828o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????g: : : : :g : : : : : : : : : :	22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????g
 
_user_specified_nameinputs:

_output_shapes
: :$ 

_output_shapes

:g :	

_output_shapes
: :$
 

_output_shapes

: :

_output_shapes
: :$ 

_output_shapes

:	
³
?
H__inference_sequential_layer_call_and_return_conditional_losses_70055654

inputsL
:dense_tfp_1_normal_sample_softplus_readvariableop_resource:g >
,dense_tfp_1_matmul_1_readvariableop_resource:g f
Xdense_tfp_1_independentdeterministic_sample_deterministic_sample_readvariableop_resource: ή
Ωdense_tfp_1_kullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70055352α
άdense_tfp_1_kullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xL
:dense_tfp_2_normal_sample_softplus_readvariableop_resource: >
,dense_tfp_2_matmul_1_readvariableop_resource: f
Xdense_tfp_2_independentdeterministic_sample_deterministic_sample_readvariableop_resource:ή
Ωdense_tfp_2_kullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70055487α
άdense_tfp_2_kullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xG
5output_normal_sample_softplus_readvariableop_resource:	9
'output_matmul_1_readvariableop_resource:	a
Soutput_independentdeterministic_sample_deterministic_sample_readvariableop_resource:	Τ
Οoutput_kullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70055622Χ
?output_kullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1

identity_2

identity_3’Odense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp’μdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp’ίdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp’#dense_tfp_1/MatMul_1/ReadVariableOp’1dense_tfp_1/Normal/sample/Softplus/ReadVariableOp’Odense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp’μdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp’ίdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp’#dense_tfp_2/MatMul_1/ReadVariableOp’1dense_tfp_2/Normal/sample/Softplus/ReadVariableOp’Joutput/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp’βoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp’Υoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp’output/MatMul_1/ReadVariableOp’,output/Normal/sample/Softplus/ReadVariableOpw
&dense_tfp_1/zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"g       a
dense_tfp_1/zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_tfp_1/zeros_likeFill/dense_tfp_1/zeros_like/shape_as_tensor:output:0%dense_tfp_1/zeros_like/Const:output:0*
T0*
_output_shapes

:g i
&dense_tfp_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ¬
1dense_tfp_1/Normal/sample/Softplus/ReadVariableOpReadVariableOp:dense_tfp_1_normal_sample_softplus_readvariableop_resource*
_output_shapes

:g *
dtype0
"dense_tfp_1/Normal/sample/SoftplusSoftplus9dense_tfp_1/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:g d
dense_tfp_1/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4«
dense_tfp_1/Normal/sample/addAddV2(dense_tfp_1/Normal/sample/add/x:output:00dense_tfp_1/Normal/sample/Softplus:activations:0*
T0*
_output_shapes

:g z
)dense_tfp_1/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"g       a
dense_tfp_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : w
-dense_tfp_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense_tfp_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense_tfp_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ϋ
'dense_tfp_1/Normal/sample/strided_sliceStridedSlice2dense_tfp_1/Normal/sample/shape_as_tensor:output:06dense_tfp_1/Normal/sample/strided_slice/stack:output:08dense_tfp_1/Normal/sample/strided_slice/stack_1:output:08dense_tfp_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
+dense_tfp_1/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"g       c
!dense_tfp_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : y
/dense_tfp_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_tfp_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_tfp_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ε
)dense_tfp_1/Normal/sample/strided_slice_1StridedSlice4dense_tfp_1/Normal/sample/shape_as_tensor_1:output:08dense_tfp_1/Normal/sample/strided_slice_1/stack:output:0:dense_tfp_1/Normal/sample/strided_slice_1/stack_1:output:0:dense_tfp_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
*dense_tfp_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB o
,dense_tfp_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ½
'dense_tfp_1/Normal/sample/BroadcastArgsBroadcastArgs5dense_tfp_1/Normal/sample/BroadcastArgs/s0_1:output:00dense_tfp_1/Normal/sample/strided_slice:output:0*
_output_shapes
:Έ
)dense_tfp_1/Normal/sample/BroadcastArgs_1BroadcastArgs,dense_tfp_1/Normal/sample/BroadcastArgs:r0:02dense_tfp_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:s
)dense_tfp_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
%dense_tfp_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ξ
 dense_tfp_1/Normal/sample/concatConcatV22dense_tfp_1/Normal/sample/concat/values_0:output:0.dense_tfp_1/Normal/sample/BroadcastArgs_1:r0:0.dense_tfp_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:x
3dense_tfp_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    z
5dense_tfp_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ΐ
Cdense_tfp_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal)dense_tfp_1/Normal/sample/concat:output:0*
T0*"
_output_shapes
:g *
dtype0τ
2dense_tfp_1/Normal/sample/normal/random_normal/mulMulLdense_tfp_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0>dense_tfp_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
:g Ϊ
.dense_tfp_1/Normal/sample/normal/random_normalAddV26dense_tfp_1/Normal/sample/normal/random_normal/mul:z:0<dense_tfp_1/Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
:g ¨
dense_tfp_1/Normal/sample/mulMul2dense_tfp_1/Normal/sample/normal/random_normal:z:0!dense_tfp_1/Normal/sample/add:z:0*
T0*"
_output_shapes
:g 
dense_tfp_1/Normal/sample/add_1AddV2!dense_tfp_1/Normal/sample/mul:z:0dense_tfp_1/zeros_like:output:0*
T0*"
_output_shapes
:g x
'dense_tfp_1/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"g       ¬
!dense_tfp_1/Normal/sample/ReshapeReshape#dense_tfp_1/Normal/sample/add_1:z:00dense_tfp_1/Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

:g G
dense_tfp_1/ShapeShapeinputs*
T0*
_output_shapes
:i
dense_tfp_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
!dense_tfp_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????k
!dense_tfp_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
dense_tfp_1/strided_sliceStridedSlicedense_tfp_1/Shape:output:0(dense_tfp_1/strided_slice/stack:output:0*dense_tfp_1/strided_slice/stack_1:output:0*dense_tfp_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
7dense_tfp_1/rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:
5dense_tfp_1/rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????{
5dense_tfp_1/rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????Έ
1dense_tfp_1/rademacher/uniform/sanitize_seed/seedRandomUniformInt@dense_tfp_1/rademacher/uniform/sanitize_seed/seed/shape:output:0>dense_tfp_1/rademacher/uniform/sanitize_seed/seed/min:output:0>dense_tfp_1/rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:}
;dense_tfp_1/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R }
;dense_tfp_1/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 Rί
Tdense_tfp_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter:dense_tfp_1/rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::}
;dense_tfp_1/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :Ά
7dense_tfp_1/rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2dense_tfp_1/Shape:output:0Zdense_tfp_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0^dense_tfp_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Ddense_tfp_1/rademacher/uniform/stateless_random_uniform/alg:output:0Ddense_tfp_1/rademacher/uniform/stateless_random_uniform/min:output:0Ddense_tfp_1/rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????g*
dtype0	^
dense_tfp_1/rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 RΌ
dense_tfp_1/rademacher/mulMul%dense_tfp_1/rademacher/mul/x:output:0@dense_tfp_1/rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????g^
dense_tfp_1/rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R
dense_tfp_1/rademacher/subSubdense_tfp_1/rademacher/mul:z:0%dense_tfp_1/rademacher/sub/y:output:0*
T0	*'
_output_shapes
:?????????g
dense_tfp_1/rademacher/CastCastdense_tfp_1/rademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????g^
dense_tfp_1/ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value	B : \
dense_tfp_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
dense_tfp_1/ExpandDims
ExpandDims%dense_tfp_1/ExpandDims/input:output:0#dense_tfp_1/ExpandDims/dim:output:0*
T0*
_output_shapes
:Y
dense_tfp_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ³
dense_tfp_1/concatConcatV2"dense_tfp_1/strided_slice:output:0dense_tfp_1/ExpandDims:output:0 dense_tfp_1/concat/axis:output:0*
N*
T0*
_output_shapes
:
9dense_tfp_1/rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:
7dense_tfp_1/rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????}
7dense_tfp_1/rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????ΐ
3dense_tfp_1/rademacher_1/uniform/sanitize_seed/seedRandomUniformIntBdense_tfp_1/rademacher_1/uniform/sanitize_seed/seed/shape:output:0@dense_tfp_1/rademacher_1/uniform/sanitize_seed/seed/min:output:0@dense_tfp_1/rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:
=dense_tfp_1/rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R 
=dense_tfp_1/rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 Rγ
Vdense_tfp_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter<dense_tfp_1/rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::
=dense_tfp_1/rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :Γ
9dense_tfp_1/rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2dense_tfp_1/concat:output:0\dense_tfp_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`dense_tfp_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Fdense_tfp_1/rademacher_1/uniform/stateless_random_uniform/alg:output:0Fdense_tfp_1/rademacher_1/uniform/stateless_random_uniform/min:output:0Fdense_tfp_1/rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:????????? *
dtype0	`
dense_tfp_1/rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 RΒ
dense_tfp_1/rademacher_1/mulMul'dense_tfp_1/rademacher_1/mul/x:output:0Bdense_tfp_1/rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:????????? `
dense_tfp_1/rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
dense_tfp_1/rademacher_1/subSub dense_tfp_1/rademacher_1/mul:z:0'dense_tfp_1/rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:????????? 
dense_tfp_1/rademacher_1/CastCast dense_tfp_1/rademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:????????? q
dense_tfp_1/mulMulinputsdense_tfp_1/rademacher/Cast:y:0*
T0*'
_output_shapes
:?????????g
dense_tfp_1/MatMulMatMuldense_tfp_1/mul:z:0*dense_tfp_1/Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:????????? 
dense_tfp_1/mul_1Muldense_tfp_1/MatMul:product:0!dense_tfp_1/rademacher_1/Cast:y:0*
T0*'
_output_shapes
:????????? 
#dense_tfp_1/MatMul_1/ReadVariableOpReadVariableOp,dense_tfp_1_matmul_1_readvariableop_resource*
_output_shapes

:g *
dtype0
dense_tfp_1/MatMul_1MatMulinputs+dense_tfp_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 
dense_tfp_1/addAddV2dense_tfp_1/MatMul_1:product:0dense_tfp_1/mul_1:z:0*
T0*'
_output_shapes
:????????? {
8dense_tfp_1/IndependentDeterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 
Mdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :¨
^dense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:δ
Odense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpReadVariableOpXdense_tfp_1_independentdeterministic_sample_deterministic_sample_readvariableop_resource*
_output_shapes
: *
dtype0
Pdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: 
Fdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Tdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:  
Vdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Vdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ndense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/strided_sliceStridedSliceYdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensor:output:0]dense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack:output:0_dense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1:output:0_dense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Qdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 
Sdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ²
Ndense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgsBroadcastArgs\dense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0Wdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:
Pdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Pdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 
Ldense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : γ
Gdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/concatConcatV2Ydense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_0:output:0Sdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs:r0:0Ydense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_2:output:0Udense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:―
Ldense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastToBroadcastToWdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp:value:0Pdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

: 
Ndense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       ¬
Hdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/ReshapeReshapeUdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastTo:output:0Wdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

: 
9dense_tfp_1/IndependentDeterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: ϊ
3dense_tfp_1/IndependentDeterministic/sample/ReshapeReshapeQdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/Reshape:output:0Bdense_tfp_1/IndependentDeterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes
: £
dense_tfp_1/BiasAddBiasAdddense_tfp_1/add:z:0<dense_tfp_1/IndependentDeterministic/sample/Reshape:output:0*
T0*'
_output_shapes
:????????? h
dense_tfp_1/ReluReludense_tfp_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? θ
μdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp:dense_tfp_1_normal_sample_softplus_readvariableop_resource*
_output_shapes

:g *
dtype0
έdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplusτdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:g  
Ϊdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4ί
Ψdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2γdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0λdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes

:g δ
Τdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLogάdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes

:g Ϋ
Φdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1LogΩdense_tfp_1_kullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70055352*
T0*
_output_shapes
: ½
Τdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSubΨdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0Ϊdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes

:g Ν
ίdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp,dense_tfp_1_matmul_1_readvariableop_resource*
_output_shapes

:g *
dtype0Σ
Ψdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDivηdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0Ωdense_tfp_1_kullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70055352*
T0*
_output_shapes

:g Κ
Ϊdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDivάdense_tfp_1_kullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xΩdense_tfp_1_kullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70055352*
T0*
_output_shapes

:g α
βdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifferenceάdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0ήdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes

:g 
Φdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Π
Τdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMulίdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0ζdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes

:g 
Ψdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ζ
Φdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mulαdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0Ψdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

:g ζ
Φdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1Ϊdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes

:g 
Ψdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Θ
Φdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mulαdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0Ϊdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes

:g Ώ
Τdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2Ψdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0Ϊdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes

:g ½
Φdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1SubΨdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0Ψdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

:g ζ
dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"????ώ???¨
dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSumΪdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: Z
dense_tfp_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * 'Gν
dense_tfp_1/truedivRealDivdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0dense_tfp_1/truediv/y:output:0*
T0*
_output_shapes
: c
dense_tfp_1/divergence_kernelIdentitydense_tfp_1/truediv:z:0*
T0*
_output_shapes
: k
dense_tfp_2/zeros_likeConst*
_output_shapes

: *
dtype0*
valueB *    i
&dense_tfp_2/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ¬
1dense_tfp_2/Normal/sample/Softplus/ReadVariableOpReadVariableOp:dense_tfp_2_normal_sample_softplus_readvariableop_resource*
_output_shapes

: *
dtype0
"dense_tfp_2/Normal/sample/SoftplusSoftplus9dense_tfp_2/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

: d
dense_tfp_2/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4«
dense_tfp_2/Normal/sample/addAddV2(dense_tfp_2/Normal/sample/add/x:output:00dense_tfp_2/Normal/sample/Softplus:activations:0*
T0*
_output_shapes

: z
)dense_tfp_2/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"       a
dense_tfp_2/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : w
-dense_tfp_2/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense_tfp_2/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense_tfp_2/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ϋ
'dense_tfp_2/Normal/sample/strided_sliceStridedSlice2dense_tfp_2/Normal/sample/shape_as_tensor:output:06dense_tfp_2/Normal/sample/strided_slice/stack:output:08dense_tfp_2/Normal/sample/strided_slice/stack_1:output:08dense_tfp_2/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
+dense_tfp_2/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"       c
!dense_tfp_2/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : y
/dense_tfp_2/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_tfp_2/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_tfp_2/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ε
)dense_tfp_2/Normal/sample/strided_slice_1StridedSlice4dense_tfp_2/Normal/sample/shape_as_tensor_1:output:08dense_tfp_2/Normal/sample/strided_slice_1/stack:output:0:dense_tfp_2/Normal/sample/strided_slice_1/stack_1:output:0:dense_tfp_2/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
*dense_tfp_2/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB o
,dense_tfp_2/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ½
'dense_tfp_2/Normal/sample/BroadcastArgsBroadcastArgs5dense_tfp_2/Normal/sample/BroadcastArgs/s0_1:output:00dense_tfp_2/Normal/sample/strided_slice:output:0*
_output_shapes
:Έ
)dense_tfp_2/Normal/sample/BroadcastArgs_1BroadcastArgs,dense_tfp_2/Normal/sample/BroadcastArgs:r0:02dense_tfp_2/Normal/sample/strided_slice_1:output:0*
_output_shapes
:s
)dense_tfp_2/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
%dense_tfp_2/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ξ
 dense_tfp_2/Normal/sample/concatConcatV22dense_tfp_2/Normal/sample/concat/values_0:output:0.dense_tfp_2/Normal/sample/BroadcastArgs_1:r0:0.dense_tfp_2/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:x
3dense_tfp_2/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    z
5dense_tfp_2/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ΐ
Cdense_tfp_2/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal)dense_tfp_2/Normal/sample/concat:output:0*
T0*"
_output_shapes
: *
dtype0τ
2dense_tfp_2/Normal/sample/normal/random_normal/mulMulLdense_tfp_2/Normal/sample/normal/random_normal/RandomStandardNormal:output:0>dense_tfp_2/Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
: Ϊ
.dense_tfp_2/Normal/sample/normal/random_normalAddV26dense_tfp_2/Normal/sample/normal/random_normal/mul:z:0<dense_tfp_2/Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
: ¨
dense_tfp_2/Normal/sample/mulMul2dense_tfp_2/Normal/sample/normal/random_normal:z:0!dense_tfp_2/Normal/sample/add:z:0*
T0*"
_output_shapes
: 
dense_tfp_2/Normal/sample/add_1AddV2!dense_tfp_2/Normal/sample/mul:z:0dense_tfp_2/zeros_like:output:0*
T0*"
_output_shapes
: x
'dense_tfp_2/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       ¬
!dense_tfp_2/Normal/sample/ReshapeReshape#dense_tfp_2/Normal/sample/add_1:z:00dense_tfp_2/Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

: _
dense_tfp_2/ShapeShapedense_tfp_1/Relu:activations:0*
T0*
_output_shapes
:i
dense_tfp_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
!dense_tfp_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????k
!dense_tfp_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
dense_tfp_2/strided_sliceStridedSlicedense_tfp_2/Shape:output:0(dense_tfp_2/strided_slice/stack:output:0*dense_tfp_2/strided_slice/stack_1:output:0*dense_tfp_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
7dense_tfp_2/rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:
5dense_tfp_2/rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????{
5dense_tfp_2/rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????Έ
1dense_tfp_2/rademacher/uniform/sanitize_seed/seedRandomUniformInt@dense_tfp_2/rademacher/uniform/sanitize_seed/seed/shape:output:0>dense_tfp_2/rademacher/uniform/sanitize_seed/seed/min:output:0>dense_tfp_2/rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:}
;dense_tfp_2/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R }
;dense_tfp_2/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 Rί
Tdense_tfp_2/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter:dense_tfp_2/rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::}
;dense_tfp_2/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :Ά
7dense_tfp_2/rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2dense_tfp_2/Shape:output:0Zdense_tfp_2/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0^dense_tfp_2/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Ddense_tfp_2/rademacher/uniform/stateless_random_uniform/alg:output:0Ddense_tfp_2/rademacher/uniform/stateless_random_uniform/min:output:0Ddense_tfp_2/rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:????????? *
dtype0	^
dense_tfp_2/rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 RΌ
dense_tfp_2/rademacher/mulMul%dense_tfp_2/rademacher/mul/x:output:0@dense_tfp_2/rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:????????? ^
dense_tfp_2/rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R
dense_tfp_2/rademacher/subSubdense_tfp_2/rademacher/mul:z:0%dense_tfp_2/rademacher/sub/y:output:0*
T0	*'
_output_shapes
:????????? 
dense_tfp_2/rademacher/CastCastdense_tfp_2/rademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:????????? ^
dense_tfp_2/ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value	B :\
dense_tfp_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
dense_tfp_2/ExpandDims
ExpandDims%dense_tfp_2/ExpandDims/input:output:0#dense_tfp_2/ExpandDims/dim:output:0*
T0*
_output_shapes
:Y
dense_tfp_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ³
dense_tfp_2/concatConcatV2"dense_tfp_2/strided_slice:output:0dense_tfp_2/ExpandDims:output:0 dense_tfp_2/concat/axis:output:0*
N*
T0*
_output_shapes
:
9dense_tfp_2/rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:
7dense_tfp_2/rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????}
7dense_tfp_2/rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????ΐ
3dense_tfp_2/rademacher_1/uniform/sanitize_seed/seedRandomUniformIntBdense_tfp_2/rademacher_1/uniform/sanitize_seed/seed/shape:output:0@dense_tfp_2/rademacher_1/uniform/sanitize_seed/seed/min:output:0@dense_tfp_2/rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:
=dense_tfp_2/rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R 
=dense_tfp_2/rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 Rγ
Vdense_tfp_2/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter<dense_tfp_2/rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::
=dense_tfp_2/rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :Γ
9dense_tfp_2/rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2dense_tfp_2/concat:output:0\dense_tfp_2/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`dense_tfp_2/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Fdense_tfp_2/rademacher_1/uniform/stateless_random_uniform/alg:output:0Fdense_tfp_2/rademacher_1/uniform/stateless_random_uniform/min:output:0Fdense_tfp_2/rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????*
dtype0	`
dense_tfp_2/rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 RΒ
dense_tfp_2/rademacher_1/mulMul'dense_tfp_2/rademacher_1/mul/x:output:0Bdense_tfp_2/rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????`
dense_tfp_2/rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
dense_tfp_2/rademacher_1/subSub dense_tfp_2/rademacher_1/mul:z:0'dense_tfp_2/rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:?????????
dense_tfp_2/rademacher_1/CastCast dense_tfp_2/rademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????
dense_tfp_2/mulMuldense_tfp_1/Relu:activations:0dense_tfp_2/rademacher/Cast:y:0*
T0*'
_output_shapes
:????????? 
dense_tfp_2/MatMulMatMuldense_tfp_2/mul:z:0*dense_tfp_2/Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????
dense_tfp_2/mul_1Muldense_tfp_2/MatMul:product:0!dense_tfp_2/rademacher_1/Cast:y:0*
T0*'
_output_shapes
:?????????
#dense_tfp_2/MatMul_1/ReadVariableOpReadVariableOp,dense_tfp_2_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0
dense_tfp_2/MatMul_1MatMuldense_tfp_1/Relu:activations:0+dense_tfp_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_tfp_2/addAddV2dense_tfp_2/MatMul_1:product:0dense_tfp_2/mul_1:z:0*
T0*'
_output_shapes
:?????????{
8dense_tfp_2/IndependentDeterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 
Mdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :¨
^dense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:δ
Odense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpReadVariableOpXdense_tfp_2_independentdeterministic_sample_deterministic_sample_readvariableop_resource*
_output_shapes
:*
dtype0
Pdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:
Fdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Tdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:  
Vdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Vdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ndense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/strided_sliceStridedSliceYdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensor:output:0]dense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack:output:0_dense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1:output:0_dense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Qdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 
Sdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ²
Ndense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgsBroadcastArgs\dense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0Wdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:
Pdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Pdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 
Ldense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : γ
Gdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/concatConcatV2Ydense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/concat/values_0:output:0Sdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs:r0:0Ydense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/concat/values_2:output:0Udense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:―
Ldense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/BroadcastToBroadcastToWdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp:value:0Pdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

:
Ndense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ¬
Hdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/ReshapeReshapeUdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/BroadcastTo:output:0Wdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

:
9dense_tfp_2/IndependentDeterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:ϊ
3dense_tfp_2/IndependentDeterministic/sample/ReshapeReshapeQdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/Reshape:output:0Bdense_tfp_2/IndependentDeterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes
:£
dense_tfp_2/BiasAddBiasAdddense_tfp_2/add:z:0<dense_tfp_2/IndependentDeterministic/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????h
dense_tfp_2/ReluReludense_tfp_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????θ
μdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp:dense_tfp_2_normal_sample_softplus_readvariableop_resource*
_output_shapes

: *
dtype0
έdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplusτdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:  
Ϊdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4ί
Ψdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2γdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0λdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes

: δ
Τdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLogάdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes

: Ϋ
Φdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1LogΩdense_tfp_2_kullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70055487*
T0*
_output_shapes
: ½
Τdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSubΨdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0Ϊdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes

: Ν
ίdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp,dense_tfp_2_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0Σ
Ψdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDivηdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0Ωdense_tfp_2_kullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70055487*
T0*
_output_shapes

: Κ
Ϊdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDivάdense_tfp_2_kullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xΩdense_tfp_2_kullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70055487*
T0*
_output_shapes

: α
βdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifferenceάdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0ήdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes

: 
Φdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Π
Τdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMulίdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0ζdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes

: 
Ψdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ζ
Φdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mulαdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0Ψdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

: ζ
Φdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1Ϊdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes

: 
Ψdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Θ
Φdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mulαdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0Ϊdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes

: Ώ
Τdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2Ψdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0Ϊdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes

: ½
Φdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1SubΨdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0Ψdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

: ζ
dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"????ώ???¨
dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSumΪdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: Z
dense_tfp_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * 'Gν
dense_tfp_2/truedivRealDivdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0dense_tfp_2/truediv/y:output:0*
T0*
_output_shapes
: c
dense_tfp_2/divergence_kernelIdentitydense_tfp_2/truediv:z:0*
T0*
_output_shapes
: f
output/zeros_likeConst*
_output_shapes

:	*
dtype0*
valueB	*    d
!output/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ’
,output/Normal/sample/Softplus/ReadVariableOpReadVariableOp5output_normal_sample_softplus_readvariableop_resource*
_output_shapes

:	*
dtype0
output/Normal/sample/SoftplusSoftplus4output/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:	_
output/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4
output/Normal/sample/addAddV2#output/Normal/sample/add/x:output:0+output/Normal/sample/Softplus:activations:0*
T0*
_output_shapes

:	u
$output/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"   	   \
output/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(output/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*output/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*output/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Β
"output/Normal/sample/strided_sliceStridedSlice-output/Normal/sample/shape_as_tensor:output:01output/Normal/sample/strided_slice/stack:output:03output/Normal/sample/strided_slice/stack_1:output:03output/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskw
&output/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"   	   ^
output/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : t
*output/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,output/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,output/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Μ
$output/Normal/sample/strided_slice_1StridedSlice/output/Normal/sample/shape_as_tensor_1:output:03output/Normal/sample/strided_slice_1/stack:output:05output/Normal/sample/strided_slice_1/stack_1:output:05output/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%output/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'output/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
"output/Normal/sample/BroadcastArgsBroadcastArgs0output/Normal/sample/BroadcastArgs/s0_1:output:0+output/Normal/sample/strided_slice:output:0*
_output_shapes
:©
$output/Normal/sample/BroadcastArgs_1BroadcastArgs'output/Normal/sample/BroadcastArgs:r0:0-output/Normal/sample/strided_slice_1:output:0*
_output_shapes
:n
$output/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:b
 output/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ϊ
output/Normal/sample/concatConcatV2-output/Normal/sample/concat/values_0:output:0)output/Normal/sample/BroadcastArgs_1:r0:0)output/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:s
.output/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    u
0output/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ά
>output/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal$output/Normal/sample/concat:output:0*
T0*"
_output_shapes
:	*
dtype0ε
-output/Normal/sample/normal/random_normal/mulMulGoutput/Normal/sample/normal/random_normal/RandomStandardNormal:output:09output/Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
:	Λ
)output/Normal/sample/normal/random_normalAddV21output/Normal/sample/normal/random_normal/mul:z:07output/Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
:	
output/Normal/sample/mulMul-output/Normal/sample/normal/random_normal:z:0output/Normal/sample/add:z:0*
T0*"
_output_shapes
:	
output/Normal/sample/add_1AddV2output/Normal/sample/mul:z:0output/zeros_like:output:0*
T0*"
_output_shapes
:	s
"output/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   	   
output/Normal/sample/ReshapeReshapeoutput/Normal/sample/add_1:z:0+output/Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

:	Z
output/ShapeShapedense_tfp_2/Relu:activations:0*
T0*
_output_shapes
:d
output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ς
output/strided_sliceStridedSliceoutput/Shape:output:0#output/strided_slice/stack:output:0%output/strided_slice/stack_1:output:0%output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
2output/rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:{
0output/rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????v
0output/rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????€
,output/rademacher/uniform/sanitize_seed/seedRandomUniformInt;output/rademacher/uniform/sanitize_seed/seed/shape:output:09output/rademacher/uniform/sanitize_seed/seed/min:output:09output/rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:x
6output/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R x
6output/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 RΥ
Ooutput/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter5output/rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::x
6output/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :
2output/rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2output/Shape:output:0Uoutput/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Youtput/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0?output/rademacher/uniform/stateless_random_uniform/alg:output:0?output/rademacher/uniform/stateless_random_uniform/min:output:0?output/rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????*
dtype0	Y
output/rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R­
output/rademacher/mulMul output/rademacher/mul/x:output:0;output/rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????Y
output/rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R
output/rademacher/subSuboutput/rademacher/mul:z:0 output/rademacher/sub/y:output:0*
T0	*'
_output_shapes
:?????????z
output/rademacher/CastCastoutput/rademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????Y
output/ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value	B :	W
output/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
output/ExpandDims
ExpandDims output/ExpandDims/input:output:0output/ExpandDims/dim:output:0*
T0*
_output_shapes
:T
output/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
output/concatConcatV2output/strided_slice:output:0output/ExpandDims:output:0output/concat/axis:output:0*
N*
T0*
_output_shapes
:~
4output/rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:}
2output/rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????x
2output/rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????¬
.output/rademacher_1/uniform/sanitize_seed/seedRandomUniformInt=output/rademacher_1/uniform/sanitize_seed/seed/shape:output:0;output/rademacher_1/uniform/sanitize_seed/seed/min:output:0;output/rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:z
8output/rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R z
8output/rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 RΩ
Qoutput/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter7output/rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::z
8output/rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B : 
4output/rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2output/concat:output:0Woutput/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0[output/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Aoutput/rademacher_1/uniform/stateless_random_uniform/alg:output:0Aoutput/rademacher_1/uniform/stateless_random_uniform/min:output:0Aoutput/rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????	*
dtype0	[
output/rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R³
output/rademacher_1/mulMul"output/rademacher_1/mul/x:output:0=output/rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????	[
output/rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R
output/rademacher_1/subSuboutput/rademacher_1/mul:z:0"output/rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:?????????	~
output/rademacher_1/CastCastoutput/rademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????	

output/mulMuldense_tfp_2/Relu:activations:0output/rademacher/Cast:y:0*
T0*'
_output_shapes
:?????????
output/MatMulMatMuloutput/mul:z:0%output/Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????	|
output/mul_1Muloutput/MatMul:product:0output/rademacher_1/Cast:y:0*
T0*'
_output_shapes
:?????????	
output/MatMul_1/ReadVariableOpReadVariableOp'output_matmul_1_readvariableop_resource*
_output_shapes

:	*
dtype0
output/MatMul_1MatMuldense_tfp_2/Relu:activations:0&output/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	r

output/addAddV2output/MatMul_1:product:0output/mul_1:z:0*
T0*'
_output_shapes
:?????????	v
3output/IndependentDeterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 
Houtput/IndependentDeterministic/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :£
Youtput/IndependentDeterministic/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ϊ
Joutput/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpReadVariableOpSoutput_independentdeterministic_sample_deterministic_sample_readvariableop_resource*
_output_shapes
:	*
dtype0
Koutput/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:	
Aoutput/IndependentDeterministic/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Ooutput/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Qoutput/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Qoutput/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ioutput/IndependentDeterministic/sample/Deterministic/sample/strided_sliceStridedSliceToutput/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensor:output:0Xoutput/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack:output:0Zoutput/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1:output:0Zoutput/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Loutput/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 
Noutput/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB £
Ioutput/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgsBroadcastArgsWoutput/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0Routput/IndependentDeterministic/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:
Koutput/IndependentDeterministic/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Koutput/IndependentDeterministic/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 
Goutput/IndependentDeterministic/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Κ
Boutput/IndependentDeterministic/sample/Deterministic/sample/concatConcatV2Toutput/IndependentDeterministic/sample/Deterministic/sample/concat/values_0:output:0Noutput/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs:r0:0Toutput/IndependentDeterministic/sample/Deterministic/sample/concat/values_2:output:0Poutput/IndependentDeterministic/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
: 
Goutput/IndependentDeterministic/sample/Deterministic/sample/BroadcastToBroadcastToRoutput/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp:value:0Koutput/IndependentDeterministic/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

:	
Ioutput/IndependentDeterministic/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   	   
Coutput/IndependentDeterministic/sample/Deterministic/sample/ReshapeReshapePoutput/IndependentDeterministic/sample/Deterministic/sample/BroadcastTo:output:0Routput/IndependentDeterministic/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

:	~
4output/IndependentDeterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:	λ
.output/IndependentDeterministic/sample/ReshapeReshapeLoutput/IndependentDeterministic/sample/Deterministic/sample/Reshape:output:0=output/IndependentDeterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes
:	
output/BiasAddBiasAddoutput/add:z:07output/IndependentDeterministic/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????	d
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	Ω
βoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp5output_normal_sample_softplus_readvariableop_resource*
_output_shapes

:	*
dtype0φ
Σoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplusκoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:	
Πoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4Α
Ξoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2Ωoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0αoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes

:	Π
Κoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog?output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes

:	Η
Μoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1LogΟoutput_kullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70055622*
T0*
_output_shapes
: 
Κoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSubΞoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0Πoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes

:	Ύ
Υoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp'output_matmul_1_readvariableop_resource*
_output_shapes

:	*
dtype0΅
Ξoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDivέoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0Οoutput_kullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70055622*
T0*
_output_shapes

:	¬
Πoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv?output_kullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xΟoutput_kullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70055622*
T0*
_output_shapes

:	Γ
Ψoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference?output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0Τoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes

:	
Μoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?²
Κoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMulΥoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0άoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes

:	
Ξoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @¨
Μoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1MulΧoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0Ξoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

:	?
Μoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1Πoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes

:	
Ξoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ͺ
Μoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2MulΧoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0Πoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes

:	‘
Κoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2Ξoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0Πoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes

:	
Μoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1SubΞoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0Ξoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

:	ά
output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"????ώ???
xoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSumΠoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: U
output/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * 'GΩ
output/truedivRealDivoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0output/truediv/y:output:0*
T0*
_output_shapes
: Y
output/divergence_kernelIdentityoutput/truediv:z:0*
T0*
_output_shapes
: g
IdentityIdentityoutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????	f

Identity_1Identity&dense_tfp_1/divergence_kernel:output:0^NoOp*
T0*
_output_shapes
: f

Identity_2Identity&dense_tfp_2/divergence_kernel:output:0^NoOp*
T0*
_output_shapes
: a

Identity_3Identity!output/divergence_kernel:output:0^NoOp*
T0*
_output_shapes
:  
NoOpNoOpP^dense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpν^dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpΰ^dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp$^dense_tfp_1/MatMul_1/ReadVariableOp2^dense_tfp_1/Normal/sample/Softplus/ReadVariableOpP^dense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpν^dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpΰ^dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp$^dense_tfp_2/MatMul_1/ReadVariableOp2^dense_tfp_2/Normal/sample/Softplus/ReadVariableOpK^output/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpγ^output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpΦ^output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp^output/MatMul_1/ReadVariableOp-^output/Normal/sample/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????g: : : : :g : : : : : : : : : :	2’
Odense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpOdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp2ή
μdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpμdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2Δ
ίdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpίdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2J
#dense_tfp_1/MatMul_1/ReadVariableOp#dense_tfp_1/MatMul_1/ReadVariableOp2f
1dense_tfp_1/Normal/sample/Softplus/ReadVariableOp1dense_tfp_1/Normal/sample/Softplus/ReadVariableOp2’
Odense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpOdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp2ή
μdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpμdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2Δ
ίdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpίdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2J
#dense_tfp_2/MatMul_1/ReadVariableOp#dense_tfp_2/MatMul_1/ReadVariableOp2f
1dense_tfp_2/Normal/sample/Softplus/ReadVariableOp1dense_tfp_2/Normal/sample/Softplus/ReadVariableOp2
Joutput/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpJoutput/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp2Κ
βoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpβoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2°
Υoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpΥoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2@
output/MatMul_1/ReadVariableOpoutput/MatMul_1/ReadVariableOp2\
,output/Normal/sample/Softplus/ReadVariableOp,output/Normal/sample/Softplus/ReadVariableOp:O K
'
_output_shapes
:?????????g
 
_user_specified_nameinputs:

_output_shapes
: :$ 

_output_shapes

:g :	

_output_shapes
: :$
 

_output_shapes

: :

_output_shapes
: :$ 

_output_shapes

:	
οφ
π

I__inference_dense_tfp_2_layer_call_and_return_conditional_losses_70054658

inputs@
.normal_sample_softplus_readvariableop_resource: 2
 matmul_1_readvariableop_resource: u
gindependentdeterministic_constructed_at_dense_tfp_2_sample_deterministic_sample_readvariableop_resource:?
Νkullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054628Υ
Πkullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1’^IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ReadVariableOp’ΰKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp’ΣKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp’MatMul_1/ReadVariableOp’%Normal/sample/Softplus/ReadVariableOp_

zeros_likeConst*
_output_shapes

: *
dtype0*
valueB *    ]
Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 
%Normal/sample/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*
_output_shapes

: *
dtype0z
Normal/sample/SoftplusSoftplus-Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

: X
Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4
Normal/sample/addAddV2Normal/sample/add/x:output:0$Normal/sample/Softplus:activations:0*
T0*
_output_shapes

: n
Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"       U
Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Normal/sample/strided_sliceStridedSlice&Normal/sample/shape_as_tensor:output:0*Normal/sample/strided_slice/stack:output:0,Normal/sample/strided_slice/stack_1:output:0,Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"       W
Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : m
#Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:©
Normal/sample/strided_slice_1StridedSlice(Normal/sample/shape_as_tensor_1:output:0,Normal/sample/strided_slice_1/stack:output:0.Normal/sample/strided_slice_1/stack_1:output:0.Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maska
Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB c
 Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 
Normal/sample/BroadcastArgsBroadcastArgs)Normal/sample/BroadcastArgs/s0_1:output:0$Normal/sample/strided_slice:output:0*
_output_shapes
:
Normal/sample/BroadcastArgs_1BroadcastArgs Normal/sample/BroadcastArgs:r0:0&Normal/sample/strided_slice_1:output:0*
_output_shapes
:g
Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:[
Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ύ
Normal/sample/concatConcatV2&Normal/sample/concat/values_0:output:0"Normal/sample/BroadcastArgs_1:r0:0"Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:l
'Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¨
7Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat:output:0*
T0*"
_output_shapes
: *
dtype0Π
&Normal/sample/normal/random_normal/mulMul@Normal/sample/normal/random_normal/RandomStandardNormal:output:02Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
: Ά
"Normal/sample/normal/random_normalAddV2*Normal/sample/normal/random_normal/mul:z:00Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
: 
Normal/sample/mulMul&Normal/sample/normal/random_normal:z:0Normal/sample/add:z:0*
T0*"
_output_shapes
: u
Normal/sample/add_1AddV2Normal/sample/mul:z:0zeros_like:output:0*
T0*"
_output_shapes
: l
Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       
Normal/sample/ReshapeReshapeNormal/sample/add_1:z:0$Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

: ;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ο
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masku
+rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:t
)rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????o
)rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????
%rademacher/uniform/sanitize_seed/seedRandomUniformInt4rademacher/uniform/sanitize_seed/seed/shape:output:02rademacher/uniform/sanitize_seed/seed/min:output:02rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:q
/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R q
/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 RΗ
Hrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter.rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::q
/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :β
+rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2Shape:output:0Nrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Rrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:08rademacher/uniform/stateless_random_uniform/alg:output:08rademacher/uniform/stateless_random_uniform/min:output:08rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:????????? *
dtype0	R
rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R
rademacher/mulMulrademacher/mul/x:output:04rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:????????? R
rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rv
rademacher/subSubrademacher/mul:z:0rademacher/sub/y:output:0*
T0	*'
_output_shapes
:????????? l
rademacher/CastCastrademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:????????? R
ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value	B :P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsExpandDims/input:output:0ExpandDims/dim:output:0*
T0*
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2strided_slice:output:0ExpandDims:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:w
-rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:v
+rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????q
+rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????
'rademacher_1/uniform/sanitize_seed/seedRandomUniformInt6rademacher_1/uniform/sanitize_seed/seed/shape:output:04rademacher_1/uniform/sanitize_seed/seed/min:output:04rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:s
1rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R s
1rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 RΛ
Jrademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter0rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::s
1rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :ο
-rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2concat:output:0Prademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Trademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0:rademacher_1/uniform/stateless_random_uniform/alg:output:0:rademacher_1/uniform/stateless_random_uniform/min:output:0:rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????*
dtype0	T
rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R
rademacher_1/mulMulrademacher_1/mul/x:output:06rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????T
rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R|
rademacher_1/subSubrademacher_1/mul:z:0rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:?????????p
rademacher_1/CastCastrademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????Y
mulMulinputsrademacher/Cast:y:0*
T0*'
_output_shapes
:????????? k
MatMulMatMulmul:z:0Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????g
mul_1MulMatMul:product:0rademacher_1/Cast:y:0*
T0*'
_output_shapes
:?????????x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0m
MatMul_1MatMulinputsMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????]
addAddV2MatMul_1:product:0	mul_1:z:0*
T0*'
_output_shapes
:?????????
GIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 
\IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :·
mIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:
^IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ReadVariableOpReadVariableOpgindependentdeterministic_constructed_at_dense_tfp_2_sample_deterministic_sample_readvariableop_resource*
_output_shapes
:*
dtype0©
_IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:
UIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ­
cIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ―
eIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:―
eIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ι
]IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_sliceStridedSlicehIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/shape_as_tensor:output:0lIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_slice/stack:output:0nIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_slice/stack_1:output:0nIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask£
`IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ₯
bIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ί
]IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/BroadcastArgsBroadcastArgskIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0fIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:©
_IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:’
_IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 
[IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
VIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concatConcatV2hIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concat/values_0:output:0bIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/BroadcastArgs:r0:0hIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concat/values_2:output:0dIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:ά
[IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/BroadcastToBroadcastTofIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ReadVariableOp:value:0_IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

:?
]IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ω
WIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ReshapeReshapedIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/BroadcastTo:output:0fIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

:
HIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:§
BIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/ReshapeReshape`IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/Reshape:output:0QIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Reshape/shape:output:0*
T0*
_output_shapes
:
BiasAddBiasAddadd:z:0KIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????Π
ΰKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*
_output_shapes

: *
dtype0ς
ΡKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplusθKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

: 
ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4»
ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2ΧKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0ίKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes

: Μ
ΘKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLogΠKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes

: Γ
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1LogΝkullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054628*
T0*
_output_shapes
: 
ΘKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSubΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes

: ΅
ΣKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0―
ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDivΫKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0Νkullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054628*
T0*
_output_shapes

: ¦
ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDivΠkullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xΝkullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054628*
T0*
_output_shapes

: ½
ΦKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifferenceΠKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes

: 
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¬
ΘKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMulΣKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0ΪKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes

: 
ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @’
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1MulΥKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

: Ξ
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes

: 
ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?€
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2MulΥKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes

: 
ΘKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes

: 
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1SubΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

: Ϊ
KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"????ώ???
vKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSumΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * 'GΘ
truedivRealDivKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0truediv/y:output:0*
T0*
_output_shapes
: K
divergence_kernelIdentitytruediv:z:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????Z

Identity_1Identitydivergence_kernel:output:0^NoOp*
T0*
_output_shapes
: €
NoOpNoOp_^IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ReadVariableOpα^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpΤ^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp^MatMul_1/ReadVariableOp&^Normal/sample/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:????????? : : : : : 2ΐ
^IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ReadVariableOp^IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_2/sample/Deterministic/sample/ReadVariableOp2Ζ
ΰKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpΰKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2¬
ΣKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpΣKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2N
%Normal/sample/Softplus/ReadVariableOp%Normal/sample/Softplus/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :$ 

_output_shapes

: 

Ί
-__inference_sequential_layer_call_fn_70055072	
input
unknown:g 
	unknown_0:g 
	unknown_1: 
	unknown_2
	unknown_3
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7
	unknown_8
	unknown_9:	

unknown_10:	

unknown_11:	

unknown_12

unknown_13
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????	: : : *+
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_70054998o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????g: : : : :g : : : : : : : : : :	22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????g

_user_specified_nameinput:

_output_shapes
: :$ 

_output_shapes

:g :	

_output_shapes
: :$
 

_output_shapes

: :

_output_shapes
: :$ 

_output_shapes

:	
ό
γ
H__inference_sequential_layer_call_and_return_conditional_losses_70054828

inputs&
dense_tfp_1_70054506:g &
dense_tfp_1_70054508:g "
dense_tfp_1_70054510: 
dense_tfp_1_70054512
dense_tfp_1_70054514&
dense_tfp_2_70054659: &
dense_tfp_2_70054661: "
dense_tfp_2_70054663:
dense_tfp_2_70054665
dense_tfp_2_70054667!
output_70054812:	!
output_70054814:	
output_70054816:	
output_70054818
output_70054820
identity

identity_1

identity_2

identity_3’#dense_tfp_1/StatefulPartitionedCall’#dense_tfp_2/StatefulPartitionedCall’output/StatefulPartitionedCallΞ
#dense_tfp_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_tfp_1_70054506dense_tfp_1_70054508dense_tfp_1_70054510dense_tfp_1_70054512dense_tfp_1_70054514*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dense_tfp_1_layer_call_and_return_conditional_losses_70054505τ
#dense_tfp_2/StatefulPartitionedCallStatefulPartitionedCall,dense_tfp_1/StatefulPartitionedCall:output:0dense_tfp_2_70054659dense_tfp_2_70054661dense_tfp_2_70054663dense_tfp_2_70054665dense_tfp_2_70054667*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????: *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dense_tfp_2_layer_call_and_return_conditional_losses_70054658Ρ
output/StatefulPartitionedCallStatefulPartitionedCall,dense_tfp_2/StatefulPartitionedCall:output:0output_70054812output_70054814output_70054816output_70054818output_70054820*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????	: *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_70054811v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	l

Identity_1Identity,dense_tfp_1/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: l

Identity_2Identity,dense_tfp_2/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: g

Identity_3Identity'output/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: ³
NoOpNoOp$^dense_tfp_1/StatefulPartitionedCall$^dense_tfp_2/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????g: : : : :g : : : : : : : : : :	2J
#dense_tfp_1/StatefulPartitionedCall#dense_tfp_1/StatefulPartitionedCall2J
#dense_tfp_2/StatefulPartitionedCall#dense_tfp_2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:?????????g
 
_user_specified_nameinputs:

_output_shapes
: :$ 

_output_shapes

:g :	

_output_shapes
: :$
 

_output_shapes

: :

_output_shapes
: :$ 

_output_shapes

:	
δ
³
&__inference_signature_wrapper_70056105	
input
unknown:g 
	unknown_0:g 
	unknown_1: 
	unknown_2
	unknown_3
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7
	unknown_8
	unknown_9:	

unknown_10:	

unknown_11:	

unknown_12

unknown_13
identity’StatefulPartitionedCallα
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*+
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__wrapped_model_70054356o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????g: : : : :g : : : : : : : : : :	22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????g

_user_specified_nameinput:

_output_shapes
: :$ 

_output_shapes

:g :	

_output_shapes
: :$
 

_output_shapes

: :

_output_shapes
: :$ 

_output_shapes

:	
³
?
H__inference_sequential_layer_call_and_return_conditional_losses_70056068

inputsL
:dense_tfp_1_normal_sample_softplus_readvariableop_resource:g >
,dense_tfp_1_matmul_1_readvariableop_resource:g f
Xdense_tfp_1_independentdeterministic_sample_deterministic_sample_readvariableop_resource: ή
Ωdense_tfp_1_kullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70055766α
άdense_tfp_1_kullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xL
:dense_tfp_2_normal_sample_softplus_readvariableop_resource: >
,dense_tfp_2_matmul_1_readvariableop_resource: f
Xdense_tfp_2_independentdeterministic_sample_deterministic_sample_readvariableop_resource:ή
Ωdense_tfp_2_kullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70055901α
άdense_tfp_2_kullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xG
5output_normal_sample_softplus_readvariableop_resource:	9
'output_matmul_1_readvariableop_resource:	a
Soutput_independentdeterministic_sample_deterministic_sample_readvariableop_resource:	Τ
Οoutput_kullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70056036Χ
?output_kullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1

identity_2

identity_3’Odense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp’μdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp’ίdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp’#dense_tfp_1/MatMul_1/ReadVariableOp’1dense_tfp_1/Normal/sample/Softplus/ReadVariableOp’Odense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp’μdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp’ίdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp’#dense_tfp_2/MatMul_1/ReadVariableOp’1dense_tfp_2/Normal/sample/Softplus/ReadVariableOp’Joutput/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp’βoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp’Υoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp’output/MatMul_1/ReadVariableOp’,output/Normal/sample/Softplus/ReadVariableOpw
&dense_tfp_1/zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"g       a
dense_tfp_1/zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_tfp_1/zeros_likeFill/dense_tfp_1/zeros_like/shape_as_tensor:output:0%dense_tfp_1/zeros_like/Const:output:0*
T0*
_output_shapes

:g i
&dense_tfp_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ¬
1dense_tfp_1/Normal/sample/Softplus/ReadVariableOpReadVariableOp:dense_tfp_1_normal_sample_softplus_readvariableop_resource*
_output_shapes

:g *
dtype0
"dense_tfp_1/Normal/sample/SoftplusSoftplus9dense_tfp_1/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:g d
dense_tfp_1/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4«
dense_tfp_1/Normal/sample/addAddV2(dense_tfp_1/Normal/sample/add/x:output:00dense_tfp_1/Normal/sample/Softplus:activations:0*
T0*
_output_shapes

:g z
)dense_tfp_1/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"g       a
dense_tfp_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : w
-dense_tfp_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense_tfp_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense_tfp_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ϋ
'dense_tfp_1/Normal/sample/strided_sliceStridedSlice2dense_tfp_1/Normal/sample/shape_as_tensor:output:06dense_tfp_1/Normal/sample/strided_slice/stack:output:08dense_tfp_1/Normal/sample/strided_slice/stack_1:output:08dense_tfp_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
+dense_tfp_1/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"g       c
!dense_tfp_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : y
/dense_tfp_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_tfp_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_tfp_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ε
)dense_tfp_1/Normal/sample/strided_slice_1StridedSlice4dense_tfp_1/Normal/sample/shape_as_tensor_1:output:08dense_tfp_1/Normal/sample/strided_slice_1/stack:output:0:dense_tfp_1/Normal/sample/strided_slice_1/stack_1:output:0:dense_tfp_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
*dense_tfp_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB o
,dense_tfp_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ½
'dense_tfp_1/Normal/sample/BroadcastArgsBroadcastArgs5dense_tfp_1/Normal/sample/BroadcastArgs/s0_1:output:00dense_tfp_1/Normal/sample/strided_slice:output:0*
_output_shapes
:Έ
)dense_tfp_1/Normal/sample/BroadcastArgs_1BroadcastArgs,dense_tfp_1/Normal/sample/BroadcastArgs:r0:02dense_tfp_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:s
)dense_tfp_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
%dense_tfp_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ξ
 dense_tfp_1/Normal/sample/concatConcatV22dense_tfp_1/Normal/sample/concat/values_0:output:0.dense_tfp_1/Normal/sample/BroadcastArgs_1:r0:0.dense_tfp_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:x
3dense_tfp_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    z
5dense_tfp_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ΐ
Cdense_tfp_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal)dense_tfp_1/Normal/sample/concat:output:0*
T0*"
_output_shapes
:g *
dtype0τ
2dense_tfp_1/Normal/sample/normal/random_normal/mulMulLdense_tfp_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0>dense_tfp_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
:g Ϊ
.dense_tfp_1/Normal/sample/normal/random_normalAddV26dense_tfp_1/Normal/sample/normal/random_normal/mul:z:0<dense_tfp_1/Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
:g ¨
dense_tfp_1/Normal/sample/mulMul2dense_tfp_1/Normal/sample/normal/random_normal:z:0!dense_tfp_1/Normal/sample/add:z:0*
T0*"
_output_shapes
:g 
dense_tfp_1/Normal/sample/add_1AddV2!dense_tfp_1/Normal/sample/mul:z:0dense_tfp_1/zeros_like:output:0*
T0*"
_output_shapes
:g x
'dense_tfp_1/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"g       ¬
!dense_tfp_1/Normal/sample/ReshapeReshape#dense_tfp_1/Normal/sample/add_1:z:00dense_tfp_1/Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

:g G
dense_tfp_1/ShapeShapeinputs*
T0*
_output_shapes
:i
dense_tfp_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
!dense_tfp_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????k
!dense_tfp_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
dense_tfp_1/strided_sliceStridedSlicedense_tfp_1/Shape:output:0(dense_tfp_1/strided_slice/stack:output:0*dense_tfp_1/strided_slice/stack_1:output:0*dense_tfp_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
7dense_tfp_1/rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:
5dense_tfp_1/rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????{
5dense_tfp_1/rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????Έ
1dense_tfp_1/rademacher/uniform/sanitize_seed/seedRandomUniformInt@dense_tfp_1/rademacher/uniform/sanitize_seed/seed/shape:output:0>dense_tfp_1/rademacher/uniform/sanitize_seed/seed/min:output:0>dense_tfp_1/rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:}
;dense_tfp_1/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R }
;dense_tfp_1/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 Rί
Tdense_tfp_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter:dense_tfp_1/rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::}
;dense_tfp_1/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :Ά
7dense_tfp_1/rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2dense_tfp_1/Shape:output:0Zdense_tfp_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0^dense_tfp_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Ddense_tfp_1/rademacher/uniform/stateless_random_uniform/alg:output:0Ddense_tfp_1/rademacher/uniform/stateless_random_uniform/min:output:0Ddense_tfp_1/rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????g*
dtype0	^
dense_tfp_1/rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 RΌ
dense_tfp_1/rademacher/mulMul%dense_tfp_1/rademacher/mul/x:output:0@dense_tfp_1/rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????g^
dense_tfp_1/rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R
dense_tfp_1/rademacher/subSubdense_tfp_1/rademacher/mul:z:0%dense_tfp_1/rademacher/sub/y:output:0*
T0	*'
_output_shapes
:?????????g
dense_tfp_1/rademacher/CastCastdense_tfp_1/rademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????g^
dense_tfp_1/ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value	B : \
dense_tfp_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
dense_tfp_1/ExpandDims
ExpandDims%dense_tfp_1/ExpandDims/input:output:0#dense_tfp_1/ExpandDims/dim:output:0*
T0*
_output_shapes
:Y
dense_tfp_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ³
dense_tfp_1/concatConcatV2"dense_tfp_1/strided_slice:output:0dense_tfp_1/ExpandDims:output:0 dense_tfp_1/concat/axis:output:0*
N*
T0*
_output_shapes
:
9dense_tfp_1/rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:
7dense_tfp_1/rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????}
7dense_tfp_1/rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????ΐ
3dense_tfp_1/rademacher_1/uniform/sanitize_seed/seedRandomUniformIntBdense_tfp_1/rademacher_1/uniform/sanitize_seed/seed/shape:output:0@dense_tfp_1/rademacher_1/uniform/sanitize_seed/seed/min:output:0@dense_tfp_1/rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:
=dense_tfp_1/rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R 
=dense_tfp_1/rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 Rγ
Vdense_tfp_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter<dense_tfp_1/rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::
=dense_tfp_1/rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :Γ
9dense_tfp_1/rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2dense_tfp_1/concat:output:0\dense_tfp_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`dense_tfp_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Fdense_tfp_1/rademacher_1/uniform/stateless_random_uniform/alg:output:0Fdense_tfp_1/rademacher_1/uniform/stateless_random_uniform/min:output:0Fdense_tfp_1/rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:????????? *
dtype0	`
dense_tfp_1/rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 RΒ
dense_tfp_1/rademacher_1/mulMul'dense_tfp_1/rademacher_1/mul/x:output:0Bdense_tfp_1/rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:????????? `
dense_tfp_1/rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
dense_tfp_1/rademacher_1/subSub dense_tfp_1/rademacher_1/mul:z:0'dense_tfp_1/rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:????????? 
dense_tfp_1/rademacher_1/CastCast dense_tfp_1/rademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:????????? q
dense_tfp_1/mulMulinputsdense_tfp_1/rademacher/Cast:y:0*
T0*'
_output_shapes
:?????????g
dense_tfp_1/MatMulMatMuldense_tfp_1/mul:z:0*dense_tfp_1/Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:????????? 
dense_tfp_1/mul_1Muldense_tfp_1/MatMul:product:0!dense_tfp_1/rademacher_1/Cast:y:0*
T0*'
_output_shapes
:????????? 
#dense_tfp_1/MatMul_1/ReadVariableOpReadVariableOp,dense_tfp_1_matmul_1_readvariableop_resource*
_output_shapes

:g *
dtype0
dense_tfp_1/MatMul_1MatMulinputs+dense_tfp_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 
dense_tfp_1/addAddV2dense_tfp_1/MatMul_1:product:0dense_tfp_1/mul_1:z:0*
T0*'
_output_shapes
:????????? {
8dense_tfp_1/IndependentDeterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 
Mdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :¨
^dense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:δ
Odense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpReadVariableOpXdense_tfp_1_independentdeterministic_sample_deterministic_sample_readvariableop_resource*
_output_shapes
: *
dtype0
Pdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: 
Fdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Tdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:  
Vdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Vdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ndense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/strided_sliceStridedSliceYdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensor:output:0]dense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack:output:0_dense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1:output:0_dense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Qdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 
Sdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ²
Ndense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgsBroadcastArgs\dense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0Wdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:
Pdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Pdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 
Ldense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : γ
Gdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/concatConcatV2Ydense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_0:output:0Sdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs:r0:0Ydense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_2:output:0Udense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:―
Ldense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastToBroadcastToWdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp:value:0Pdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

: 
Ndense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       ¬
Hdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/ReshapeReshapeUdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastTo:output:0Wdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

: 
9dense_tfp_1/IndependentDeterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: ϊ
3dense_tfp_1/IndependentDeterministic/sample/ReshapeReshapeQdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/Reshape:output:0Bdense_tfp_1/IndependentDeterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes
: £
dense_tfp_1/BiasAddBiasAdddense_tfp_1/add:z:0<dense_tfp_1/IndependentDeterministic/sample/Reshape:output:0*
T0*'
_output_shapes
:????????? h
dense_tfp_1/ReluReludense_tfp_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? θ
μdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp:dense_tfp_1_normal_sample_softplus_readvariableop_resource*
_output_shapes

:g *
dtype0
έdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplusτdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:g  
Ϊdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4ί
Ψdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2γdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0λdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes

:g δ
Τdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLogάdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes

:g Ϋ
Φdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1LogΩdense_tfp_1_kullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70055766*
T0*
_output_shapes
: ½
Τdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSubΨdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0Ϊdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes

:g Ν
ίdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp,dense_tfp_1_matmul_1_readvariableop_resource*
_output_shapes

:g *
dtype0Σ
Ψdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDivηdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0Ωdense_tfp_1_kullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70055766*
T0*
_output_shapes

:g Κ
Ϊdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDivάdense_tfp_1_kullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xΩdense_tfp_1_kullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70055766*
T0*
_output_shapes

:g α
βdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifferenceάdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0ήdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes

:g 
Φdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Π
Τdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMulίdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0ζdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes

:g 
Ψdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ζ
Φdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mulαdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0Ψdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

:g ζ
Φdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1Ϊdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes

:g 
Ψdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Θ
Φdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mulαdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0Ϊdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes

:g Ώ
Τdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2Ψdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0Ϊdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes

:g ½
Φdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1SubΨdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0Ψdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

:g ζ
dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"????ώ???¨
dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSumΪdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: Z
dense_tfp_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * 'Gν
dense_tfp_1/truedivRealDivdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0dense_tfp_1/truediv/y:output:0*
T0*
_output_shapes
: c
dense_tfp_1/divergence_kernelIdentitydense_tfp_1/truediv:z:0*
T0*
_output_shapes
: k
dense_tfp_2/zeros_likeConst*
_output_shapes

: *
dtype0*
valueB *    i
&dense_tfp_2/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ¬
1dense_tfp_2/Normal/sample/Softplus/ReadVariableOpReadVariableOp:dense_tfp_2_normal_sample_softplus_readvariableop_resource*
_output_shapes

: *
dtype0
"dense_tfp_2/Normal/sample/SoftplusSoftplus9dense_tfp_2/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

: d
dense_tfp_2/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4«
dense_tfp_2/Normal/sample/addAddV2(dense_tfp_2/Normal/sample/add/x:output:00dense_tfp_2/Normal/sample/Softplus:activations:0*
T0*
_output_shapes

: z
)dense_tfp_2/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"       a
dense_tfp_2/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : w
-dense_tfp_2/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense_tfp_2/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense_tfp_2/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ϋ
'dense_tfp_2/Normal/sample/strided_sliceStridedSlice2dense_tfp_2/Normal/sample/shape_as_tensor:output:06dense_tfp_2/Normal/sample/strided_slice/stack:output:08dense_tfp_2/Normal/sample/strided_slice/stack_1:output:08dense_tfp_2/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
+dense_tfp_2/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"       c
!dense_tfp_2/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : y
/dense_tfp_2/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_tfp_2/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_tfp_2/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ε
)dense_tfp_2/Normal/sample/strided_slice_1StridedSlice4dense_tfp_2/Normal/sample/shape_as_tensor_1:output:08dense_tfp_2/Normal/sample/strided_slice_1/stack:output:0:dense_tfp_2/Normal/sample/strided_slice_1/stack_1:output:0:dense_tfp_2/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
*dense_tfp_2/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB o
,dense_tfp_2/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ½
'dense_tfp_2/Normal/sample/BroadcastArgsBroadcastArgs5dense_tfp_2/Normal/sample/BroadcastArgs/s0_1:output:00dense_tfp_2/Normal/sample/strided_slice:output:0*
_output_shapes
:Έ
)dense_tfp_2/Normal/sample/BroadcastArgs_1BroadcastArgs,dense_tfp_2/Normal/sample/BroadcastArgs:r0:02dense_tfp_2/Normal/sample/strided_slice_1:output:0*
_output_shapes
:s
)dense_tfp_2/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
%dense_tfp_2/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ξ
 dense_tfp_2/Normal/sample/concatConcatV22dense_tfp_2/Normal/sample/concat/values_0:output:0.dense_tfp_2/Normal/sample/BroadcastArgs_1:r0:0.dense_tfp_2/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:x
3dense_tfp_2/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    z
5dense_tfp_2/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ΐ
Cdense_tfp_2/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal)dense_tfp_2/Normal/sample/concat:output:0*
T0*"
_output_shapes
: *
dtype0τ
2dense_tfp_2/Normal/sample/normal/random_normal/mulMulLdense_tfp_2/Normal/sample/normal/random_normal/RandomStandardNormal:output:0>dense_tfp_2/Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
: Ϊ
.dense_tfp_2/Normal/sample/normal/random_normalAddV26dense_tfp_2/Normal/sample/normal/random_normal/mul:z:0<dense_tfp_2/Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
: ¨
dense_tfp_2/Normal/sample/mulMul2dense_tfp_2/Normal/sample/normal/random_normal:z:0!dense_tfp_2/Normal/sample/add:z:0*
T0*"
_output_shapes
: 
dense_tfp_2/Normal/sample/add_1AddV2!dense_tfp_2/Normal/sample/mul:z:0dense_tfp_2/zeros_like:output:0*
T0*"
_output_shapes
: x
'dense_tfp_2/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       ¬
!dense_tfp_2/Normal/sample/ReshapeReshape#dense_tfp_2/Normal/sample/add_1:z:00dense_tfp_2/Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

: _
dense_tfp_2/ShapeShapedense_tfp_1/Relu:activations:0*
T0*
_output_shapes
:i
dense_tfp_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
!dense_tfp_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????k
!dense_tfp_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
dense_tfp_2/strided_sliceStridedSlicedense_tfp_2/Shape:output:0(dense_tfp_2/strided_slice/stack:output:0*dense_tfp_2/strided_slice/stack_1:output:0*dense_tfp_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
7dense_tfp_2/rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:
5dense_tfp_2/rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????{
5dense_tfp_2/rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????Έ
1dense_tfp_2/rademacher/uniform/sanitize_seed/seedRandomUniformInt@dense_tfp_2/rademacher/uniform/sanitize_seed/seed/shape:output:0>dense_tfp_2/rademacher/uniform/sanitize_seed/seed/min:output:0>dense_tfp_2/rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:}
;dense_tfp_2/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R }
;dense_tfp_2/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 Rί
Tdense_tfp_2/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter:dense_tfp_2/rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::}
;dense_tfp_2/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :Ά
7dense_tfp_2/rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2dense_tfp_2/Shape:output:0Zdense_tfp_2/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0^dense_tfp_2/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Ddense_tfp_2/rademacher/uniform/stateless_random_uniform/alg:output:0Ddense_tfp_2/rademacher/uniform/stateless_random_uniform/min:output:0Ddense_tfp_2/rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:????????? *
dtype0	^
dense_tfp_2/rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 RΌ
dense_tfp_2/rademacher/mulMul%dense_tfp_2/rademacher/mul/x:output:0@dense_tfp_2/rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:????????? ^
dense_tfp_2/rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R
dense_tfp_2/rademacher/subSubdense_tfp_2/rademacher/mul:z:0%dense_tfp_2/rademacher/sub/y:output:0*
T0	*'
_output_shapes
:????????? 
dense_tfp_2/rademacher/CastCastdense_tfp_2/rademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:????????? ^
dense_tfp_2/ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value	B :\
dense_tfp_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
dense_tfp_2/ExpandDims
ExpandDims%dense_tfp_2/ExpandDims/input:output:0#dense_tfp_2/ExpandDims/dim:output:0*
T0*
_output_shapes
:Y
dense_tfp_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ³
dense_tfp_2/concatConcatV2"dense_tfp_2/strided_slice:output:0dense_tfp_2/ExpandDims:output:0 dense_tfp_2/concat/axis:output:0*
N*
T0*
_output_shapes
:
9dense_tfp_2/rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:
7dense_tfp_2/rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????}
7dense_tfp_2/rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????ΐ
3dense_tfp_2/rademacher_1/uniform/sanitize_seed/seedRandomUniformIntBdense_tfp_2/rademacher_1/uniform/sanitize_seed/seed/shape:output:0@dense_tfp_2/rademacher_1/uniform/sanitize_seed/seed/min:output:0@dense_tfp_2/rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:
=dense_tfp_2/rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R 
=dense_tfp_2/rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 Rγ
Vdense_tfp_2/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter<dense_tfp_2/rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::
=dense_tfp_2/rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :Γ
9dense_tfp_2/rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2dense_tfp_2/concat:output:0\dense_tfp_2/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`dense_tfp_2/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Fdense_tfp_2/rademacher_1/uniform/stateless_random_uniform/alg:output:0Fdense_tfp_2/rademacher_1/uniform/stateless_random_uniform/min:output:0Fdense_tfp_2/rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????*
dtype0	`
dense_tfp_2/rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 RΒ
dense_tfp_2/rademacher_1/mulMul'dense_tfp_2/rademacher_1/mul/x:output:0Bdense_tfp_2/rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????`
dense_tfp_2/rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
dense_tfp_2/rademacher_1/subSub dense_tfp_2/rademacher_1/mul:z:0'dense_tfp_2/rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:?????????
dense_tfp_2/rademacher_1/CastCast dense_tfp_2/rademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????
dense_tfp_2/mulMuldense_tfp_1/Relu:activations:0dense_tfp_2/rademacher/Cast:y:0*
T0*'
_output_shapes
:????????? 
dense_tfp_2/MatMulMatMuldense_tfp_2/mul:z:0*dense_tfp_2/Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????
dense_tfp_2/mul_1Muldense_tfp_2/MatMul:product:0!dense_tfp_2/rademacher_1/Cast:y:0*
T0*'
_output_shapes
:?????????
#dense_tfp_2/MatMul_1/ReadVariableOpReadVariableOp,dense_tfp_2_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0
dense_tfp_2/MatMul_1MatMuldense_tfp_1/Relu:activations:0+dense_tfp_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_tfp_2/addAddV2dense_tfp_2/MatMul_1:product:0dense_tfp_2/mul_1:z:0*
T0*'
_output_shapes
:?????????{
8dense_tfp_2/IndependentDeterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 
Mdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :¨
^dense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:δ
Odense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpReadVariableOpXdense_tfp_2_independentdeterministic_sample_deterministic_sample_readvariableop_resource*
_output_shapes
:*
dtype0
Pdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:
Fdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Tdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:  
Vdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Vdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ndense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/strided_sliceStridedSliceYdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensor:output:0]dense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack:output:0_dense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1:output:0_dense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Qdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 
Sdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ²
Ndense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgsBroadcastArgs\dense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0Wdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:
Pdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Pdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 
Ldense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : γ
Gdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/concatConcatV2Ydense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/concat/values_0:output:0Sdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs:r0:0Ydense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/concat/values_2:output:0Udense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:―
Ldense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/BroadcastToBroadcastToWdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp:value:0Pdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

:
Ndense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ¬
Hdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/ReshapeReshapeUdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/BroadcastTo:output:0Wdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

:
9dense_tfp_2/IndependentDeterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:ϊ
3dense_tfp_2/IndependentDeterministic/sample/ReshapeReshapeQdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/Reshape:output:0Bdense_tfp_2/IndependentDeterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes
:£
dense_tfp_2/BiasAddBiasAdddense_tfp_2/add:z:0<dense_tfp_2/IndependentDeterministic/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????h
dense_tfp_2/ReluReludense_tfp_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????θ
μdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp:dense_tfp_2_normal_sample_softplus_readvariableop_resource*
_output_shapes

: *
dtype0
έdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplusτdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:  
Ϊdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4ί
Ψdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2γdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0λdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes

: δ
Τdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLogάdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes

: Ϋ
Φdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1LogΩdense_tfp_2_kullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70055901*
T0*
_output_shapes
: ½
Τdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSubΨdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0Ϊdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes

: Ν
ίdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp,dense_tfp_2_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0Σ
Ψdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDivηdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0Ωdense_tfp_2_kullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70055901*
T0*
_output_shapes

: Κ
Ϊdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDivάdense_tfp_2_kullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xΩdense_tfp_2_kullbackleibler_independentnormal_constructed_at_dense_tfp_2_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70055901*
T0*
_output_shapes

: α
βdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifferenceάdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0ήdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes

: 
Φdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Π
Τdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMulίdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0ζdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes

: 
Ψdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ζ
Φdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mulαdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0Ψdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

: ζ
Φdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1Ϊdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes

: 
Ψdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Θ
Φdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mulαdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0Ϊdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes

: Ώ
Τdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2Ψdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0Ϊdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes

: ½
Φdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1SubΨdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0Ψdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

: ζ
dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"????ώ???¨
dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSumΪdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: Z
dense_tfp_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * 'Gν
dense_tfp_2/truedivRealDivdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0dense_tfp_2/truediv/y:output:0*
T0*
_output_shapes
: c
dense_tfp_2/divergence_kernelIdentitydense_tfp_2/truediv:z:0*
T0*
_output_shapes
: f
output/zeros_likeConst*
_output_shapes

:	*
dtype0*
valueB	*    d
!output/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ’
,output/Normal/sample/Softplus/ReadVariableOpReadVariableOp5output_normal_sample_softplus_readvariableop_resource*
_output_shapes

:	*
dtype0
output/Normal/sample/SoftplusSoftplus4output/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:	_
output/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4
output/Normal/sample/addAddV2#output/Normal/sample/add/x:output:0+output/Normal/sample/Softplus:activations:0*
T0*
_output_shapes

:	u
$output/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"   	   \
output/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(output/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*output/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*output/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Β
"output/Normal/sample/strided_sliceStridedSlice-output/Normal/sample/shape_as_tensor:output:01output/Normal/sample/strided_slice/stack:output:03output/Normal/sample/strided_slice/stack_1:output:03output/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskw
&output/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"   	   ^
output/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : t
*output/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,output/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,output/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Μ
$output/Normal/sample/strided_slice_1StridedSlice/output/Normal/sample/shape_as_tensor_1:output:03output/Normal/sample/strided_slice_1/stack:output:05output/Normal/sample/strided_slice_1/stack_1:output:05output/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%output/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'output/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
"output/Normal/sample/BroadcastArgsBroadcastArgs0output/Normal/sample/BroadcastArgs/s0_1:output:0+output/Normal/sample/strided_slice:output:0*
_output_shapes
:©
$output/Normal/sample/BroadcastArgs_1BroadcastArgs'output/Normal/sample/BroadcastArgs:r0:0-output/Normal/sample/strided_slice_1:output:0*
_output_shapes
:n
$output/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:b
 output/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ϊ
output/Normal/sample/concatConcatV2-output/Normal/sample/concat/values_0:output:0)output/Normal/sample/BroadcastArgs_1:r0:0)output/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:s
.output/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    u
0output/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ά
>output/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal$output/Normal/sample/concat:output:0*
T0*"
_output_shapes
:	*
dtype0ε
-output/Normal/sample/normal/random_normal/mulMulGoutput/Normal/sample/normal/random_normal/RandomStandardNormal:output:09output/Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
:	Λ
)output/Normal/sample/normal/random_normalAddV21output/Normal/sample/normal/random_normal/mul:z:07output/Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
:	
output/Normal/sample/mulMul-output/Normal/sample/normal/random_normal:z:0output/Normal/sample/add:z:0*
T0*"
_output_shapes
:	
output/Normal/sample/add_1AddV2output/Normal/sample/mul:z:0output/zeros_like:output:0*
T0*"
_output_shapes
:	s
"output/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   	   
output/Normal/sample/ReshapeReshapeoutput/Normal/sample/add_1:z:0+output/Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

:	Z
output/ShapeShapedense_tfp_2/Relu:activations:0*
T0*
_output_shapes
:d
output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ς
output/strided_sliceStridedSliceoutput/Shape:output:0#output/strided_slice/stack:output:0%output/strided_slice/stack_1:output:0%output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
2output/rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:{
0output/rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????v
0output/rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????€
,output/rademacher/uniform/sanitize_seed/seedRandomUniformInt;output/rademacher/uniform/sanitize_seed/seed/shape:output:09output/rademacher/uniform/sanitize_seed/seed/min:output:09output/rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:x
6output/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R x
6output/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 RΥ
Ooutput/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter5output/rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::x
6output/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :
2output/rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2output/Shape:output:0Uoutput/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Youtput/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0?output/rademacher/uniform/stateless_random_uniform/alg:output:0?output/rademacher/uniform/stateless_random_uniform/min:output:0?output/rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????*
dtype0	Y
output/rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R­
output/rademacher/mulMul output/rademacher/mul/x:output:0;output/rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????Y
output/rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R
output/rademacher/subSuboutput/rademacher/mul:z:0 output/rademacher/sub/y:output:0*
T0	*'
_output_shapes
:?????????z
output/rademacher/CastCastoutput/rademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????Y
output/ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value	B :	W
output/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
output/ExpandDims
ExpandDims output/ExpandDims/input:output:0output/ExpandDims/dim:output:0*
T0*
_output_shapes
:T
output/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
output/concatConcatV2output/strided_slice:output:0output/ExpandDims:output:0output/concat/axis:output:0*
N*
T0*
_output_shapes
:~
4output/rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:}
2output/rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????x
2output/rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????¬
.output/rademacher_1/uniform/sanitize_seed/seedRandomUniformInt=output/rademacher_1/uniform/sanitize_seed/seed/shape:output:0;output/rademacher_1/uniform/sanitize_seed/seed/min:output:0;output/rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:z
8output/rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R z
8output/rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 RΩ
Qoutput/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter7output/rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::z
8output/rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B : 
4output/rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2output/concat:output:0Woutput/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0[output/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Aoutput/rademacher_1/uniform/stateless_random_uniform/alg:output:0Aoutput/rademacher_1/uniform/stateless_random_uniform/min:output:0Aoutput/rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????	*
dtype0	[
output/rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R³
output/rademacher_1/mulMul"output/rademacher_1/mul/x:output:0=output/rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????	[
output/rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R
output/rademacher_1/subSuboutput/rademacher_1/mul:z:0"output/rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:?????????	~
output/rademacher_1/CastCastoutput/rademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????	

output/mulMuldense_tfp_2/Relu:activations:0output/rademacher/Cast:y:0*
T0*'
_output_shapes
:?????????
output/MatMulMatMuloutput/mul:z:0%output/Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????	|
output/mul_1Muloutput/MatMul:product:0output/rademacher_1/Cast:y:0*
T0*'
_output_shapes
:?????????	
output/MatMul_1/ReadVariableOpReadVariableOp'output_matmul_1_readvariableop_resource*
_output_shapes

:	*
dtype0
output/MatMul_1MatMuldense_tfp_2/Relu:activations:0&output/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	r

output/addAddV2output/MatMul_1:product:0output/mul_1:z:0*
T0*'
_output_shapes
:?????????	v
3output/IndependentDeterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 
Houtput/IndependentDeterministic/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :£
Youtput/IndependentDeterministic/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:Ϊ
Joutput/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpReadVariableOpSoutput_independentdeterministic_sample_deterministic_sample_readvariableop_resource*
_output_shapes
:	*
dtype0
Koutput/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:	
Aoutput/IndependentDeterministic/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Ooutput/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Qoutput/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Qoutput/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ioutput/IndependentDeterministic/sample/Deterministic/sample/strided_sliceStridedSliceToutput/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensor:output:0Xoutput/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack:output:0Zoutput/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1:output:0Zoutput/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Loutput/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 
Noutput/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB £
Ioutput/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgsBroadcastArgsWoutput/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0Routput/IndependentDeterministic/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:
Koutput/IndependentDeterministic/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Koutput/IndependentDeterministic/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 
Goutput/IndependentDeterministic/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Κ
Boutput/IndependentDeterministic/sample/Deterministic/sample/concatConcatV2Toutput/IndependentDeterministic/sample/Deterministic/sample/concat/values_0:output:0Noutput/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs:r0:0Toutput/IndependentDeterministic/sample/Deterministic/sample/concat/values_2:output:0Poutput/IndependentDeterministic/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
: 
Goutput/IndependentDeterministic/sample/Deterministic/sample/BroadcastToBroadcastToRoutput/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp:value:0Koutput/IndependentDeterministic/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

:	
Ioutput/IndependentDeterministic/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   	   
Coutput/IndependentDeterministic/sample/Deterministic/sample/ReshapeReshapePoutput/IndependentDeterministic/sample/Deterministic/sample/BroadcastTo:output:0Routput/IndependentDeterministic/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

:	~
4output/IndependentDeterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:	λ
.output/IndependentDeterministic/sample/ReshapeReshapeLoutput/IndependentDeterministic/sample/Deterministic/sample/Reshape:output:0=output/IndependentDeterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes
:	
output/BiasAddBiasAddoutput/add:z:07output/IndependentDeterministic/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????	d
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	Ω
βoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp5output_normal_sample_softplus_readvariableop_resource*
_output_shapes

:	*
dtype0φ
Σoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplusκoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:	
Πoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4Α
Ξoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2Ωoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0αoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes

:	Π
Κoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog?output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes

:	Η
Μoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1LogΟoutput_kullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70056036*
T0*
_output_shapes
: 
Κoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSubΞoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0Πoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes

:	Ύ
Υoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp'output_matmul_1_readvariableop_resource*
_output_shapes

:	*
dtype0΅
Ξoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDivέoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0Οoutput_kullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70056036*
T0*
_output_shapes

:	¬
Πoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv?output_kullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xΟoutput_kullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70056036*
T0*
_output_shapes

:	Γ
Ψoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference?output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0Τoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes

:	
Μoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?²
Κoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMulΥoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0άoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes

:	
Ξoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @¨
Μoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1MulΧoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0Ξoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

:	?
Μoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1Πoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes

:	
Ξoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ͺ
Μoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2MulΧoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0Πoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes

:	‘
Κoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2Ξoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0Πoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes

:	
Μoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1SubΞoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0Ξoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

:	ά
output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"????ώ???
xoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSumΠoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: U
output/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * 'GΩ
output/truedivRealDivoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0output/truediv/y:output:0*
T0*
_output_shapes
: Y
output/divergence_kernelIdentityoutput/truediv:z:0*
T0*
_output_shapes
: g
IdentityIdentityoutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????	f

Identity_1Identity&dense_tfp_1/divergence_kernel:output:0^NoOp*
T0*
_output_shapes
: f

Identity_2Identity&dense_tfp_2/divergence_kernel:output:0^NoOp*
T0*
_output_shapes
: a

Identity_3Identity!output/divergence_kernel:output:0^NoOp*
T0*
_output_shapes
:  
NoOpNoOpP^dense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpν^dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpΰ^dense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp$^dense_tfp_1/MatMul_1/ReadVariableOp2^dense_tfp_1/Normal/sample/Softplus/ReadVariableOpP^dense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpν^dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpΰ^dense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp$^dense_tfp_2/MatMul_1/ReadVariableOp2^dense_tfp_2/Normal/sample/Softplus/ReadVariableOpK^output/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpγ^output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpΦ^output/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp^output/MatMul_1/ReadVariableOp-^output/Normal/sample/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????g: : : : :g : : : : : : : : : :	2’
Odense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpOdense_tfp_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp2ή
μdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpμdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2Δ
ίdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpίdense_tfp_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2J
#dense_tfp_1/MatMul_1/ReadVariableOp#dense_tfp_1/MatMul_1/ReadVariableOp2f
1dense_tfp_1/Normal/sample/Softplus/ReadVariableOp1dense_tfp_1/Normal/sample/Softplus/ReadVariableOp2’
Odense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpOdense_tfp_2/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp2ή
μdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpμdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2Δ
ίdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpίdense_tfp_2/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_2/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2J
#dense_tfp_2/MatMul_1/ReadVariableOp#dense_tfp_2/MatMul_1/ReadVariableOp2f
1dense_tfp_2/Normal/sample/Softplus/ReadVariableOp1dense_tfp_2/Normal/sample/Softplus/ReadVariableOp2
Joutput/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpJoutput/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp2Κ
βoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpβoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2°
Υoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpΥoutput/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2@
output/MatMul_1/ReadVariableOpoutput/MatMul_1/ReadVariableOp2\
,output/Normal/sample/Softplus/ReadVariableOp,output/Normal/sample/Softplus/ReadVariableOp:O K
'
_output_shapes
:?????????g
 
_user_specified_nameinputs:

_output_shapes
: :$ 

_output_shapes

:g :	

_output_shapes
: :$
 

_output_shapes

: :

_output_shapes
: :$ 

_output_shapes

:	
ω
β
H__inference_sequential_layer_call_and_return_conditional_losses_70055158	
input&
dense_tfp_1_70055118:g &
dense_tfp_1_70055120:g "
dense_tfp_1_70055122: 
dense_tfp_1_70055124
dense_tfp_1_70055126&
dense_tfp_2_70055130: &
dense_tfp_2_70055132: "
dense_tfp_2_70055134:
dense_tfp_2_70055136
dense_tfp_2_70055138!
output_70055142:	!
output_70055144:	
output_70055146:	
output_70055148
output_70055150
identity

identity_1

identity_2

identity_3’#dense_tfp_1/StatefulPartitionedCall’#dense_tfp_2/StatefulPartitionedCall’output/StatefulPartitionedCallΝ
#dense_tfp_1/StatefulPartitionedCallStatefulPartitionedCallinputdense_tfp_1_70055118dense_tfp_1_70055120dense_tfp_1_70055122dense_tfp_1_70055124dense_tfp_1_70055126*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dense_tfp_1_layer_call_and_return_conditional_losses_70054505τ
#dense_tfp_2/StatefulPartitionedCallStatefulPartitionedCall,dense_tfp_1/StatefulPartitionedCall:output:0dense_tfp_2_70055130dense_tfp_2_70055132dense_tfp_2_70055134dense_tfp_2_70055136dense_tfp_2_70055138*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????: *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dense_tfp_2_layer_call_and_return_conditional_losses_70054658Ρ
output/StatefulPartitionedCallStatefulPartitionedCall,dense_tfp_2/StatefulPartitionedCall:output:0output_70055142output_70055144output_70055146output_70055148output_70055150*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????	: *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_70054811v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	l

Identity_1Identity,dense_tfp_1/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: l

Identity_2Identity,dense_tfp_2/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: g

Identity_3Identity'output/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: ³
NoOpNoOp$^dense_tfp_1/StatefulPartitionedCall$^dense_tfp_2/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????g: : : : :g : : : : : : : : : :	2J
#dense_tfp_1/StatefulPartitionedCall#dense_tfp_1/StatefulPartitionedCall2J
#dense_tfp_2/StatefulPartitionedCall#dense_tfp_2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:N J
'
_output_shapes
:?????????g

_user_specified_nameinput:

_output_shapes
: :$ 

_output_shapes

:g :	

_output_shapes
: :$
 

_output_shapes

: :

_output_shapes
: :$ 

_output_shapes

:	
ης
Ν

D__inference_output_layer_call_and_return_conditional_losses_70054811

inputs@
.normal_sample_softplus_readvariableop_resource:	2
 matmul_1_readvariableop_resource:	p
bindependentdeterministic_constructed_at_output_sample_deterministic_sample_readvariableop_resource:	Ν
Θkullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054781Π
Λkullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1’YIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ReadVariableOp’ΫKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp’ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp’MatMul_1/ReadVariableOp’%Normal/sample/Softplus/ReadVariableOp_

zeros_likeConst*
_output_shapes

:	*
dtype0*
valueB	*    ]
Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 
%Normal/sample/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*
_output_shapes

:	*
dtype0z
Normal/sample/SoftplusSoftplus-Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:	X
Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4
Normal/sample/addAddV2Normal/sample/add/x:output:0$Normal/sample/Softplus:activations:0*
T0*
_output_shapes

:	n
Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"   	   U
Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Normal/sample/strided_sliceStridedSlice&Normal/sample/shape_as_tensor:output:0*Normal/sample/strided_slice/stack:output:0,Normal/sample/strided_slice/stack_1:output:0,Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"   	   W
Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : m
#Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:©
Normal/sample/strided_slice_1StridedSlice(Normal/sample/shape_as_tensor_1:output:0,Normal/sample/strided_slice_1/stack:output:0.Normal/sample/strided_slice_1/stack_1:output:0.Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maska
Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB c
 Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 
Normal/sample/BroadcastArgsBroadcastArgs)Normal/sample/BroadcastArgs/s0_1:output:0$Normal/sample/strided_slice:output:0*
_output_shapes
:
Normal/sample/BroadcastArgs_1BroadcastArgs Normal/sample/BroadcastArgs:r0:0&Normal/sample/strided_slice_1:output:0*
_output_shapes
:g
Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:[
Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ύ
Normal/sample/concatConcatV2&Normal/sample/concat/values_0:output:0"Normal/sample/BroadcastArgs_1:r0:0"Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:l
'Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¨
7Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat:output:0*
T0*"
_output_shapes
:	*
dtype0Π
&Normal/sample/normal/random_normal/mulMul@Normal/sample/normal/random_normal/RandomStandardNormal:output:02Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
:	Ά
"Normal/sample/normal/random_normalAddV2*Normal/sample/normal/random_normal/mul:z:00Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
:	
Normal/sample/mulMul&Normal/sample/normal/random_normal:z:0Normal/sample/add:z:0*
T0*"
_output_shapes
:	u
Normal/sample/add_1AddV2Normal/sample/mul:z:0zeros_like:output:0*
T0*"
_output_shapes
:	l
Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   	   
Normal/sample/ReshapeReshapeNormal/sample/add_1:z:0$Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

:	;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ο
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masku
+rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:t
)rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????o
)rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????
%rademacher/uniform/sanitize_seed/seedRandomUniformInt4rademacher/uniform/sanitize_seed/seed/shape:output:02rademacher/uniform/sanitize_seed/seed/min:output:02rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:q
/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R q
/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 RΗ
Hrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter.rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::q
/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :β
+rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2Shape:output:0Nrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Rrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:08rademacher/uniform/stateless_random_uniform/alg:output:08rademacher/uniform/stateless_random_uniform/min:output:08rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????*
dtype0	R
rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R
rademacher/mulMulrademacher/mul/x:output:04rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????R
rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rv
rademacher/subSubrademacher/mul:z:0rademacher/sub/y:output:0*
T0	*'
_output_shapes
:?????????l
rademacher/CastCastrademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????R
ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value	B :	P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsExpandDims/input:output:0ExpandDims/dim:output:0*
T0*
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2strided_slice:output:0ExpandDims:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:w
-rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:v
+rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????q
+rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????
'rademacher_1/uniform/sanitize_seed/seedRandomUniformInt6rademacher_1/uniform/sanitize_seed/seed/shape:output:04rademacher_1/uniform/sanitize_seed/seed/min:output:04rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:s
1rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R s
1rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 RΛ
Jrademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter0rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::s
1rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :ο
-rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2concat:output:0Prademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Trademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0:rademacher_1/uniform/stateless_random_uniform/alg:output:0:rademacher_1/uniform/stateless_random_uniform/min:output:0:rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????	*
dtype0	T
rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R
rademacher_1/mulMulrademacher_1/mul/x:output:06rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????	T
rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R|
rademacher_1/subSubrademacher_1/mul:z:0rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:?????????	p
rademacher_1/CastCastrademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????	Y
mulMulinputsrademacher/Cast:y:0*
T0*'
_output_shapes
:?????????k
MatMulMatMulmul:z:0Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????	g
mul_1MulMatMul:product:0rademacher_1/Cast:y:0*
T0*'
_output_shapes
:?????????	x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:	*
dtype0m
MatMul_1MatMulinputsMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	]
addAddV2MatMul_1:product:0	mul_1:z:0*
T0*'
_output_shapes
:?????????	
BIndependentDeterministic_CONSTRUCTED_AT_output/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 
WIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :²
hIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:ψ
YIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ReadVariableOpReadVariableOpbindependentdeterministic_constructed_at_output_sample_deterministic_sample_readvariableop_resource*
_output_shapes
:	*
dtype0€
ZIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:	
PIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ¨
^IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ͺ
`IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ͺ
`IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Π
XIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_sliceStridedSlicecIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/shape_as_tensor:output:0gIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_slice/stack:output:0iIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_slice/stack_1:output:0iIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
[IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB  
]IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB Π
XIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/BroadcastArgsBroadcastArgsfIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0aIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:€
ZIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
ZIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 
VIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
QIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concatConcatV2cIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concat/values_0:output:0]IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/BroadcastArgs:r0:0cIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concat/values_2:output:0_IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:Ν
VIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/BroadcastToBroadcastToaIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ReadVariableOp:value:0ZIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

:	©
XIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   	   Κ
RIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ReshapeReshape_IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/BroadcastTo:output:0aIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

:	
CIndependentDeterministic_CONSTRUCTED_AT_output/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:	
=IndependentDeterministic_CONSTRUCTED_AT_output/sample/ReshapeReshape[IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/Reshape:output:0LIndependentDeterministic_CONSTRUCTED_AT_output/sample/Reshape/shape:output:0*
T0*
_output_shapes
:	
BiasAddBiasAddadd:z:0FIndependentDeterministic_CONSTRUCTED_AT_output/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????	V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????	Λ
ΫKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*
_output_shapes

:	*
dtype0θ
ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplusγKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:	
ΙKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4¬
ΗKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0ΪKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes

:	Β
ΓKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLogΛKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes

:	Ή
ΕKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1LogΘkullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054781*
T0*
_output_shapes
: 
ΓKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSubΗKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0ΙKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes

:	°
ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:	*
dtype0 
ΗKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDivΦKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0Θkullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054781*
T0*
_output_shapes

:	
ΙKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDivΛkullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xΘkullbackleibler_independentnormal_constructed_at_output_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054781*
T0*
_output_shapes

:	?
ΡKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifferenceΛKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0ΝKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes

:	
ΕKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
ΓKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMulΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0ΥKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes

:	
ΗKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
ΕKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1MulΠKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0ΗKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

:	Δ
ΕKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1ΙKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes

:	
ΗKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
ΕKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2MulΠKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0ΙKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes

:	
ΓKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2ΗKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0ΙKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes

:	
ΕKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1SubΗKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0ΗKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

:	Υ
KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"????ώ???τ
qKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSumΙKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * 'GΓ
truedivRealDivzKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0truediv/y:output:0*
T0*
_output_shapes
: K
divergence_kernelIdentitytruediv:z:0*
T0*
_output_shapes
: `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????	Z

Identity_1Identitydivergence_kernel:output:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOpZ^IndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ReadVariableOpά^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpΟ^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp^MatMul_1/ReadVariableOp&^Normal/sample/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : :	2Ά
YIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ReadVariableOpYIndependentDeterministic_CONSTRUCTED_AT_output/sample/Deterministic/sample/ReadVariableOp2Ό
ΫKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpΫKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2’
ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_output/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2N
%Normal/sample/Softplus/ReadVariableOp%Normal/sample/Softplus/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :$ 

_output_shapes

:	
Οψ
π

I__inference_dense_tfp_1_layer_call_and_return_conditional_losses_70054505

inputs@
.normal_sample_softplus_readvariableop_resource:g 2
 matmul_1_readvariableop_resource:g u
gindependentdeterministic_constructed_at_dense_tfp_1_sample_deterministic_sample_readvariableop_resource: ?
Νkullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054475Υ
Πkullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1’^IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ReadVariableOp’ΰKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp’ΣKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp’MatMul_1/ReadVariableOp’%Normal/sample/Softplus/ReadVariableOpk
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"g       U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes

:g ]
Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 
%Normal/sample/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*
_output_shapes

:g *
dtype0z
Normal/sample/SoftplusSoftplus-Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:g X
Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4
Normal/sample/addAddV2Normal/sample/add/x:output:0$Normal/sample/Softplus:activations:0*
T0*
_output_shapes

:g n
Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"g       U
Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Normal/sample/strided_sliceStridedSlice&Normal/sample/shape_as_tensor:output:0*Normal/sample/strided_slice/stack:output:0,Normal/sample/strided_slice/stack_1:output:0,Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"g       W
Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : m
#Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:©
Normal/sample/strided_slice_1StridedSlice(Normal/sample/shape_as_tensor_1:output:0,Normal/sample/strided_slice_1/stack:output:0.Normal/sample/strided_slice_1/stack_1:output:0.Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maska
Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB c
 Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 
Normal/sample/BroadcastArgsBroadcastArgs)Normal/sample/BroadcastArgs/s0_1:output:0$Normal/sample/strided_slice:output:0*
_output_shapes
:
Normal/sample/BroadcastArgs_1BroadcastArgs Normal/sample/BroadcastArgs:r0:0&Normal/sample/strided_slice_1:output:0*
_output_shapes
:g
Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:[
Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ύ
Normal/sample/concatConcatV2&Normal/sample/concat/values_0:output:0"Normal/sample/BroadcastArgs_1:r0:0"Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:l
'Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¨
7Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat:output:0*
T0*"
_output_shapes
:g *
dtype0Π
&Normal/sample/normal/random_normal/mulMul@Normal/sample/normal/random_normal/RandomStandardNormal:output:02Normal/sample/normal/random_normal/stddev:output:0*
T0*"
_output_shapes
:g Ά
"Normal/sample/normal/random_normalAddV2*Normal/sample/normal/random_normal/mul:z:00Normal/sample/normal/random_normal/mean:output:0*
T0*"
_output_shapes
:g 
Normal/sample/mulMul&Normal/sample/normal/random_normal:z:0Normal/sample/add:z:0*
T0*"
_output_shapes
:g u
Normal/sample/add_1AddV2Normal/sample/mul:z:0zeros_like:output:0*
T0*"
_output_shapes
:g l
Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"g       
Normal/sample/ReshapeReshapeNormal/sample/add_1:z:0$Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes

:g ;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ο
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masku
+rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:t
)rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????o
)rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????
%rademacher/uniform/sanitize_seed/seedRandomUniformInt4rademacher/uniform/sanitize_seed/seed/shape:output:02rademacher/uniform/sanitize_seed/seed/min:output:02rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:q
/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R q
/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 RΗ
Hrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter.rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::q
/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :β
+rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2Shape:output:0Nrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Rrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:08rademacher/uniform/stateless_random_uniform/alg:output:08rademacher/uniform/stateless_random_uniform/min:output:08rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????g*
dtype0	R
rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R
rademacher/mulMulrademacher/mul/x:output:04rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????gR
rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rv
rademacher/subSubrademacher/mul:z:0rademacher/sub/y:output:0*
T0	*'
_output_shapes
:?????????gl
rademacher/CastCastrademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????gR
ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value	B : P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsExpandDims/input:output:0ExpandDims/dim:output:0*
T0*
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2strided_slice:output:0ExpandDims:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:w
-rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:v
+rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
ψ????q
+rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :????
'rademacher_1/uniform/sanitize_seed/seedRandomUniformInt6rademacher_1/uniform/sanitize_seed/seed/shape:output:04rademacher_1/uniform/sanitize_seed/seed/min:output:04rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:s
1rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R s
1rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 RΛ
Jrademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter0rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::s
1rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :ο
-rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2concat:output:0Prademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Trademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0:rademacher_1/uniform/stateless_random_uniform/alg:output:0:rademacher_1/uniform/stateless_random_uniform/min:output:0:rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:????????? *
dtype0	T
rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R
rademacher_1/mulMulrademacher_1/mul/x:output:06rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:????????? T
rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R|
rademacher_1/subSubrademacher_1/mul:z:0rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:????????? p
rademacher_1/CastCastrademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:????????? Y
mulMulinputsrademacher/Cast:y:0*
T0*'
_output_shapes
:?????????gk
MatMulMatMulmul:z:0Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:????????? g
mul_1MulMatMul:product:0rademacher_1/Cast:y:0*
T0*'
_output_shapes
:????????? x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:g *
dtype0m
MatMul_1MatMulinputsMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ]
addAddV2MatMul_1:product:0	mul_1:z:0*
T0*'
_output_shapes
:????????? 
GIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 
\IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :·
mIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:
^IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ReadVariableOpReadVariableOpgindependentdeterministic_constructed_at_dense_tfp_1_sample_deterministic_sample_readvariableop_resource*
_output_shapes
: *
dtype0©
_IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: 
UIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ­
cIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ―
eIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:―
eIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ι
]IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_sliceStridedSlicehIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/shape_as_tensor:output:0lIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_slice/stack:output:0nIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_slice/stack_1:output:0nIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask£
`IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ₯
bIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ί
]IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/BroadcastArgsBroadcastArgskIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0fIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:©
_IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:’
_IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 
[IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
VIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concatConcatV2hIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concat/values_0:output:0bIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/BroadcastArgs:r0:0hIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concat/values_2:output:0dIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:ά
[IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/BroadcastToBroadcastTofIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ReadVariableOp:value:0_IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

: ?
]IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       Ω
WIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ReshapeReshapedIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/BroadcastTo:output:0fIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

: 
HIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: §
BIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/ReshapeReshape`IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/Reshape:output:0QIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Reshape/shape:output:0*
T0*
_output_shapes
: 
BiasAddBiasAddadd:z:0KIndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Reshape:output:0*
T0*'
_output_shapes
:????????? P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? Π
ΰKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*
_output_shapes

:g *
dtype0ς
ΡKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplusθKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes

:g 
ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4»
ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2ΧKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0ίKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes

:g Μ
ΘKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLogΠKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes

:g Γ
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1LogΝkullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054475*
T0*
_output_shapes
: 
ΘKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSubΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes

:g ΅
ΣKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:g *
dtype0―
ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDivΫKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0Νkullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054475*
T0*
_output_shapes

:g ¦
ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDivΠkullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xΝkullbackleibler_independentnormal_constructed_at_dense_tfp_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_70054475*
T0*
_output_shapes

:g ½
ΦKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifferenceΠKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes

:g 
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¬
ΘKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMulΣKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0ΪKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes

:g 
ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @’
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1MulΥKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

:g Ξ
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes

:g 
ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?€
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2MulΥKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes

:g 
ΘKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0ΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes

:g 
ΚKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1SubΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0ΜKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes

:g Ϊ
KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"????ώ???
vKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSumΞKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * 'GΘ
truedivRealDivKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0truediv/y:output:0*
T0*
_output_shapes
: K
divergence_kernelIdentitytruediv:z:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? Z

Identity_1Identitydivergence_kernel:output:0^NoOp*
T0*
_output_shapes
: €
NoOpNoOp_^IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ReadVariableOpα^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpΤ^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp^MatMul_1/ReadVariableOp&^Normal/sample/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????g: : : : :g 2ΐ
^IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ReadVariableOp^IndependentDeterministic_CONSTRUCTED_AT_dense_tfp_1/sample/Deterministic/sample/ReadVariableOp2Ζ
ΰKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpΰKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2¬
ΣKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpΣKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_tfp_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2N
%Normal/sample/Softplus/ReadVariableOp%Normal/sample/Softplus/ReadVariableOp:O K
'
_output_shapes
:?????????g
 
_user_specified_nameinputs:

_output_shapes
: :$ 

_output_shapes

:g 
ω
β
H__inference_sequential_layer_call_and_return_conditional_losses_70055115	
input&
dense_tfp_1_70055075:g &
dense_tfp_1_70055077:g "
dense_tfp_1_70055079: 
dense_tfp_1_70055081
dense_tfp_1_70055083&
dense_tfp_2_70055087: &
dense_tfp_2_70055089: "
dense_tfp_2_70055091:
dense_tfp_2_70055093
dense_tfp_2_70055095!
output_70055099:	!
output_70055101:	
output_70055103:	
output_70055105
output_70055107
identity

identity_1

identity_2

identity_3’#dense_tfp_1/StatefulPartitionedCall’#dense_tfp_2/StatefulPartitionedCall’output/StatefulPartitionedCallΝ
#dense_tfp_1/StatefulPartitionedCallStatefulPartitionedCallinputdense_tfp_1_70055075dense_tfp_1_70055077dense_tfp_1_70055079dense_tfp_1_70055081dense_tfp_1_70055083*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dense_tfp_1_layer_call_and_return_conditional_losses_70054505τ
#dense_tfp_2/StatefulPartitionedCallStatefulPartitionedCall,dense_tfp_1/StatefulPartitionedCall:output:0dense_tfp_2_70055087dense_tfp_2_70055089dense_tfp_2_70055091dense_tfp_2_70055093dense_tfp_2_70055095*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????: *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dense_tfp_2_layer_call_and_return_conditional_losses_70054658Ρ
output/StatefulPartitionedCallStatefulPartitionedCall,dense_tfp_2/StatefulPartitionedCall:output:0output_70055099output_70055101output_70055103output_70055105output_70055107*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????	: *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_70054811v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	l

Identity_1Identity,dense_tfp_1/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: l

Identity_2Identity,dense_tfp_2/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: g

Identity_3Identity'output/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: ³
NoOpNoOp$^dense_tfp_1/StatefulPartitionedCall$^dense_tfp_2/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????g: : : : :g : : : : : : : : : :	2J
#dense_tfp_1/StatefulPartitionedCall#dense_tfp_1/StatefulPartitionedCall2J
#dense_tfp_2/StatefulPartitionedCall#dense_tfp_2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:N J
'
_output_shapes
:?????????g

_user_specified_nameinput:

_output_shapes
: :$ 

_output_shapes

:g :	

_output_shapes
: :$
 

_output_shapes

: :

_output_shapes
: :$ 

_output_shapes

:	

»
-__inference_sequential_layer_call_fn_70055240

inputs
unknown:g 
	unknown_0:g 
	unknown_1: 
	unknown_2
	unknown_3
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7
	unknown_8
	unknown_9:	

unknown_10:	

unknown_11:	

unknown_12

unknown_13
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:?????????	: : : *+
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_70054998o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????g: : : : :g : : : : : : : : : :	22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????g
 
_user_specified_nameinputs:

_output_shapes
: :$ 

_output_shapes

:g :	

_output_shapes
: :$
 

_output_shapes

: :

_output_shapes
: :$ 

_output_shapes

:	"ΫL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*₯
serving_default
7
input.
serving_default_input:0?????????g:
output0
StatefulPartitionedCall:0?????????	tensorflow/serving/predict:Ζp
Ϋ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
Ϊ
kernel_posterior_loc
($kernel_posterior_untransformed_scale
kernel_posterior
kernel_prior
bias_posterior_loc
bias_posterior
kernel_posterior_affine
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Ϊ
kernel_posterior_loc
($kernel_posterior_untransformed_scale
kernel_posterior
kernel_prior
bias_posterior_loc
bias_posterior
 kernel_posterior_affine
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
Ϊ
'kernel_posterior_loc
(($kernel_posterior_untransformed_scale
)kernel_posterior
*kernel_prior
+bias_posterior_loc
,bias_posterior
-kernel_posterior_affine
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer

4iter

5beta_1

6beta_2
	7decay
8learning_ratemzm{m|m}m~m'm(m+mvvvvvv'v(v+v"
	optimizer
_
0
1
2
3
4
5
'6
(7
+8"
trackable_list_wrapper
_
0
1
2
3
4
5
'6
(7
+8"
trackable_list_wrapper
 "
trackable_list_wrapper
Κ
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
2?
-__inference_sequential_layer_call_fn_70054864
-__inference_sequential_layer_call_fn_70055202
-__inference_sequential_layer_call_fn_70055240
-__inference_sequential_layer_call_fn_70055072ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ξ2λ
H__inference_sequential_layer_call_and_return_conditional_losses_70055654
H__inference_sequential_layer_call_and_return_conditional_losses_70056068
H__inference_sequential_layer_call_and_return_conditional_losses_70055115
H__inference_sequential_layer_call_and_return_conditional_losses_70055158ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ΜBΙ
#__inference__wrapped_model_70054356input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
,
>serving_default"
signature_map
2:0g 2 dense_tfp_1/kernel_posterior_loc
B:@g 20dense_tfp_1/kernel_posterior_untransformed_scale
E
?_distribution
@_graph_parents"
_generic_user_object
E
A_distribution
B_graph_parents"
_generic_user_object
,:* 2dense_tfp_1/bias_posterior_loc
E
C_distribution
D_graph_parents"
_generic_user_object
>

E_scale
F_graph_parents"
_generic_user_object
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ψ2Υ
.__inference_dense_tfp_1_layer_call_fn_70056121’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_dense_tfp_1_layer_call_and_return_conditional_losses_70056263’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2:0 2 dense_tfp_2/kernel_posterior_loc
B:@ 20dense_tfp_2/kernel_posterior_untransformed_scale
E
L_distribution
M_graph_parents"
_generic_user_object
E
N_distribution
O_graph_parents"
_generic_user_object
,:*2dense_tfp_2/bias_posterior_loc
E
P_distribution
Q_graph_parents"
_generic_user_object
>

R_scale
S_graph_parents"
_generic_user_object
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
Ψ2Υ
.__inference_dense_tfp_2_layer_call_fn_70056279’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
σ2π
I__inference_dense_tfp_2_layer_call_and_return_conditional_losses_70056419’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
-:+	2output/kernel_posterior_loc
=:;	2+output/kernel_posterior_untransformed_scale
E
Y_distribution
Z_graph_parents"
_generic_user_object
E
[_distribution
\_graph_parents"
_generic_user_object
':%	2output/bias_posterior_loc
E
]_distribution
^_graph_parents"
_generic_user_object
>

__scale
`_graph_parents"
_generic_user_object
5
'0
(1
+2"
trackable_list_wrapper
5
'0
(1
+2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
Σ2Π
)__inference_output_layer_call_fn_70056435’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
D__inference_output_layer_call_and_return_conditional_losses_70056575’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ΛBΘ
&__inference_signature_wrapper_70056105input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
H
_loc

E_scale
h_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
2
i_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
<
_loc
j_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
9
_pretransformed_input"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
H
_loc

R_scale
k_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
2
l_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
<
_loc
m_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
9
_pretransformed_input"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
H
'_loc

__scale
n_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
2
o_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
<
+_loc
p_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
9
(_pretransformed_input"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	qtotal
	rcount
s	variables
t	keras_api"
_tf_keras_metric
^
	utotal
	vcount
w
_fn_kwargs
x	variables
y	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
q0
r1"
trackable_list_wrapper
-
s	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
u0
v1"
trackable_list_wrapper
-
x	variables"
_generic_user_object
7:5g 2'Adam/dense_tfp_1/kernel_posterior_loc/m
G:Eg 27Adam/dense_tfp_1/kernel_posterior_untransformed_scale/m
1:/ 2%Adam/dense_tfp_1/bias_posterior_loc/m
7:5 2'Adam/dense_tfp_2/kernel_posterior_loc/m
G:E 27Adam/dense_tfp_2/kernel_posterior_untransformed_scale/m
1:/2%Adam/dense_tfp_2/bias_posterior_loc/m
2:0	2"Adam/output/kernel_posterior_loc/m
B:@	22Adam/output/kernel_posterior_untransformed_scale/m
,:*	2 Adam/output/bias_posterior_loc/m
7:5g 2'Adam/dense_tfp_1/kernel_posterior_loc/v
G:Eg 27Adam/dense_tfp_1/kernel_posterior_untransformed_scale/v
1:/ 2%Adam/dense_tfp_1/bias_posterior_loc/v
7:5 2'Adam/dense_tfp_2/kernel_posterior_loc/v
G:E 27Adam/dense_tfp_2/kernel_posterior_untransformed_scale/v
1:/2%Adam/dense_tfp_2/bias_posterior_loc/v
2:0	2"Adam/output/kernel_posterior_loc/v
B:@	22Adam/output/kernel_posterior_untransformed_scale/v
,:*	2 Adam/output/bias_posterior_loc/v
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5
#__inference__wrapped_model_70054356x('+.’+
$’!

input?????????g
ͺ "/ͺ,
*
output 
output?????????	Ό
I__inference_dense_tfp_1_layer_call_and_return_conditional_losses_70056263o/’,
%’"
 
inputs?????????g
ͺ "3’0

0????????? 

	
1/0 
.__inference_dense_tfp_1_layer_call_fn_70056121T/’,
%’"
 
inputs?????????g
ͺ "????????? Ό
I__inference_dense_tfp_2_layer_call_and_return_conditional_losses_70056419o/’,
%’"
 
inputs????????? 
ͺ "3’0

0?????????

	
1/0 
.__inference_dense_tfp_2_layer_call_fn_70056279T/’,
%’"
 
inputs????????? 
ͺ "?????????·
D__inference_output_layer_call_and_return_conditional_losses_70056575o('+/’,
%’"
 
inputs?????????
ͺ "3’0

0?????????	

	
1/0 
)__inference_output_layer_call_fn_70056435T('+/’,
%’"
 
inputs?????????
ͺ "?????????	ν
H__inference_sequential_layer_call_and_return_conditional_losses_70055115 ('+6’3
,’)

input?????????g
p 

 
ͺ "O’L

0?????????	
-*
	
1/0 
	
1/1 
	
1/2 ν
H__inference_sequential_layer_call_and_return_conditional_losses_70055158 ('+6’3
,’)

input?????????g
p

 
ͺ "O’L

0?????????	
-*
	
1/0 
	
1/1 
	
1/2 ξ
H__inference_sequential_layer_call_and_return_conditional_losses_70055654‘('+7’4
-’*
 
inputs?????????g
p 

 
ͺ "O’L

0?????????	
-*
	
1/0 
	
1/1 
	
1/2 ξ
H__inference_sequential_layer_call_and_return_conditional_losses_70056068‘('+7’4
-’*
 
inputs?????????g
p

 
ͺ "O’L

0?????????	
-*
	
1/0 
	
1/1 
	
1/2 
-__inference_sequential_layer_call_fn_70054864i('+6’3
,’)

input?????????g
p 

 
ͺ "?????????	
-__inference_sequential_layer_call_fn_70055072i('+6’3
,’)

input?????????g
p

 
ͺ "?????????	
-__inference_sequential_layer_call_fn_70055202j('+7’4
-’*
 
inputs?????????g
p 

 
ͺ "?????????	
-__inference_sequential_layer_call_fn_70055240j('+7’4
-’*
 
inputs?????????g
p

 
ͺ "?????????	¬
&__inference_signature_wrapper_70056105('+7’4
’ 
-ͺ*
(
input
input?????????g"/ͺ,
*
output 
output?????????	