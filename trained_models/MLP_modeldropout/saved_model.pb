®
§
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
®
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
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

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
0
Sigmoid
x"T
y"T"
Ttype:

2
Á
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
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02unknown8ðÅ
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:*
dtype0

Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0

Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

:@*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

: @*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
: *
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	 *
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
î*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
î*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:*
dtype0

Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0

Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

: @*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
: *
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	 *
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
î*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
î*
dtype0
v
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0
t
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
{
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:È*"
shared_namefalse_negatives_1
t
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes	
:È*
dtype0
{
false_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:È*"
shared_namefalse_positives_1
t
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes	
:È*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:È*
dtype0
y
true_positives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:È*!
shared_nametrue_positives_2
r
$true_positives_2/Read/ReadVariableOpReadVariableOptrue_positives_2*
_output_shapes	
:È*
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
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:@*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:@*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

: @*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
: *
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	 *
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
î*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
î*
dtype0
¤
serving_default_dense_inputPlaceholder*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî*
dtype0*/
shape&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
í
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_inputdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_42505

NoOpNoOp
ÇU
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*U
valueøTBõT BîT
©
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias*
¥
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_random_generator* 
¦
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias*
¥
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6_random_generator* 
¦
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias*
¦
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Ekernel
Fbias*
J
0
1
2
 3
.4
/5
=6
>7
E8
F9*
J
0
1
2
 3
.4
/5
=6
>7
E8
F9*
* 
°
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_3* 
6
Ptrace_0
Qtrace_1
Rtrace_2
Strace_3* 
* 

Titer

Ubeta_1

Vbeta_2
	Wdecay
Xlearning_ratem§m¨m© mª.m«/m¬=m­>m®Em¯Fm°v±v²v³ v´.vµ/v¶=v·>v¸Ev¹Fvº*

Yserving_default* 

0
1*

0
1*
* 

Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

_trace_0* 

`trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
 1*

0
 1*
* 

anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

ftrace_0* 

gtrace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses* 

mtrace_0
ntrace_1* 

otrace_0
ptrace_1* 
* 

.0
/1*

.0
/1*
* 

qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

vtrace_0* 

wtrace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 

}trace_0
~trace_1* 

trace_0
trace_1* 
* 

=0
>1*

=0
>1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

trace_0* 

trace_0* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

E0
F1*

E0
F1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*

trace_0* 

trace_0* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
0
1
2
3
4
5
6*
$
0
1
2
3*
* 
* 
* 
* 
* 
* 
* 
* 
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
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
	variables
	keras_api

total

count*
z
	variables
	keras_api
true_positives
true_negatives
false_positives
false_negatives*
`
	variables
	keras_api

thresholds
 true_positives
¡false_negatives*
`
¢	variables
£	keras_api
¤
thresholds
¥true_positives
¦false_positives*

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
$
0
1
2
3*

	variables*
ga
VARIABLE_VALUEtrue_positives_2=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_positives_1>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*

 0
¡1*

	variables*
* 
ga
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_negatives>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*

¥0
¦1*

¢	variables*
* 
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ì
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp$true_positives_2/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp%false_positives_1/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_43263
ã
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttrue_positives_2true_negativesfalse_positives_1false_negatives_1true_positives_1false_negativestrue_positivesfalse_positivesAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_43408ßý

ã

ó
*__inference_sequential_layer_call_fn_42530

inputs
unknown:
î
	unknown_0:	
	unknown_1:	 
	unknown_2: 
	unknown_3: @
	unknown_4:@
	unknown_5:@
	unknown_6:
	unknown_7:
	unknown_8:
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_42185
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
 
_user_specified_nameinputs

`
B__inference_dropout_layer_call_and_return_conditional_losses_42064

inputs

identity_1`
IdentityIdentityinputs*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc m

Identity_1IdentityIdentity:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc :a ]
9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
 
_user_specified_nameinputs


'__inference_dense_4_layer_call_fn_43074

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_42178
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc: : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
 
_user_specified_nameinputs
Æ¹
´	
 __inference__wrapped_model_41978
dense_inputF
2sequential_dense_tensordot_readvariableop_resource:
î?
0sequential_dense_biasadd_readvariableop_resource:	G
4sequential_dense_1_tensordot_readvariableop_resource:	 @
2sequential_dense_1_biasadd_readvariableop_resource: F
4sequential_dense_2_tensordot_readvariableop_resource: @@
2sequential_dense_2_biasadd_readvariableop_resource:@F
4sequential_dense_3_tensordot_readvariableop_resource:@@
2sequential_dense_3_biasadd_readvariableop_resource:F
4sequential_dense_4_tensordot_readvariableop_resource:@
2sequential_dense_4_biasadd_readvariableop_resource:
identity¢'sequential/dense/BiasAdd/ReadVariableOp¢)sequential/dense/Tensordot/ReadVariableOp¢)sequential/dense_1/BiasAdd/ReadVariableOp¢+sequential/dense_1/Tensordot/ReadVariableOp¢)sequential/dense_2/BiasAdd/ReadVariableOp¢+sequential/dense_2/Tensordot/ReadVariableOp¢)sequential/dense_3/BiasAdd/ReadVariableOp¢+sequential/dense_3/Tensordot/ReadVariableOp¢)sequential/dense_4/BiasAdd/ReadVariableOp¢+sequential/dense_4/Tensordot/ReadVariableOp
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource* 
_output_shapes
:
î*
dtype0i
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          [
 sequential/dense/Tensordot/ShapeShapedense_input*
T0*
_output_shapes
:j
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : à
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¬
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¯
$sequential/dense/Tensordot/transpose	Transposedense_input*sequential/dense/Tensordot/concat:output:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî½
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¾
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:j
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Å
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¾
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
sequential/dense/TanhTanh!sequential/dense/BiasAdd:output:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc¡
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	 *
dtype0k
!sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:v
!sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          k
"sequential/dense_1/Tensordot/ShapeShapesequential/dense/Tanh:y:0*
T0*
_output_shapes
:l
*sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense_1/Tensordot/GatherV2GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/free:output:03sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
'sequential/dense_1/Tensordot/GatherV2_1GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/axes:output:05sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense_1/Tensordot/ProdProd.sequential/dense_1/Tensordot/GatherV2:output:0+sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_1/Tensordot/Prod_1Prod0sequential/dense_1/Tensordot/GatherV2_1:output:0-sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : è
#sequential/dense_1/Tensordot/concatConcatV2*sequential/dense_1/Tensordot/free:output:0*sequential/dense_1/Tensordot/axes:output:01sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:²
"sequential/dense_1/Tensordot/stackPack*sequential/dense_1/Tensordot/Prod:output:0,sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Á
&sequential/dense_1/Tensordot/transpose	Transposesequential/dense/Tanh:y:0,sequential/dense_1/Tensordot/concat:output:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcÃ
$sequential/dense_1/Tensordot/ReshapeReshape*sequential/dense_1/Tensordot/transpose:y:0+sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
$sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: l
*sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
%sequential/dense_1/Tensordot/concat_1ConcatV2.sequential/dense_1/Tensordot/GatherV2:output:0-sequential/dense_1/Tensordot/Const_2:output:03sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ê
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0.sequential/dense_1/Tensordot/concat_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ã
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
sequential/dense_1/TanhTanh#sequential/dense_1/BiasAdd:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
sequential/dropout/IdentityIdentitysequential/dense_1/Tanh:y:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc  
+sequential/dense_2/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_2_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0k
!sequential/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:v
!sequential/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          v
"sequential/dense_2/Tensordot/ShapeShape$sequential/dropout/Identity:output:0*
T0*
_output_shapes
:l
*sequential/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense_2/Tensordot/GatherV2GatherV2+sequential/dense_2/Tensordot/Shape:output:0*sequential/dense_2/Tensordot/free:output:03sequential/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
'sequential/dense_2/Tensordot/GatherV2_1GatherV2+sequential/dense_2/Tensordot/Shape:output:0*sequential/dense_2/Tensordot/axes:output:05sequential/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense_2/Tensordot/ProdProd.sequential/dense_2/Tensordot/GatherV2:output:0+sequential/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_2/Tensordot/Prod_1Prod0sequential/dense_2/Tensordot/GatherV2_1:output:0-sequential/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : è
#sequential/dense_2/Tensordot/concatConcatV2*sequential/dense_2/Tensordot/free:output:0*sequential/dense_2/Tensordot/axes:output:01sequential/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:²
"sequential/dense_2/Tensordot/stackPack*sequential/dense_2/Tensordot/Prod:output:0,sequential/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ë
&sequential/dense_2/Tensordot/transpose	Transpose$sequential/dropout/Identity:output:0,sequential/dense_2/Tensordot/concat:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc Ã
$sequential/dense_2/Tensordot/ReshapeReshape*sequential/dense_2/Tensordot/transpose:y:0+sequential/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
#sequential/dense_2/Tensordot/MatMulMatMul-sequential/dense_2/Tensordot/Reshape:output:03sequential/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
$sequential/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@l
*sequential/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
%sequential/dense_2/Tensordot/concat_1ConcatV2.sequential/dense_2/Tensordot/GatherV2:output:0-sequential/dense_2/Tensordot/Const_2:output:03sequential/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ê
sequential/dense_2/TensordotReshape-sequential/dense_2/Tensordot/MatMul:product:0.sequential/dense_2/Tensordot/concat_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ã
sequential/dense_2/BiasAddBiasAdd%sequential/dense_2/Tensordot:output:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
sequential/dense_2/TanhTanh#sequential/dense_2/BiasAdd:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
sequential/dropout_1/IdentityIdentitysequential/dense_2/Tanh:y:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@ 
+sequential/dense_3/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_3_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0k
!sequential/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:v
!sequential/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          x
"sequential/dense_3/Tensordot/ShapeShape&sequential/dropout_1/Identity:output:0*
T0*
_output_shapes
:l
*sequential/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense_3/Tensordot/GatherV2GatherV2+sequential/dense_3/Tensordot/Shape:output:0*sequential/dense_3/Tensordot/free:output:03sequential/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
'sequential/dense_3/Tensordot/GatherV2_1GatherV2+sequential/dense_3/Tensordot/Shape:output:0*sequential/dense_3/Tensordot/axes:output:05sequential/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense_3/Tensordot/ProdProd.sequential/dense_3/Tensordot/GatherV2:output:0+sequential/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_3/Tensordot/Prod_1Prod0sequential/dense_3/Tensordot/GatherV2_1:output:0-sequential/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : è
#sequential/dense_3/Tensordot/concatConcatV2*sequential/dense_3/Tensordot/free:output:0*sequential/dense_3/Tensordot/axes:output:01sequential/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:²
"sequential/dense_3/Tensordot/stackPack*sequential/dense_3/Tensordot/Prod:output:0,sequential/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Í
&sequential/dense_3/Tensordot/transpose	Transpose&sequential/dropout_1/Identity:output:0,sequential/dense_3/Tensordot/concat:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@Ã
$sequential/dense_3/Tensordot/ReshapeReshape*sequential/dense_3/Tensordot/transpose:y:0+sequential/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
#sequential/dense_3/Tensordot/MatMulMatMul-sequential/dense_3/Tensordot/Reshape:output:03sequential/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
$sequential/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:l
*sequential/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
%sequential/dense_3/Tensordot/concat_1ConcatV2.sequential/dense_3/Tensordot/GatherV2:output:0-sequential/dense_3/Tensordot/Const_2:output:03sequential/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ê
sequential/dense_3/TensordotReshape-sequential/dense_3/Tensordot/MatMul:product:0.sequential/dense_3/Tensordot/concat_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
)sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ã
sequential/dense_3/BiasAddBiasAdd%sequential/dense_3/Tensordot:output:01sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
sequential/dense_3/TanhTanh#sequential/dense_3/BiasAdd:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
+sequential/dense_4/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_4_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0k
!sequential/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:v
!sequential/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          m
"sequential/dense_4/Tensordot/ShapeShapesequential/dense_3/Tanh:y:0*
T0*
_output_shapes
:l
*sequential/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%sequential/dense_4/Tensordot/GatherV2GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/free:output:03sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,sequential/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
'sequential/dense_4/Tensordot/GatherV2_1GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/axes:output:05sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"sequential/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: §
!sequential/dense_4/Tensordot/ProdProd.sequential/dense_4/Tensordot/GatherV2:output:0+sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#sequential/dense_4/Tensordot/Prod_1Prod0sequential/dense_4/Tensordot/GatherV2_1:output:0-sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : è
#sequential/dense_4/Tensordot/concatConcatV2*sequential/dense_4/Tensordot/free:output:0*sequential/dense_4/Tensordot/axes:output:01sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:²
"sequential/dense_4/Tensordot/stackPack*sequential/dense_4/Tensordot/Prod:output:0,sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Â
&sequential/dense_4/Tensordot/transpose	Transposesequential/dense_3/Tanh:y:0,sequential/dense_4/Tensordot/concat:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcÃ
$sequential/dense_4/Tensordot/ReshapeReshape*sequential/dense_4/Tensordot/transpose:y:0+sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
#sequential/dense_4/Tensordot/MatMulMatMul-sequential/dense_4/Tensordot/Reshape:output:03sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
$sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:l
*sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
%sequential/dense_4/Tensordot/concat_1ConcatV2.sequential/dense_4/Tensordot/GatherV2:output:0-sequential/dense_4/Tensordot/Const_2:output:03sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ê
sequential/dense_4/TensordotReshape-sequential/dense_4/Tensordot/MatMul:product:0.sequential/dense_4/Tensordot/concat_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ã
sequential/dense_4/BiasAddBiasAdd%sequential/dense_4/Tensordot:output:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
sequential/dense_4/SigmoidSigmoid#sequential/dense_4/BiasAdd:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
IdentityIdentitysequential/dense_4/Sigmoid:y:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/Tensordot/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp,^sequential/dense_2/Tensordot/ReadVariableOp*^sequential/dense_3/BiasAdd/ReadVariableOp,^sequential/dense_3/Tensordot/ReadVariableOp*^sequential/dense_4/BiasAdd/ReadVariableOp,^sequential/dense_4/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî: : : : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/Tensordot/ReadVariableOp+sequential/dense_1/Tensordot/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2Z
+sequential/dense_2/Tensordot/ReadVariableOp+sequential/dense_2/Tensordot/ReadVariableOp2V
)sequential/dense_3/BiasAdd/ReadVariableOp)sequential/dense_3/BiasAdd/ReadVariableOp2Z
+sequential/dense_3/Tensordot/ReadVariableOp+sequential/dense_3/Tensordot/ReadVariableOp2V
)sequential/dense_4/BiasAdd/ReadVariableOp)sequential/dense_4/BiasAdd/ReadVariableOp2Z
+sequential/dense_4/Tensordot/ReadVariableOp+sequential/dense_4/Tensordot/ReadVariableOp:g c
:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
%
_user_specified_namedense_input
ÿ
ù
B__inference_dense_3_layer_call_and_return_conditional_losses_42141

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcb
TanhTanhBiasAdd:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØci
IdentityIdentityTanh:y:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:a ]
9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
 
_user_specified_nameinputs

a
B__inference_dropout_layer_call_and_return_conditional_losses_42958

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?v
dropout/MulMulinputsdropout/Const:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¸
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc {
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc k
IdentityIdentitydropout/Mul_1:z:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc :a ]
9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
 
_user_specified_nameinputs
å
E
)__inference_dropout_1_layer_call_fn_43003

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_42108r
IdentityIdentityPartitionedCall:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@:a ]
9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
 
_user_specified_nameinputs

ú
@__inference_dense_layer_call_and_return_conditional_losses_42891

inputs5
!tensordot_readvariableop_resource:
î.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
î*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcc
TanhTanhBiasAdd:output:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcj
IdentityIdentityTanh:y:0^NoOp*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:b ^
:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
 
_user_specified_nameinputs
Æ

ñ
#__inference_signature_wrapper_42505
dense_input
unknown:
î
	unknown_0:	
	unknown_1:	 
	unknown_2: 
	unknown_3: @
	unknown_4:@
	unknown_5:@
	unknown_6:
	unknown_7:
	unknown_8:
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_41978
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
%
_user_specified_namedense_input
ÿ
ù
B__inference_dense_3_layer_call_and_return_conditional_losses_43065

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcb
TanhTanhBiasAdd:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØci
IdentityIdentityTanh:y:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:a ]
9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
 
_user_specified_nameinputs

`
B__inference_dropout_layer_call_and_return_conditional_losses_42946

inputs

identity_1`
IdentityIdentityinputs*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc m

Identity_1IdentityIdentity:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc :a ]
9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
 
_user_specified_nameinputs

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_43025

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?v
dropout/MulMulinputsdropout/Const:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¸
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@k
IdentityIdentitydropout/Mul_1:z:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@:a ]
9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
 
_user_specified_nameinputs
ò

ø
*__inference_sequential_layer_call_fn_42410
dense_input
unknown:
î
	unknown_0:	
	unknown_1:	 
	unknown_2: 
	unknown_3: @
	unknown_4:@
	unknown_5:@
	unknown_6:
	unknown_7:
	unknown_8:
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_42362
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
%
_user_specified_namedense_input
ò

ø
*__inference_sequential_layer_call_fn_42208
dense_input
unknown:
î
	unknown_0:	
	unknown_1:	 
	unknown_2: 
	unknown_3: @
	unknown_4:@
	unknown_5:@
	unknown_6:
	unknown_7:
	unknown_8:
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_42185
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
%
_user_specified_namedense_input
É
ø
E__inference_sequential_layer_call_and_return_conditional_losses_42696

inputs;
'dense_tensordot_readvariableop_resource:
î4
%dense_biasadd_readvariableop_resource:	<
)dense_1_tensordot_readvariableop_resource:	 5
'dense_1_biasadd_readvariableop_resource: ;
)dense_2_tensordot_readvariableop_resource: @5
'dense_2_biasadd_readvariableop_resource:@;
)dense_3_tensordot_readvariableop_resource:@5
'dense_3_biasadd_readvariableop_resource:;
)dense_4_tensordot_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/Tensordot/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢ dense_1/Tensordot/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢ dense_2/Tensordot/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢ dense_3/Tensordot/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢ dense_4/Tensordot/ReadVariableOp
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource* 
_output_shapes
:
î*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:i
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          K
dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ´
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/transpose	Transposeinputsdense/Tensordot/concat:output:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¤
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØco

dense/TanhTanhdense/BiasAdd:output:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	 *
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          U
dense_1/Tensordot/ShapeShapedense/Tanh:y:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
: 
dense_1/Tensordot/transpose	Transposedense/Tanh:y:0!dense_1/Tensordot/concat:output:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc¢
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:©
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¢
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc r
dense_1/TanhTanhdense_1/BiasAdd:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc r
dropout/IdentityIdentitydense_1/Tanh:y:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0`
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          `
dense_2/Tensordot/ShapeShapedropout/Identity:output:0*
T0*
_output_shapes
:a
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ª
dense_2/Tensordot/transpose	Transposedropout/Identity:output:0!dense_2/Tensordot/concat:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc ¢
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@a
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:©
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¢
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@r
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@t
dropout_1/IdentityIdentitydense_2/Tanh:y:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0`
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          b
dense_3/Tensordot/ShapeShapedropout_1/Identity:output:0*
T0*
_output_shapes
:a
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¬
dense_3/Tensordot/transpose	Transposedropout_1/Identity:output:0!dense_3/Tensordot/concat:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@¢
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:©
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¢
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcr
dense_3/TanhTanhdense_3/BiasAdd:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          W
dense_4/Tensordot/ShapeShapedense_3/Tanh:y:0*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¡
dense_4/Tensordot/transpose	Transposedense_3/Tanh:y:0!dense_4/Tensordot/concat:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc¢
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:©
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¢
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcx
dense_4/SigmoidSigmoiddense_4/BiasAdd:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØct
IdentityIdentitydense_4/Sigmoid:y:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp:b ^
:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
 
_user_specified_nameinputs


'__inference_dense_3_layer_call_fn_43034

inputs
unknown:@
	unknown_0:
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_42141
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@: : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
 
_user_specified_nameinputs
à!
Ç
E__inference_sequential_layer_call_and_return_conditional_losses_42441
dense_input
dense_42413:
î
dense_42415:	 
dense_1_42418:	 
dense_1_42420: 
dense_2_42424: @
dense_2_42426:@
dense_3_42430:@
dense_3_42432:
dense_4_42435:
dense_4_42437:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCallù
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_42413dense_42415*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_42016
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_42418dense_1_42420*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_42053é
dropout/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_42064
dense_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_2_42424dense_2_42426*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_42097í
dropout_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_42108
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_3_42430dense_3_42432*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_42141
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_42435dense_4_42437*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_42178
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:g c
:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
%
_user_specified_namedense_input
ÿ
ù
B__inference_dense_2_layer_call_and_return_conditional_losses_42097

inputs3
!tensordot_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@b
TanhTanhBiasAdd:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@i
IdentityIdentityTanh:y:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:a ]
9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
 
_user_specified_nameinputs


'__inference_dense_2_layer_call_fn_42967

inputs
unknown: @
	unknown_0:@
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_42097
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
 
_user_specified_nameinputs
Ñ!
Â
E__inference_sequential_layer_call_and_return_conditional_losses_42185

inputs
dense_42017:
î
dense_42019:	 
dense_1_42054:	 
dense_1_42056: 
dense_2_42098: @
dense_2_42100:@
dense_3_42142:@
dense_3_42144:
dense_4_42179:
dense_4_42181:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCallô
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_42017dense_42019*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_42016
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_42054dense_1_42056*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_42053é
dropout/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_42064
dense_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_2_42098dense_2_42100*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_42097í
dropout_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_42108
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_3_42142dense_3_42144*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_42141
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_42179dense_4_42181*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_42178
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:b ^
:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
 
_user_specified_nameinputs

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_42248

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?v
dropout/MulMulinputsdropout/Const:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¸
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@k
IdentityIdentitydropout/Mul_1:z:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@:a ]
9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
 
_user_specified_nameinputs

a
B__inference_dropout_layer_call_and_return_conditional_losses_42281

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?v
dropout/MulMulinputsdropout/Const:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¸
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc {
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc k
IdentityIdentitydropout/Mul_1:z:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc :a ]
9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
 
_user_specified_nameinputs


%__inference_dense_layer_call_fn_42860

inputs
unknown:
î
	unknown_0:	
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_42016
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî: : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
 
_user_specified_nameinputs
Ê$

E__inference_sequential_layer_call_and_return_conditional_losses_42472
dense_input
dense_42444:
î
dense_42446:	 
dense_1_42449:	 
dense_1_42451: 
dense_2_42455: @
dense_2_42457:@
dense_3_42461:@
dense_3_42463:
dense_4_42466:
dense_4_42468:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCallù
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_42444dense_42446*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_42016
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_42449dense_1_42451*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_42053ù
dropout/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_42281
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_2_42455dense_2_42457*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_42097
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_42248
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_3_42461dense_3_42463*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_42141
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_42466dense_4_42468*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_42178
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc´
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:g c
:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
%
_user_specified_namedense_input
çY
ÿ
__inference__traced_save_43263
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop/
+savev2_true_positives_2_read_readvariableop-
)savev2_true_negatives_read_readvariableop0
,savev2_false_positives_1_read_readvariableop0
,savev2_false_negatives_1_read_readvariableop/
+savev2_true_positives_1_read_readvariableop.
*savev2_false_negatives_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
: ù
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*¢
valueB.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÉ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ½
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop+savev2_true_positives_2_read_readvariableop)savev2_true_negatives_read_readvariableop,savev2_false_positives_1_read_readvariableop,savev2_false_negatives_1_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_negatives_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	
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

identity_1Identity_1:output:0*×
_input_shapesÅ
Â: :
î::	 : : @:@:@:::: : : : : : : :È:È:È:È:::::
î::	 : : @:@:@::::
î::	 : : @:@:@:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
î:!

_output_shapes	
::%!

_output_shapes
:	 : 

_output_shapes
: :$ 

_output_shapes

: @: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::
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
: :!

_output_shapes	
:È:!

_output_shapes	
:È:!

_output_shapes	
:È:!

_output_shapes	
:È: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
î:!

_output_shapes	
::%!

_output_shapes
:	 : 

_output_shapes
: :$ 

_output_shapes

: @: 

_output_shapes
:@:$  

_output_shapes

:@: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::&$"
 
_output_shapes
:
î:!%

_output_shapes	
::%&!

_output_shapes
:	 : '

_output_shapes
: :$( 

_output_shapes

: @: )

_output_shapes
:@:$* 

_output_shapes

:@: +

_output_shapes
::$, 

_output_shapes

:: -

_output_shapes
::.

_output_shapes
: 
´
`
'__inference_dropout_layer_call_fn_42941

inputs
identity¢StatefulPartitionedCallÏ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_42281
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
 
_user_specified_nameinputs


'__inference_dense_1_layer_call_fn_42900

inputs
unknown:	 
	unknown_0: 
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_42053
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc: : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
 
_user_specified_nameinputs
ÿ
ù
B__inference_dense_2_layer_call_and_return_conditional_losses_42998

inputs3
!tensordot_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@b
TanhTanhBiasAdd:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@i
IdentityIdentityTanh:y:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:a ]
9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
 
_user_specified_nameinputs

ù
B__inference_dense_4_layer_call_and_return_conditional_losses_43105

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØch
SigmoidSigmoidBiasAdd:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcl
IdentityIdentitySigmoid:y:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:a ]
9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
 
_user_specified_nameinputs

ú
B__inference_dense_1_layer_call_and_return_conditional_losses_42053

inputs4
!tensordot_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	 *
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc b
TanhTanhBiasAdd:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc i
IdentityIdentityTanh:y:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:b ^
:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
 
_user_specified_nameinputs
á
C
'__inference_dropout_layer_call_fn_42936

inputs
identity¿
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_42064r
IdentityIdentityPartitionedCall:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc :a ]
9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
 
_user_specified_nameinputs
²
Ù
!__inference__traced_restore_43408
file_prefix1
assignvariableop_dense_kernel:
î,
assignvariableop_1_dense_bias:	4
!assignvariableop_2_dense_1_kernel:	 -
assignvariableop_3_dense_1_bias: 3
!assignvariableop_4_dense_2_kernel: @-
assignvariableop_5_dense_2_bias:@3
!assignvariableop_6_dense_3_kernel:@-
assignvariableop_7_dense_3_bias:3
!assignvariableop_8_dense_4_kernel:-
assignvariableop_9_dense_4_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: #
assignvariableop_15_total: #
assignvariableop_16_count: 3
$assignvariableop_17_true_positives_2:	È1
"assignvariableop_18_true_negatives:	È4
%assignvariableop_19_false_positives_1:	È4
%assignvariableop_20_false_negatives_1:	È2
$assignvariableop_21_true_positives_1:1
#assignvariableop_22_false_negatives:0
"assignvariableop_23_true_positives:1
#assignvariableop_24_false_positives:;
'assignvariableop_25_adam_dense_kernel_m:
î4
%assignvariableop_26_adam_dense_bias_m:	<
)assignvariableop_27_adam_dense_1_kernel_m:	 5
'assignvariableop_28_adam_dense_1_bias_m: ;
)assignvariableop_29_adam_dense_2_kernel_m: @5
'assignvariableop_30_adam_dense_2_bias_m:@;
)assignvariableop_31_adam_dense_3_kernel_m:@5
'assignvariableop_32_adam_dense_3_bias_m:;
)assignvariableop_33_adam_dense_4_kernel_m:5
'assignvariableop_34_adam_dense_4_bias_m:;
'assignvariableop_35_adam_dense_kernel_v:
î4
%assignvariableop_36_adam_dense_bias_v:	<
)assignvariableop_37_adam_dense_1_kernel_v:	 5
'assignvariableop_38_adam_dense_1_bias_v: ;
)assignvariableop_39_adam_dense_2_kernel_v: @5
'assignvariableop_40_adam_dense_2_bias_v:@;
)assignvariableop_41_adam_dense_3_kernel_v:@5
'assignvariableop_42_adam_dense_3_bias_v:;
)assignvariableop_43_adam_dense_4_kernel_v:5
'assignvariableop_44_adam_dense_4_bias_v:
identity_46¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ü
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*¢
valueB.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÌ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Î
_output_shapes»
¸::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp$assignvariableop_17_true_positives_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp"assignvariableop_18_true_negativesIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp%assignvariableop_19_false_positives_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp%assignvariableop_20_false_negatives_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp$assignvariableop_21_true_positives_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp#assignvariableop_22_false_negativesIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp"assignvariableop_23_true_positivesIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp#assignvariableop_24_false_positivesIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_dense_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_1_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_1_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_2_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_2_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_3_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_3_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_4_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_4_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_dense_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp%assignvariableop_36_adam_dense_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_2_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_2_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_3_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_3_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_4_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_4_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ­
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_46IdentityIdentity_45:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
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

ú
@__inference_dense_layer_call_and_return_conditional_losses_42016

inputs5
!tensordot_readvariableop_resource:
î.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
î*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcc
TanhTanhBiasAdd:output:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcj
IdentityIdentityTanh:y:0^NoOp*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:b ^
:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
 
_user_specified_nameinputs

ù
B__inference_dense_4_layer_call_and_return_conditional_losses_42178

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØch
SigmoidSigmoidBiasAdd:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcl
IdentityIdentitySigmoid:y:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:a ]
9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
 
_user_specified_nameinputs
¯¬
ø
E__inference_sequential_layer_call_and_return_conditional_losses_42851

inputs;
'dense_tensordot_readvariableop_resource:
î4
%dense_biasadd_readvariableop_resource:	<
)dense_1_tensordot_readvariableop_resource:	 5
'dense_1_biasadd_readvariableop_resource: ;
)dense_2_tensordot_readvariableop_resource: @5
'dense_2_biasadd_readvariableop_resource:@;
)dense_3_tensordot_readvariableop_resource:@5
'dense_3_biasadd_readvariableop_resource:;
)dense_4_tensordot_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/Tensordot/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢ dense_1/Tensordot/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢ dense_2/Tensordot/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢ dense_3/Tensordot/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢ dense_4/Tensordot/ReadVariableOp
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource* 
_output_shapes
:
î*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:i
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          K
dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ´
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/transpose	Transposeinputsdense/Tensordot/concat:output:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¤
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØco

dense/TanhTanhdense/BiasAdd:output:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	 *
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          U
dense_1/Tensordot/ShapeShapedense/Tanh:y:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
: 
dense_1/Tensordot/transpose	Transposedense/Tanh:y:0!dense_1/Tensordot/concat:output:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc¢
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:©
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¢
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc r
dense_1/TanhTanhdense_1/BiasAdd:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout/dropout/MulMuldense_1/Tanh:y:0dropout/dropout/Const:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc U
dropout/dropout/ShapeShapedense_1/Tanh:y:0*
T0*
_output_shapes
:®
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc *
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ð
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0`
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          `
dense_2/Tensordot/ShapeShapedropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:a
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ª
dense_2/Tensordot/transpose	Transposedropout/dropout/Mul_1:z:0!dense_2/Tensordot/concat:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc ¢
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@a
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:©
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¢
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@r
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout_1/dropout/MulMuldense_2/Tanh:y:0 dropout_1/dropout/Const:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@W
dropout_1/dropout/ShapeShapedense_2/Tanh:y:0*
T0*
_output_shapes
:²
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ö
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0`
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          b
dense_3/Tensordot/ShapeShapedropout_1/dropout/Mul_1:z:0*
T0*
_output_shapes
:a
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¬
dense_3/Tensordot/transpose	Transposedropout_1/dropout/Mul_1:z:0!dense_3/Tensordot/concat:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@¢
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:©
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¢
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcr
dense_3/TanhTanhdense_3/BiasAdd:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          W
dense_4/Tensordot/ShapeShapedense_3/Tanh:y:0*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¡
dense_4/Tensordot/transpose	Transposedense_3/Tanh:y:0!dense_4/Tensordot/concat:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc¢
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:©
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¢
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcx
dense_4/SigmoidSigmoiddense_4/BiasAdd:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØct
IdentityIdentitydense_4/Sigmoid:y:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp:b ^
:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
 
_user_specified_nameinputs
ã

ó
*__inference_sequential_layer_call_fn_42555

inputs
unknown:
î
	unknown_0:	
	unknown_1:	 
	unknown_2: 
	unknown_3: @
	unknown_4:@
	unknown_5:@
	unknown_6:
	unknown_7:
	unknown_8:
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_42362
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
 
_user_specified_nameinputs

b
D__inference_dropout_1_layer_call_and_return_conditional_losses_42108

inputs

identity_1`
IdentityIdentityinputs*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@m

Identity_1IdentityIdentity:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@:a ]
9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
 
_user_specified_nameinputs

ú
B__inference_dense_1_layer_call_and_return_conditional_losses_42931

inputs4
!tensordot_readvariableop_resource:	 -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	 *
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc b
TanhTanhBiasAdd:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc i
IdentityIdentityTanh:y:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:b ^
:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
 
_user_specified_nameinputs

b
D__inference_dropout_1_layer_call_and_return_conditional_losses_43013

inputs

identity_1`
IdentityIdentityinputs*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@m

Identity_1IdentityIdentity:output:0*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@:a ]
9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
 
_user_specified_nameinputs
»$

E__inference_sequential_layer_call_and_return_conditional_losses_42362

inputs
dense_42334:
î
dense_42336:	 
dense_1_42339:	 
dense_1_42341: 
dense_2_42345: @
dense_2_42347:@
dense_3_42351:@
dense_3_42353:
dense_4_42356:
dense_4_42358:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCallô
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_42334dense_42336*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_42016
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_42339dense_1_42341*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_42053ù
dropout/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_42281
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_2_42345dense_2_42347*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_42097
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_42248
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_3_42351dense_3_42353*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_42141
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_42356dense_4_42358*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_42178
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc´
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:b ^
:
_output_shapes(
&:$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
 
_user_specified_nameinputs
¸
b
)__inference_dropout_1_layer_call_fn_43008

inputs
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_42248
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@22
StatefulPartitionedCallStatefulPartitionedCall:a ]
9
_output_shapes'
%:#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
 
_user_specified_nameinputs"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*×
serving_defaultÃ
V
dense_inputG
serving_default_dense_input:0$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcîM
dense_4B
StatefulPartitionedCall:0#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØctensorflow/serving/predict:êÑ
Ã
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias"
_tf_keras_layer
¼
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_random_generator"
_tf_keras_layer
»
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias"
_tf_keras_layer
¼
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6_random_generator"
_tf_keras_layer
»
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias"
_tf_keras_layer
»
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Ekernel
Fbias"
_tf_keras_layer
f
0
1
2
 3
.4
/5
=6
>7
E8
F9"
trackable_list_wrapper
f
0
1
2
 3
.4
/5
=6
>7
E8
F9"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ý
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_32ò
*__inference_sequential_layer_call_fn_42208
*__inference_sequential_layer_call_fn_42530
*__inference_sequential_layer_call_fn_42555
*__inference_sequential_layer_call_fn_42410¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zLtrace_0zMtrace_1zNtrace_2zOtrace_3
É
Ptrace_0
Qtrace_1
Rtrace_2
Strace_32Þ
E__inference_sequential_layer_call_and_return_conditional_losses_42696
E__inference_sequential_layer_call_and_return_conditional_losses_42851
E__inference_sequential_layer_call_and_return_conditional_losses_42441
E__inference_sequential_layer_call_and_return_conditional_losses_42472¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zPtrace_0zQtrace_1zRtrace_2zStrace_3
ÏBÌ
 __inference__wrapped_model_41978dense_input"
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
annotationsª *
 

Titer

Ubeta_1

Vbeta_2
	Wdecay
Xlearning_ratem§m¨m© mª.m«/m¬=m­>m®Em¯Fm°v±v²v³ v´.vµ/v¶=v·>v¸Ev¹Fvº"
	optimizer
,
Yserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
é
_trace_02Ì
%__inference_dense_layer_call_fn_42860¢
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
annotationsª *
 z_trace_0

`trace_02ç
@__inference_dense_layer_call_and_return_conditional_losses_42891¢
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
annotationsª *
 z`trace_0
 :
î2dense/kernel
:2
dense/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ë
ftrace_02Î
'__inference_dense_1_layer_call_fn_42900¢
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
annotationsª *
 zftrace_0

gtrace_02é
B__inference_dense_1_layer_call_and_return_conditional_losses_42931¢
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
annotationsª *
 zgtrace_0
!:	 2dense_1/kernel
: 2dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
¿
mtrace_0
ntrace_12
'__inference_dropout_layer_call_fn_42936
'__inference_dropout_layer_call_fn_42941³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zmtrace_0zntrace_1
õ
otrace_0
ptrace_12¾
B__inference_dropout_layer_call_and_return_conditional_losses_42946
B__inference_dropout_layer_call_and_return_conditional_losses_42958³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zotrace_0zptrace_1
"
_generic_user_object
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
ë
vtrace_02Î
'__inference_dense_2_layer_call_fn_42967¢
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
annotationsª *
 zvtrace_0

wtrace_02é
B__inference_dense_2_layer_call_and_return_conditional_losses_42998¢
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
annotationsª *
 zwtrace_0
 : @2dense_2/kernel
:@2dense_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
Ã
}trace_0
~trace_12
)__inference_dropout_1_layer_call_fn_43003
)__inference_dropout_1_layer_call_fn_43008³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z}trace_0z~trace_1
û
trace_0
trace_12Â
D__inference_dropout_1_layer_call_and_return_conditional_losses_43013
D__inference_dropout_1_layer_call_and_return_conditional_losses_43025³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
"
_generic_user_object
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
í
trace_02Î
'__inference_dense_3_layer_call_fn_43034¢
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
annotationsª *
 ztrace_0

trace_02é
B__inference_dense_3_layer_call_and_return_conditional_losses_43065¢
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
annotationsª *
 ztrace_0
 :@2dense_3/kernel
:2dense_3/bias
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
í
trace_02Î
'__inference_dense_4_layer_call_fn_43074¢
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
annotationsª *
 ztrace_0

trace_02é
B__inference_dense_4_layer_call_and_return_conditional_losses_43105¢
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
annotationsª *
 ztrace_0
 :2dense_4/kernel
:2dense_4/bias
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Bý
*__inference_sequential_layer_call_fn_42208dense_input"¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
*__inference_sequential_layer_call_fn_42530inputs"¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
*__inference_sequential_layer_call_fn_42555inputs"¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bý
*__inference_sequential_layer_call_fn_42410dense_input"¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_42696inputs"¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_42851inputs"¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_42441dense_input"¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_42472dense_input"¿
¶²²
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÎBË
#__inference_signature_wrapper_42505dense_input"
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
annotationsª *
 
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
ÙBÖ
%__inference_dense_layer_call_fn_42860inputs"¢
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
annotationsª *
 
ôBñ
@__inference_dense_layer_call_and_return_conditional_losses_42891inputs"¢
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
annotationsª *
 
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
ÛBØ
'__inference_dense_1_layer_call_fn_42900inputs"¢
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
annotationsª *
 
öBó
B__inference_dense_1_layer_call_and_return_conditional_losses_42931inputs"¢
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
annotationsª *
 
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
ìBé
'__inference_dropout_layer_call_fn_42936inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ìBé
'__inference_dropout_layer_call_fn_42941inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
B__inference_dropout_layer_call_and_return_conditional_losses_42946inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
B__inference_dropout_layer_call_and_return_conditional_losses_42958inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ÛBØ
'__inference_dense_2_layer_call_fn_42967inputs"¢
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
annotationsª *
 
öBó
B__inference_dense_2_layer_call_and_return_conditional_losses_42998inputs"¢
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
annotationsª *
 
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
îBë
)__inference_dropout_1_layer_call_fn_43003inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
îBë
)__inference_dropout_1_layer_call_fn_43008inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
D__inference_dropout_1_layer_call_and_return_conditional_losses_43013inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
D__inference_dropout_1_layer_call_and_return_conditional_losses_43025inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ÛBØ
'__inference_dense_3_layer_call_fn_43034inputs"¢
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
annotationsª *
 
öBó
B__inference_dense_3_layer_call_and_return_conditional_losses_43065inputs"¢
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
annotationsª *
 
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
ÛBØ
'__inference_dense_4_layer_call_fn_43074inputs"¢
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
annotationsª *
 
öBó
B__inference_dense_4_layer_call_and_return_conditional_losses_43105inputs"¢
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
annotationsª *
 
R
	variables
	keras_api

total

count"
_tf_keras_metric

	variables
	keras_api
true_positives
true_negatives
false_positives
false_negatives"
_tf_keras_metric
v
	variables
	keras_api

thresholds
 true_positives
¡false_negatives"
_tf_keras_metric
v
¢	variables
£	keras_api
¤
thresholds
¥true_positives
¦false_positives"
_tf_keras_metric
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
@
0
1
2
3"
trackable_list_wrapper
.
	variables"
_generic_user_object
:È (2true_positives
:È (2true_negatives
 :È (2false_positives
 :È (2false_negatives
0
 0
¡1"
trackable_list_wrapper
.
	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
¥0
¦1"
trackable_list_wrapper
.
¢	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
%:#
î2Adam/dense/kernel/m
:2Adam/dense/bias/m
&:$	 2Adam/dense_1/kernel/m
: 2Adam/dense_1/bias/m
%:# @2Adam/dense_2/kernel/m
:@2Adam/dense_2/bias/m
%:#@2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
%:#2Adam/dense_4/kernel/m
:2Adam/dense_4/bias/m
%:#
î2Adam/dense/kernel/v
:2Adam/dense/bias/v
&:$	 2Adam/dense_1/kernel/v
: 2Adam/dense_1/bias/v
%:# @2Adam/dense_2/kernel/v
:@2Adam/dense_2/bias/v
%:#@2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
%:#2Adam/dense_4/kernel/v
:2Adam/dense_4/bias/v¿
 __inference__wrapped_model_41978
 ./=>EFG¢D
=¢:
85
dense_input$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
ª "Cª@
>
dense_430
dense_4#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcÈ
B__inference_dense_1_layer_call_and_return_conditional_losses_42931 B¢?
8¢5
30
inputs$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
ª "7¢4
-*
0#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
 
'__inference_dense_1_layer_call_fn_42900t B¢?
8¢5
30
inputs$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
ª "*'#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc Ç
B__inference_dense_2_layer_call_and_return_conditional_losses_42998./A¢>
7¢4
2/
inputs#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
ª "7¢4
-*
0#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
 
'__inference_dense_2_layer_call_fn_42967s./A¢>
7¢4
2/
inputs#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
ª "*'#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@Ç
B__inference_dense_3_layer_call_and_return_conditional_losses_43065=>A¢>
7¢4
2/
inputs#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
ª "7¢4
-*
0#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
 
'__inference_dense_3_layer_call_fn_43034s=>A¢>
7¢4
2/
inputs#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
ª "*'#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcÇ
B__inference_dense_4_layer_call_and_return_conditional_losses_43105EFA¢>
7¢4
2/
inputs#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
ª "7¢4
-*
0#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
 
'__inference_dense_4_layer_call_fn_43074sEFA¢>
7¢4
2/
inputs#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
ª "*'#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcÇ
@__inference_dense_layer_call_and_return_conditional_losses_42891B¢?
8¢5
30
inputs$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
ª "8¢5
.+
0$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
 
%__inference_dense_layer_call_fn_42860uB¢?
8¢5
30
inputs$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
ª "+($ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcÉ
D__inference_dropout_1_layer_call_and_return_conditional_losses_43013E¢B
;¢8
2/
inputs#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
p 
ª "7¢4
-*
0#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
 É
D__inference_dropout_1_layer_call_and_return_conditional_losses_43025E¢B
;¢8
2/
inputs#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
p
ª "7¢4
-*
0#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
  
)__inference_dropout_1_layer_call_fn_43003sE¢B
;¢8
2/
inputs#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
p 
ª "*'#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@ 
)__inference_dropout_1_layer_call_fn_43008sE¢B
;¢8
2/
inputs#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@
p
ª "*'#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc@Ç
B__inference_dropout_layer_call_and_return_conditional_losses_42946E¢B
;¢8
2/
inputs#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
p 
ª "7¢4
-*
0#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
 Ç
B__inference_dropout_layer_call_and_return_conditional_losses_42958E¢B
;¢8
2/
inputs#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
p
ª "7¢4
-*
0#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
 
'__inference_dropout_layer_call_fn_42936sE¢B
;¢8
2/
inputs#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
p 
ª "*'#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
'__inference_dropout_layer_call_fn_42941sE¢B
;¢8
2/
inputs#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc 
p
ª "*'#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc à
E__inference_sequential_layer_call_and_return_conditional_losses_42441
 ./=>EFO¢L
E¢B
85
dense_input$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
p 

 
ª "7¢4
-*
0#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
 à
E__inference_sequential_layer_call_and_return_conditional_losses_42472
 ./=>EFO¢L
E¢B
85
dense_input$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
p

 
ª "7¢4
-*
0#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
 Û
E__inference_sequential_layer_call_and_return_conditional_losses_42696
 ./=>EFJ¢G
@¢=
30
inputs$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
p 

 
ª "7¢4
-*
0#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
 Û
E__inference_sequential_layer_call_and_return_conditional_losses_42851
 ./=>EFJ¢G
@¢=
30
inputs$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
p

 
ª "7¢4
-*
0#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc
 ¸
*__inference_sequential_layer_call_fn_42208
 ./=>EFO¢L
E¢B
85
dense_input$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
p 

 
ª "*'#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc¸
*__inference_sequential_layer_call_fn_42410
 ./=>EFO¢L
E¢B
85
dense_input$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
p

 
ª "*'#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc³
*__inference_sequential_layer_call_fn_42530
 ./=>EFJ¢G
@¢=
30
inputs$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
p 

 
ª "*'#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc³
*__inference_sequential_layer_call_fn_42555
 ./=>EFJ¢G
@¢=
30
inputs$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî
p

 
ª "*'#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcÑ
#__inference_signature_wrapper_42505©
 ./=>EFV¢S
¢ 
LªI
G
dense_input85
dense_input$ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØcî"Cª@
>
dense_430
dense_4#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØc