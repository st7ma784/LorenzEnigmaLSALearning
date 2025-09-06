Hi Claude, 

I want you to think about the following idea: 

I'm trying to find a way to differentiate a graient between permutation matrices based on a loss function (I.E cost minimization?)

One of the toy problems for this would be solving the enigma code - by learning the permutation matrices of each rotor. 

I've tried a few other approaches to limited success, but I recently came across an idea. 

Enigma and lorenz are 2 different code approaches:
Lorenz views the coded text as the source + a cipher. adding the 2 as a modulo-2 arithmetic mask. The mask could be equally generated on the recieve end too. 

I want you to model an engima code (3 rotors, with unique starting positions, and new mappings to the alphabet). Try and express the encoded text a lorenz encryption, by subtracting the source from the encryption to get the lorenz mask. 
 
I want you to define the statistical relation between the mask and the rotor settings and rotor positions. I want to find out if this is a good key to differentiating permutation matrices. please use many different statistical methods. You may find it useful to use tricks like the relation with magic squares and permutation matrices, to generate double-stochastic 2d matrices per rotor config prior to learning the relation between the rotors (so 4 x 26x26) and the lorenz cipher. 

When you have a solution that works accurately, build a web visualisation of how the lorenz cipher text relates to the rotor and whether this can support gradient based ml operations. 


