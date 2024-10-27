Engine Goals:

# search spaces

1. search feature stream space - levelage multiple streams (for interplay and predictive power)
2. search the feature transformation space - permutation of pipeline components
3. search the meta parameter space
4. search the hyperparameter space

# meta searching

1. meta modeling - learning how to learn - multiple models comparison, etc.

# example discussion of multi-feature approach

Z GOLD
A SILVER
B GOOG
a predictors
Ap1
ap2
ap3
ap4

## cpu engine?

## model of world data + prediction

Z | A | B
---------

z | a | b
z | a | b
z | a | b
z | a | b
? | (ap1 +ap2 +ap3 +ap4)/4 | ensambling avg b predictors
(dependant model)

## deep learning approach - all in one model

## ensamble and model in one

Z | A | Ap1 | Ap2 | ... | B | ... |
-----------------------------------

z | a | ap1 | ap2 | ... | b | ... |
z | a | ap1 | ap2 | ... | b | ... |
z | a | ap1 | ap2 | ... | b | ... |
z | a | ap1 | ap2 | ... | b | ... |
? | ? | ap1 | ap2 | ... | ? | ... |
(weights for ensamble and model that ensamble uses are all trained together)
