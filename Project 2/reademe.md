# Evolutionary Algorithms

### Colaborators:
- Bruno Cebola
- Rui Abrantes
---
## How to Run the Program

Since the project had some specific demands on the output data we created two ways of starting the program.
The first way is by typing the following in the terminal:

```
$ python3 project2.py config.yaml
```
with this the program will read the config.yaml, and using its parameteres it will run 1 study case for the Single Objective Problem, or 1 study case for the Multi Objetive Problem.

To have the desired study case is pretty simple:
- dist_file = file that contains the matrix of distances
- coord_file = file that contains the coordenates of customers and warehouse
- orders_file = file that contains all the orders of each customer


This variables are always needed in the config since they are a core of the problem

For trying new study cases we have:
- nb_customers = number of customers to travel to
- heuristic = if we want to run the heuristic
- multi_objetive = if we want to run the multi objetive problem

Now in the example config.yaml there will be 2 more variable commented

- fill_orders
- orders: 50

If fill orders is not commented, instead of using the orders_file, the program will create the number of orders equal to the orders of the "orders" variable, in this example it is pre-defined 50, because that's what we were asked to test.

Now, to test all the study cases of the single problem we just need to run in the terminal the following:

```
$ python3 project2.py --all
```

But there is a nuance, since there were 3 groups of study case, 10,30 and 50 customers, when running the program in this way, it will show the graphs of the evolutions, for the program to keep running it is necessary to close the graph.

The folder "configs" will have all the tests done, so if it is necessary to test the program with or without the heuristic, it is necessary to change it value in each file, but since it is only wanted to run those scenarios once, it is only necessary to change the 12 files once or twice