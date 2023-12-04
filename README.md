# OOP - 2023/24 - Assignment 1

This is the base repository for assignment 1.
Please follow the instructions given in the [PDF](https://brightspace.rug.nl/content/enforced/243046-WBAI045-05.2023-2024.1/2023_24_OOP.pdf) for the content of the exercise.

## How to carry out your assignment

Fork this repo on your private github account.
You can do so by clicking this button on the top-right panel:
![](fork.png) 

The assignment is divided into 4 blocks.
Block 1, 2, and 3 all define different classes.

Put the three classes in three separate files in the `src` folder, with the names specified in the PDF.
**Leave the __init__.py file untouched**.

Put the **main.py** script **outside** of the `src` folder, in the root of this repo.

Below this line, you can write your report to motivate your design choices.

## Submission

The code should be submitted on GitHub by opening a Pull Request from the branch you were working on to the `submission` branch.

There are automated checks that verify that your submission is correct:

1. Deadline - checks that the last commit in a PR was made before the deadline
2. Reproducibility - downloads libraries included in `requirements.txt` and runs `python3 src/main.py`. If your code does not throw any errors, it will be marked as reproducible.
3. Style - runs `flake8` on your code to ensure adherence to style guides.

---

## Your report
The class MultipleLinearRegression is set up with four different functions, of which one is called automatically when creating a new instance of the MultipleLinearRegression. This class uses encapsulation by having one private variable, called _parameters. This variable is private, as this should not be changed manually by the user. The user can get the parameters with the defined function **get_parameters()**. 
The other two functions in this class are **train()** and **predict**. These functions are both public, as they need to be called by the user to make use of the class. \\

The class RegressionPlotter is an independent class that contains four functions, of which one is called automatically when crating a new instance of the RegressionPlotter. This function stores the variables used across the three functions. All of these values are private, as they must not be found or manipulated manually by the user. The only public function in this class is the **plot()** function. This function decides whether the plot must be 3D or 2D, depending on how many features should be plotted. Based on that, it calls either **_plot_2D()** or **_plot_3D()**. These are both private functions, as they must only be called via the public **plot()** function.
