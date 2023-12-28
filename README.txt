step1: Create a new virtual environment for python (we can also use conda default "base" environment). Example: conda activate base or python -m venv <virtual_environment_name>
step 1.1: If the virtual environment is created using python -m venv <virtual_environment_name>, please activate it using the command "source assignment2_env/bin/activate" (for linux based systems)
step2: Install all the requirements from "requirement.txt". Command: pip install -r requirements.txt
step3: Run the main file that is Main.py. Command: python Main.py
step4: We can validate the plots generate "initial_plot.png" and "final_plot.png"
step5: We can also validate the output on the terminal which I have copies below. We can validate the same in output.txt
===========================================================
Initial cost_function value:  0.6931471805599453
===========================================================
===========================================================
Optimal Parameters: 
===========================================================
optimal cost function:  0.2034977015894438
Optimal theta values:  [-25.16133284   0.2062317    0.2014716 ]
Total number of iterations to reach the optimal value:  23
===========================================================
The accuracy of the model in percentage is 89.0
===========================================================
====================================================================================================================================
Excepted probability of getting admission for a student with an Exam 1 score of 45 and an Exam 2 score of 85:  [0.77629072]
====================================================================================================================================