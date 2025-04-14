from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
#Question 1

# Parameters
n = 2           # Number of trials (rolls)
p = 0.5         # Probability of success (rolling 4, 5, or 6)

# Binomial PMF
x = [0, 1, 2]
pmf_vals = binom.pmf(x, n, p)

# Plotting
plt.bar(x, pmf_vals, tick_label=x, color='skyblue')
plt.title('Binomial Distribution: P(X successes in 2 dice rolls)')
plt.xlabel('Number of successes (rolling 4 or more)')
plt.ylabel('Probability')
plt.grid(axis='y')
plt.show()

# Printing probabilities
for i in x:
    print(f"P(X = {i}) = {binom.pmf(i, n, p):.3f}")


#Question 2
from scipy.stats import expon

λ = 1/5
x = np.linspace(0, 5, 100)
y = expon.pdf(x, scale=1/λ)

plt.plot(x, y, color='red')
plt.fill_between(x, y, alpha=0.4)
plt.xlabel('Time')
plt.ylabel('Density')
plt.title('Exponential Distribution (λ=1/5)')
plt.plot(x, y, 'r-', lw=2, label='Exponential PDF')
plt.title('Rainfall Modeled as Exponential Distribution')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# Simulated daily rainfall data: assume average rainfall is 5 mm
lambda_param = 1/5  # Mean = 1/lambda
size = 1000

# Generate synthetic rainfall data
rainfall = np.random.exponential(scale=1/lambda_param, size=size)

# Plot histogram and PDF
x = np.linspace(0, 30, 1000)
pdf = expon.pdf(x, scale=1/lambda_param)

plt.figure(figsize=(8, 4))
plt.hist(rainfall, bins=30, density=True, alpha=0.6, color='skyblue', label='Simulated Rainfall')
plt.plot(x, pdf, 'r-', lw=2, label='Exponential PDF')
plt.title('Rainfall Modeled as Exponential Distribution')
plt.xlabel('Rainfall (mm)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()


#Question 3
# y =poisson.pmf(k=5, mu=5)
# print(y)

λ = 5  # Average rate of occurrences
x = np.arange(0, 5)
y = poisson.pmf(x, λ)

plt.bar(x, y, color='orange')
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.title('Poisson Distribution for Calls(λ=5)')
plt.grid(True)
plt.show()

#Question 4

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, weibull_min

# Simulate light bulb lifespans
np.random.seed(42)

# Exponential model (mean lifespan = 1000 hours)
exp_mean = 1000
lifespan_exp = np.random.exponential(scale=exp_mean, size=1000)

# Plotting
plt.figure(figsize=(10, 5))

# Histogram for Exponential
plt.hist(lifespan_exp, bins=30, density=True, alpha=0.5, label='Exponential Lifespans (Simulated)', color='skyblue')

# PDFs
x = np.linspace(0, 4000, 500)
plt.plot(x, expon.pdf(x, scale=exp_mean), 'b--', label='Exponential PDF')

plt.title("Modeling Light Bulb Lifespan with Exponential and Weibull Distributions")
plt.xlabel("Lifespan (hours)")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
