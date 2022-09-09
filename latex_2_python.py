from latex2sympy2 import latex2sympy

tex = r"1  - cos(\theta + \epsilon) cos(\nu)(2(sin^{2}(\frac{\theta + \epsilon}{2})(k_{X\theta} + k_{Y\theta}) + sin^{2}(\frac{\phi + \mu}{2})(k_{X\phi} + k_{Y\phi})) - 1)"
print(latex2sympy(tex))
