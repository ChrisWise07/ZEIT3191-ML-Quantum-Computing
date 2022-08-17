from latex2sympy2 import latex2sympy, latex2latex

tex = r"1 + 2\nu\tau\left(\nu\tau + \sqrt{(\nu^2 - 1)(\tau^2 -1)}\right) - \tau^2 - \nu^2"
print(latex2sympy(tex))
