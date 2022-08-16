from latex2sympy2 import latex2sympy, latex2latex

tex = r"(2\nu\tau + 1 -\tau - \nu)^{2}(2\nu(\nu -1) + 1))"
print(latex2sympy(tex))
print(latex2latex(tex))
