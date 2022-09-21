from latex2sympy2 import latex2sympy

tex = r"1/2 + 0.48785 \cos \left(\theta +\epsilon  \sin ^2\left(\frac{\theta }{2}\right)\right) \left((x+y) \cos \left(\frac{1}{2} (2 \theta +\epsilon  (-\cos (\theta ))+\epsilon )\right)-x-y+1\right)+0.109555 \sin \left(\theta +\epsilon  \sin ^2\left(\frac{\theta }{2}\right)\right) \left((y+z) \cos \left(\frac{1}{2} (2 \theta +\epsilon  (-\cos (\theta ))+\epsilon )\right)-y-z+1\right)+\frac{1}{2}"
print(latex2sympy(tex))
