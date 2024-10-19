import numpy as np
import sympy as sym
import json
import sympy as sym
from pydae.bmapu import bmapu_builder

grid = bmapu_builder.bmapu('k12p6.hjson')
grid.checker()
grid.uz_jacs = True
grid.build('k12p6')