import importlib

print("trying torch._C")
mod = importlib.import_module("torch._C")
print(mod, mod.__file__)
print(dir(mod))


print("trying torch._C._nn")
mod = importlib.import_module("torch._C._nn")
print(mod, mod.__package__)
print(dir(mod))

