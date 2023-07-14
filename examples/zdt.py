from machinable import get

x = [10, 50, 100, 200, 300]
y = []

for population_size in x:
    zdt = get("interface.zdt", {"population_size": population_size}).launch()
    hv = zdt.hypervolume()

    y.append(hv)

print(x)
print(y)
