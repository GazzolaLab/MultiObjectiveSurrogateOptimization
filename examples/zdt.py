from machinable import get

x = [10, 50, 100, 200, 300]
y = []
y2 = []

for population_size in x:
    zdt = get("interface.zdt", {"population_size": population_size}).launch()
    hv = zdt.hypervolume([11,11])
    igd = zdt.igd([11,11])
    y.append(hv)
    y2.append(igd)

print(x)
print(y)
print(y2)
