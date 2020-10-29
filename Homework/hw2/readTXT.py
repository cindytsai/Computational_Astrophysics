import pandas as pd

# data = pd.read_csv('strong-shock.txt', skiprows=(0, 1, 2, 3, 4, 5), sep='\s{2,}', engine='python')
# print(data)
# print(data["r"])

headerList = ["r", "Rho", "Vx", "Vy", "Vz", "Pres"]
data = pd.read_csv('strong-shock.txt', skiprows=(0, 1, 2, 3, 4, 5), comment='#', names=headerList, sep='\s{2,}', engine='python')
print(data)
