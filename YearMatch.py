import pandas as pd

# read by default 1st sheet of an excel file
yearLookUp = pd.read_excel("YearLookUp.xlsx")
# print(yearLookUp)

coord = pd.read_excel("rawMainOutput.xlsx")
coord = coord.assign(Year=["nan" for i in range(len(coord))])
print(coord)
for i in range(len(yearLookUp)):
    name = yearLookUp["Street"][i]
    # if any(coord[coord["Street"] == name]):
    if len(coord.index[coord["Street"] == name]) > 0:
        coord["Year"][coord.index[coord["Street"] == name][0]] = yearLookUp["Year"][i]
    # break

print(coord)

coord.to_excel("XLPlotInput.xlsx", index=False)
