from lxml import etree
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
import xlsxwriter


def representationPoint(points):
    if len(points) == 2:
        return ((points[0, 0] + points[1, 0]) / 2, (points[0, 1] + points[1, 1]) / 2)
    hull = ConvexHull(points)
    plt.scatter(points[:, 0], points[:, 1])
    hullList = hull.points[np.concatenate((hull.vertices, [(hull.vertices)[0]]))]
    # print(hullList)
    n = len(hullList)
    Xacc = 0
    Yacc = 0
    area = 0
    for i in range(n):
        if i == n - 1:
            (xi, yi) = (hullList[i, 0], hullList[i, 1])
            (xii, yii) = (hullList[0, 0], hullList[0, 1])
        else:
            (xi, yi) = (hullList[i, 0], hullList[i, 1])
            (xii, yii) = (hullList[i + 1, 0], hullList[i + 1, 1])

        Xacc += (xi + xii) * (xi * yii - xii * yi)
        Yacc += (yi + yii) * (xi * yii - xii * yi)
        area += (xi * yii - xii * yi) / 2
    centroid = (Xacc / (6 * area), Yacc / (6 * area))
    if Xacc <= 1e-7 or Yacc <= 1e-7 or area <= 1e-9:
        centroid = (np.mean(points[:, 0]), np.mean(points[:, 1]))
    # print(Xacc, Yacc, area, centroid)
    m = len(points)
    fstDist = 1e9
    fstIdx = -1
    for i in range(m):
        dist = (points[i, 0] - centroid[0]) ** 2 + (points[i, 1] - centroid[1]) ** 2
        if dist < fstDist:
            fstDist = dist
            fstIdx = i

    tmp1 = (fstIdx + 1 + m) % m
    tmp2 = (fstIdx - 1 + m) % m
    distTmp1 = (points[tmp1, 0] - centroid[0]) ** 2 + (
        points[tmp1, 1] - centroid[1]
    ) ** 2
    distTmp2 = (points[tmp2, 0] - centroid[0]) ** 2 + (
        points[tmp2, 1] - centroid[1]
    ) ** 2
    if distTmp1 <= distTmp2:
        sndIdx = tmp1
    else:
        sndIdx = tmp2
    # return centroid
    if (
        (points[fstIdx, 0] - points[sndIdx, 0]) ** 2
        + (points[sndIdx, 1] - points[fstIdx, 1]) ** 2
    ) == 0:
        return (
            (points[fstIdx, 0] + points[sndIdx, 0]) / 2,
            (points[fstIdx, 1] + points[sndIdx, 1]) / 2,
        )
    t = (
        (centroid[0] - points[fstIdx, 0]) * (points[sndIdx, 0] - points[fstIdx, 0])
        + (centroid[1] - points[fstIdx, 1]) * (points[sndIdx, 1] - points[fstIdx, 1])
    ) / (
        (points[fstIdx, 0] - points[sndIdx, 0]) ** 2
        + (points[sndIdx, 1] - points[fstIdx, 1]) ** 2
    )
    return (
        points[fstIdx, 0] + t * (points[sndIdx, 0] - points[fstIdx, 0]),
        points[fstIdx, 1] + t * (points[sndIdx, 1] - points[fstIdx, 1]),
    )

    if fstDist < midDist:
        print("OMG")
        return (points[fstIdx, 0], points[fstIdx, 1])
    else:
        return midpoint

    # return (Xacc / (6 * area), Yacc / (6 * area))


nodeCordDict = {}
streetDict = {}

tree = etree.parse(r"HanoiXL.osm")
root = tree.getroot()

minLat = 20.9614
maxLat = 21.0956
minLon = 105.7764
maxLon = 105.9415
nodesList = root.findall("node[@id]")
for node in nodesList:
    nodeAtt = node.attrib
    nodeCordDict[nodeAtt["id"]] = (float(nodeAtt["lat"]), float(nodeAtt["lon"]))

print(maxLat, minLat, maxLon, minLon)
origin = ((maxLat + minLat) / 2, (maxLon + minLon) / 2)
scaling = max(1 / (maxLat - minLat), 1 / (maxLon - minLon))

print(origin, " ", scaling)
waysList = root.findall("way")

for way in waysList:
    highway = way.findall('tag[@k = "highway"]')
    if len(highway) == 0:
        continue
    name = way.findall('tag[@k = "name"]')
    if len(name) == 0:
        continue

    nameVal = name[0].attrib["v"]
    if not (nameVal.startswith("Đường") or nameVal.startswith("Phố")):
        continue
    # nodes = way.findall("nd")
    Clat = 0
    Clon = 0
    area = 0
    if nameVal.startswith("Đường"):
        shortNameVal = nameVal[6:]
    if nameVal.startswith("Phố"):
        shortNameVal = nameVal[4:]
    if not (shortNameVal in streetDict):
        nodes = root.xpath('way/nd[../tag[@k="name" and @v = "' + nameVal + '"]]/@ref')
        # nodes = root.xpath('way/nd[../tag[@k="name" and @v = "Phố Nguyễn Xí"]]/@ref')
        if len(nodes) > 0:
            points = np.zeros((len(nodes), 2))
            for idx, node in enumerate(nodes):
                points[idx, 0] = nodeCordDict[node][0]
                points[idx, 1] = nodeCordDict[node][1]
            print(nameVal, end="")
            streetDict[shortNameVal] = representationPoint(points)
            print(streetDict[shortNameVal])
            # break

workbook = xlsxwriter.Workbook("rawMainOutput.xlsx")
worksheet = workbook.add_worksheet()
worksheet.write_row("A{}".format(1), ["Street", "X", "Y"])
line = 2
for street in streetDict.items():
    # print(street[0], " ", street[1][0], " ", street[1][1])
    name = street[0]
    lat = float(street[1][0])
    lon = float(street[1][1])
    plotX = scaling * (lon - origin[1])
    plotY = scaling * (lat - origin[0])
    worksheet.write_row("A{}".format(line), [name, plotX, plotY])
    line += 1
    # plt.scatter(plotX, plotY)
    # plt.text(plotX, plotY, name)
workbook.close()
# ax = plt.gca()
# ax.set_aspect("equal", adjustable="box")
# plt.show()
