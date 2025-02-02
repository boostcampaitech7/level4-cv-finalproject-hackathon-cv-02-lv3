import folium
from folium.plugins import MarkerCluster
import alphashape


# 필요한 라이브러리 다시 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# 데이터 불러오기
file_path = "/data/ephemeral/home/Jeongseon/melb_split.csv"
df = pd.read_csv(file_path)

# 불필요한 컬럼 제거
drop_tables = ['Suburb', 'Address', 'Rooms', 'Method', 'SellerG', 'Date', 'Distance', 'Postcode',
               'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'YearBuilt', 'CouncilArea',
               'Regionname', 'Propertycount', 'Split']
df = df.drop(drop_tables, axis=1, errors='ignore')  # 존재하지 않는 컬럼이 있어도 오류 방지
df = df.dropna(axis=0)

# BuildingArea 0값 제거
df = df[df['BuildingArea'] > 0.1]

# 원핫 인코딩
train_data = pd.get_dummies(df, dtype='float')

# 타겟 변수와 특성 분리
y_train = train_data['Price']
X_train = train_data.drop(['Price'], axis=1)

# 위도, 경도의 유니크한 조합 가져오기
lat_lon_unique = X_train[['Lattitude', 'Longtitude']].drop_duplicates().values

# Convex Hull 계산
#hull = ConvexHull(lat_lon_unique)

# Convex Hull의 경계 좌표
#polygon_coords = lat_lon_unique[hull.vertices]

# 시각화
#plt.figure(figsize=(10, 6))
#plt.scatter(X_train['Longtitude'], X_train['Lattitude'], c='green', s=5, label="Interior Points")  # 내부 점 초록색
#plt.scatter(polygon_coords[:, 1], polygon_coords[:, 0], c='red', s=30, label="Convex Hull Points")  # 외각선 점 빨간색

# Convex Hull 외곽선 그리기
#for simplex in hull.simplices:
#    plt.plot(lat_lon_unique[simplex, 1], lat_lon_unique[simplex, 0], 'r-', linewidth=1.5)

#plt.xlabel("Longitude")
#plt.ylabel("Latitude")
#plt.title("Convex Hull Visualization")
#plt.legend()
#plt.show()




# 중심 좌표 설정 (데이터의 평균값)
# center_lat = X_train['Lattitude'].mean()
# center_lon = X_train['Longtitude'].mean()

# # 지도 생성
# m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

# # Convex Hull 외곽선(빨간색)
# for i in range(len(polygon_coords)):
#     folium.CircleMarker(
#         location=[polygon_coords[i][0], polygon_coords[i][1]],
#         radius=5,
#         color='red',
#         fill=True,
#         fill_color='red',
#         fill_opacity=0.8,
#         popup=f"Convex Hull Point {i}"
#     ).add_to(m)

# # 내부 데이터(초록색)
# marker_cluster = MarkerCluster().add_to(m)
# for i in range(len(X_train)):
#     folium.CircleMarker(
#         location=[X_train.iloc[i]['Lattitude'], X_train.iloc[i]['Longtitude']],
#         radius=2,
#         color='green',
#         fill=True,
#         fill_color='green',
#         fill_opacity=0.6
#     ).add_to(marker_cluster)

# # 지도 저장
# map_path = "/data/ephemeral/home/Jeongseon/melb_convex_hull_map.html"
# m.save(map_path)
# map_path

###################################################
# Alpha Shape을 사용한 다각형 생성
# alpha_value = 0.000001  # alpha 값을 조정해서 다각형의 세밀도를 조절합니다.
# alpha_shape = alphashape.alphashape(lat_lon_unique, alpha_value)


# # Alpha Shape의 외곽선 좌표를 리스트로 추출
# if alpha_shape.geom_type == "Polygon":
#     alpha_shape_coords = list(alpha_shape.exterior.coords)  # (경도, 위도) 순서로 반환됨
#     alpha_shape_coords = [(lat, lon) for lon, lat in alpha_shape_coords]  # 위도, 경도 순서로 변환
#     print("📍 Alpha Shape 외곽선 좌표 리스트:", alpha_shape_coords)
# else:
#     print("Alpha Shape이 다각형이 아닙니다. Alpha 값을 조정하세요.")
#########################################################

import geopandas as gpd
from shapely.geometry import MultiPoint, LineString
from shapely.ops import unary_union, polygonize

# Concave Hull 생성
points = MultiPoint(lat_lon_unique)

# Convex Hull의 외곽선을 LineString으로 변환
convex_hull_line = LineString(points.convex_hull.exterior.coords)

# Convex Hull을 기반으로 Concave Hull 생성
#concave_hull = unary_union(polygonize([convex_hull_line]))
concave_hull = unary_union(list(polygonize([convex_hull_line])))

# 시각화
plt.figure(figsize=(10, 6))

# 내부 데이터(초록색)
plt.scatter(X_train['Longtitude'], X_train['Lattitude'], c='green', s=5, label="Interior Points")

# Concave Hull 외곽선(빨간색)
if concave_hull.geom_type == "Polygon":
    concave_hull_coords = list(concave_hull.exterior.coords)
    concave_hull_coords = [(lat, lon) for lon, lat in concave_hull_coords]
    
    # 좌표를 분리하여 플롯
    hull_lats, hull_lons = zip(*concave_hull_coords)
    plt.plot(hull_lons, hull_lats, 'r-', linewidth=1.5, label="Concave Hull")

# 그래프 설정
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Concave Hull Visualization")
plt.legend()
plt.show()


# Concave Hull의 좌표 리스트로 변환
if concave_hull.geom_type == "Polygon":
    concave_hull_coords = list(concave_hull.exterior.coords)
    concave_hull_coords = [(lat, lon) for lon, lat in concave_hull_coords]
    print(f"📍 Concave Hull 외곽선 좌표 ({len(concave_hull_coords)}각형):", concave_hull_coords)
else:
    print("Concave Hull이 다각형이 아닙니다. 파라미터를 조정하세요.")

# # Folium 지도 생성
# center_lat = X_train['Lattitude'].mean()
# center_lon = X_train['Longtitude'].mean()

# #지도 생성
# m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

# # Alpha Shape 외곽선 그리기
# folium.GeoJson(alpha_shape.__geo_interface__, name='Alpha Shape').add_to(m)

# # 내부 데이터 점 표시
# marker_cluster = folium.plugins.MarkerCluster().add_to(m)
# for i in range(len(X_train)):
#     folium.CircleMarker(
#         location=[X_train.iloc[i]['Lattitude'], X_train.iloc[i]['Longtitude']],
#         radius=2,
#         color='green',
#         fill=True,
#         fill_color='green',
#         fill_opacity=0.6
#     ).add_to(marker_cluster)

# # 지도 저장
# alpha_shape_map_path = "/data/ephemeral/home/Jeongseon/melb_alpha_shape_map_3.html"
# m.save(alpha_shape_map_path)
# alpha_shape_map_path