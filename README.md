# car_accident_prediction

## mutable vs immutable
- mutable : 값이 변한다 (List, Dict)
- immutable : 값이 변하지 않는다. (Number, String)
- copy() : 얕은 복사한 객체의 값 중 mutable 변수를 변경하면 기존 객체도 변경

## MCLP(Maximal Covering Location Problem)

### 정의: MCLP는 최대지역커버문제로, 설비가 커버하는 수요 (covered demand)의 합을 최대화 하면서 주어진 K개의 설비를 세울 위치를 선정하는 문제

### 가정
- 설비의 위치가 수요 발생 지점으로부터 일정 거리 Residual 이내에 수요를 커버함.
- 이때 거리 Residual은 커버리지 거리(covered distance) 라고 함.
- 커버되지 못한 수여는 서비스를 받지 못하는 수요가 아니라 서비스를 받긴 하지만 서비스 받는 설비로 부터의 거리가 커버리지 밖에 있어 만족할 만한 서비스 수준을 제공받지 못하는 수요를 의미

### Mathematical statement
- i : 수요 포인트 index
- j : 설비 후보지역 index
- I : 수요 포인트 집합
- J : 설비 후보지역 집합
- K : 총 설치해야 하는 설비 개수
- x : 설비 후보 지역 중 위치 j에 설비가 설치되면 1, 그렇지 않으면 0
- y : 적어도 하나의 설비로 그 포인트가 커버가 되면 1, 그렇지 않으면 0

### Formulation
 \begin{align} &\text{maximize} \sum_{i\in I} w_i y_i ...(1) \ \text{s.t.} \quad & y_i \le \sum_{j\in N_i}x_j \qquad for \quad all \quad i\in I ... (2)\ &\sum_{j\in J}x_j = K ... (3)\ &x_j, y_i \in {0,1} \qquad for \quad all \quad i\in I,j\in J \end{align}
 
-(1) : 목적함수, 가중치 w인 수요 포인트를 최대한 많이 커버하게 해라
-(2) : 수요포인트 i는 설비 후보 지역이 커버하는 거리안에서 적어도 하나 이상의 설비로 부터 커버가 된다. 
-(3) : 총 설치할 설비는 K개 이다.

``` python3 
def generate_candidate_sites(points,M=100):
    '''
    Generate M candidate sites with the convex hull of a point set
    Input:
        points: a Numpy array with shape of (N,2)
        M: the number of candidate sites to generate
    Return:
        sites: a Numpy array with shape of (M,2)
    '''
    hull = ConvexHull(points)
    polygon_points = points[hull.vertices]
    poly = Polygon(polygon_points)
    min_x, min_y, max_x, max_y = poly.bounds
    sites = []
    while len(sites) < M:
        random_point = Point([random.uniform(min_x, max_x),
                             random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            sites.append(random_point)
    return np.array([(p.x,p.y) for p in sites])




def generate_candidate_sites(df_result_fin,M=100):
    from shapely.geometry import Polygon, Point
    sites = []
    idx=np.random.choice(np.array(range(0,len(df_result_fin))), M)
    for i in range(len(idx)):
        random_point = Point(np.array(df_result_fin.iloc[idx]['coord_cent'])[i][0],
                             np.array(df_result_fin.iloc[idx]['coord_cent'])[i][1])
        sites.append(random_point)
    return np.array([(p.x,p.y) for p in sites])

def generate_candidate_sites(df_result_fin,Weight,M=100):
    sites = []
    idx = df_result_fin.sort_values(by = Weight, ascending = False).iloc[1:M].index
    for i in range(len(idx)):
        random_point = Point(np.array(df_result_fin.loc[idx]['coord_cent'])[i][0],
                             np.array(df_result_fin.loc[idx]['coord_cent'])[i][1])
        sites.append(random_point)
    return np.array([(p.x,p.y) for p in sites])



from scipy.spatial import distance_matrix
def mclp(points,K,radius,M,df_result_fin,w,Weight):

    """
    Solve maximum covering location problem
    Input:
        points: input points, Numpy array in shape of [N,2]
        K: the number of sites to select
        radius: the radius of circle
        M: the number of candidate sites, which will randomly generated inside
        the ConvexHull wrapped by the polygon
    Return:
        opt_sites: locations K optimal sites, Numpy array in shape of [K,2]
        f: the optimal value of the objective function
    """
    print('----- Configurations -----')
    print('  Number of points %g' % points.shape[0])
    print('  K %g' % K)
    print('  Radius %g' % radius)
    print('  M %g' % M)
    import time
    start = time.time()
    sites = generate_candidate_sites(df_result_fin,Weight,M)
    J = sites.shape[0]
    I = points.shape[0]
    D = distance_matrix(points,sites)
    mask1 = D<=radius
    D[mask1]=1
    D[~mask1]=0

    from mip import Model, xsum, maximize, BINARY

    # Build model
    m = Model("mclp")
    # Add variables

    x = [m.add_var(name = "x%d" % j, var_type = BINARY) for j in range(J)]
    y = [m.add_var(name = "y%d" % i, var_type = BINARY) for i in range(I)]


    m.objective = maximize(xsum(w[i]*y[i] for i in range (I)))

    m += xsum(x[j] for j in range(J)) == K

    for i in range(I):
        m += xsum(x[j] for j in np.where(D[i]==1)[0]) >= y[i]

    m.optimize()
    
    end = time.time()
    print('----- Output -----')
    print('  Running time : %s seconds' % float(end-start))
    print('  Optimal coverage points: %g' % m.objective_value)

    solution = []
    for i in range(J):
        if x[i].x ==1:
            solution.append(int(x[i].name[1:]))
    opt_sites = sites[solution]
            
    return opt_sites,m.objective_value
```