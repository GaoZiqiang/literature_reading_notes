### 00 文章基本信息

 - 文章来源：IJCAI-20
 - 作者信息：![在这里插入图片描述](https://img-blog.csdnimg.cn/20210228221347942.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzNDI5OTY4,size_16,color_FFFFFF,t_70#pic_center)

### 01 摘要

 - POI recommandation目前存在的不足： none of existing methods utilize
   similarity explicitly to make recommendations.
 - 我们的contributions： we propose a new framework for POI recommendation,
   which explicitly utilizes similarity with contextual information.

## 1 Introduction
Point of interest (POI) recommendation的目标：
aims to find new places for users that they might be interested in.

应用场景：

 - help users find interesting spots that will make them enjoy their
   vacations when they are in unfamiliar regions.
 - increase the shopkeepers’income by attracting more customers who
   would like to spend time and money at the store.

**目前存在的问题和挑战：（要解决的问题）**
one of the most challenging one is the **data sparsity problem**.

以前的解决方法：
To tackle this problem, many methods **incorporate the contextual information into the recommendation method** with different **assumptions**.
以往方法的共同特点：
the common property behind them is that similar users should visit similar POIs and similar POIs should be visited by similar users.
存在的不同点：
the only difference between these assumptions is the way they construct the similarity
以往方法存在的弊端：可拓展性（extensibility.）低以及上下文信息利用不充分（explicity）

 - they usually consider only one type of contextual information in one entity.
 - they design the model specially for a specific type of context, making the models **lack extensibility**.
 - they are not utilizing the contextual information explicitly, as most
   of the models focuse on check-in history, making contextual
   information utilization only to be an accessory component in the
   objective function.Making contextual information cannot be fully
   utilized.

为了提高模型的可拓展性和更充分地利用上下文信息，我们提出了一个新的POI recommendation方法，该方法的主要创新点如下：

 - we construct one **user matrix** and one **POI similarity matrix**
   according to the corresponding user and POI contextual information. A
   lot of types of similarity can be computed by cosine similarity
   between feature vectors between two entities.
 - different types of similarity can be combined as a weighted sum.(our
   framework is extensible for a large class of contextual information)
 - Once the similarity matrices of users and POIs are constructed, we
   will use two global Laplacian regularization terms to constrain the
   predicted preference matrix.（These can directly make sure that, in
   the final prediction, similar users should visit similar POIs and
   similar POIs should be visited by similar users.）
 - to exploit the contextual information hierarchically, we also impose
   a local regularizer to make the predicted preference matrix have
   local patterns.
 - Based on user similarity, we use spectral clustering to sort users
   into different groups. Then we impose an l 2 -norm as a
   regularization term for the predicted preference matrix of every
   group（which can make the preference of users in the same group to be
   sparse and have similar patterns.）

关于目标函数objective function：
We form the objective function by putting the global and local **regularizers** together.
如何优化目标函数：

 - To solve this optimization problem efficiently, we propose an
   alternating optimization method which decompose the objective
   function into two parts with an auxiliary variable.
 - Accelerated proximal grdient (APG) algorithm is used to optimize the
   l2 -regularized part of the problem.


**文章的贡献contributions：**

 - We propose a new framework for POI recommendation, which focuses on
   the explicit utilization of contextual information;
 - We can utilize different types of contextual information of both user
   and POI in our method;
 - We categorize contextual information into global and local types and
   utilize them by different regularization terms respectively;
 - We design an alternating optimization method to optimize the model;
 - The results of our method outperform the state-of-the-art methods on
   two large datasets.

## 2 Related Works
相关方法主要有三大类：

 - One group of methods exploit the user-based context.
 - Another group of methods focus on exploiting the POI-based context.
 - there also exist some methods exploit hybrid context for recommendation.

## 3 Preliminaries
这部分简述POI推荐问题和图Laplacian正则化的背景知识。

### 3.1 Problem Definition
相关定义：

 - the set of users： U = {u 1 , u 2 , . . . , u m }
 - the set of POIs： V = {v 1 , v 2 , . . . , v n }
 - 用户u访问过的POIs：Pu
 - POI recommendation的工作： 为每一个用户推荐一个新的POIs索引集P̂ u（P u 交P̂ u = ∅）来满足用户的偏好需求preference。
 - transaction history D：a set of tuples of the users and their visited
   POIs, i.e., D = {(u, v)|u ∈ U, v ∈ V }.
 - contextual information of the users：e.g., social relations
 - contextual information of users' visited POIs：e.g., geographical
   coordinates
### 3.2 Graph Laplacian Regularization
相关定义：
 - weighted undirected graph无向加权图：G
 - vertex set顶点集：V = {ν 1 , ν 2 , . . . , ν |V| }
 - weight matrix：W = [W ij ] i,j=1...|V|，W ij denotes the weight of edge
   between ν i and ν j .W为对称矩阵.
 - degree matrix D of G：D = diag (d 1 , d 2 , . . . , d |V|)，where d i =
   j=1 W ij . normalized Laplacian matrix of G：
 - the normalized Laplacian matrix of G：
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210228171521501.png#pic_center)
 - Let f : V → R be a realvalued function defined on the vertex space V.
 - The normalized Laplacian regularization of f on graph G is defined as
   the quadratic form：
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210228171654449.png#pic_center)
   其中，![在这里插入图片描述](https://img-blog.csdnimg.cn/20210228171735407.png#pic_center)

## 4 The Proposed Recommendation Model
本文提出的模型/方法的目标是：
predict a **rating matrix** R ∈ R m×n by optimizing an objective function.其中，Each element R i,j in R represents the inferred preference of user u i over POI v j.然后，**The new POIs for user u i are then recommended based on the values of R i,1 , R i,2 , . . . , R i,n .**

关于目标函数：
The objective function includes three regularization terms on R，并且each of them will explicitly utilize **user-based global** contextual information, **POI-based global** contextual information, and **local** contextual information, respectively.
### 4.1 Exploiting Global Context Information
**1.User-based Global Context**

先声明几个定义：

 - G user：a weighted undirected graph，其中，the vertex set is the user set
   U，The edge weights are given by a symmetric weight matrix
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210228201452697.png#pic_center)
   另外， ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210228201552665.png)
   is the similarity between user i and user j.
 - ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210228201704745.png) ：the
   degree matrix of G user.
 - Luser = 
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210228201804351.png) ：the
   normalized Laplacian matrix of G user.
 - R：rating matrix，其中R ij represents the rating of user u i on POI v j

然后我们可以得到：

 - the normalized graph Laplacian regularization of R on G user for a
   particular POI v
   j：![在这里插入图片描述](https://img-blog.csdnimg.cn/20210228202222273.png#pic_center)
   ，其中“:” indicates taking all items of a row/column.
 - 如果sum all the normalized graph Laplacian regularization of R on graph
   G user for all POIs，我们可以得到：
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210228202415206.png#pic_center)
   该式可以作为 the user-based Laplacian regularization term for the rating
   matrix R.
   
**2.POI-based Global Context**
POI-based Global Context与上述User-based Global Context的定义类似，the POI-based Laplacian regularization term for the rating matrix R的定义如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210228210750297.png#pic_center)
### 4.2 Exploiting Local Context Information
this section takes into account **local** contextual information between users and POIs.
与global context information的不同之处：
the local regularizer divide users into groups, therefore making the similarity “local”.(实现方法：spectral clustering谱聚类)

下面是两个定义：
R (g),j：the rating of POI v j by the users in g-th cluste.
local regularizer J(R)：![在这里插入图片描述](https://img-blog.csdnimg.cn/20210228213127916.png#pic_center)
the final objective function：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210228213244152.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzNDI5OTY4,size_16,color_FFFFFF,t_70#pic_center)
optimal rating matrix：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210228213533748.png#pic_center)
### 4.3 Similarity Graph Construction
声明：
我们可以在数据集中获得以下信息：

 - the check-in transactions of each user;
 - the check-in time of each user at each POI;
 - the social relations between users;
 - the geographic coordinates of the POIs;

然后我们可以构建the user similarity graph和the POI similarity graph。

**1.User Similarity Graph Construction**

The user similarity graph G user的构建基于以下假设：

 - Users that visit similar set of POIs at similar periods of a day are
   more similar (spatial-temporal similarity);
 - The users having social relationships are more similar (social
   relation similarity);

下面几个定义：

 - feature vector
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210228214709343.png)
   ：whose entries records the number of check-ins of user u i at each
   POI in each hour of a day（because the number of POIs is n and there
   are 24 hours in a day, the dimension of f chkin-u is 24 × n）
 - ![在这里插入图片描述](https://img-blog.csdnimg.cn/2021022821481746.png) ：The
   spatial-temporal similarity between two users, u i and u j.
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210228214859848.png#pic_center)
   由此得到the spatial-temporal similarity matrix as S ST.
 - ![在这里插入图片描述](https://img-blog.csdnimg.cn/2021022821500171.png)
   ：records the **social SR** relations between users.
 - the user similarity matrix：
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210228215125575.png#pic_center#pic_center)

**2.POI Similarity Graph Construction**
与上述User Similarity Graph Construction类似，只给出最终的the POI similarity matrix：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021022821575426.png#pic_center)
### 4.4 Optimization
算法优化分为user step和user step两部分，具体过程不详细展开。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210228220751159.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzNDI5OTY4,size_16,color_FFFFFF,t_70)

## 5 Experiments
略。
## 6 Conclusion
In this paper, we designed a new framework for POI recommendation, which explicitly exploits **global contextual information** of users and POIs through Laplacian regularization and **local contextual information** through a local regularizer.

